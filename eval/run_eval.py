import json
import os
import re
from datetime import datetime

from src.agent import run_agent

ROOT = os.path.dirname(os.path.dirname(__file__))
EVAL_SET_PATH = os.path.join(ROOT, "eval", "eval_set.json")
REPORT_PATH = os.path.join(ROOT, "eval", "eval_report.md")


def _extract_numbers(text):
    nums = []
    for m in re.findall(r"-?\d+\.?\d*", text or ""):
        try:
            nums.append(float(m))
        except ValueError:
            continue
    return nums


def _numbers_from_tool_calls(tool_calls):
    joined = json.dumps(tool_calls, default=str)
    return _extract_numbers(joined)


def _hallucinated_numbers(answer, tool_calls, tolerance=0.05):
    answer_numbers = _extract_numbers(answer)
    tool_numbers = _numbers_from_tool_calls(tool_calls)

    if not answer_numbers:
        return []
    if not tool_numbers:
        return answer_numbers

    missing = []
    for n in answer_numbers:
        matched = any(abs(n - t) <= tolerance for t in tool_numbers)
        if not matched:
            missing.append(n)
    return missing


def _case_accuracy(case, result):
    answer = (result.get("answer") or "").lower()
    keywords = [k.lower() for k in case.get("expected_keywords", [])]
    keyword_hits = sum(1 for k in keywords if k in answer)
    keyword_ok = keyword_hits >= max(1, len(keywords) // 2)

    min_tools = int(case.get("min_tool_calls", 1))
    tool_ok = len(result.get("tool_calls_made", [])) >= min_tools

    no_error = "error" not in answer
    return keyword_ok and tool_ok and no_error, {
        "keyword_hits": keyword_hits,
        "keyword_total": len(keywords),
        "tool_calls": len(result.get("tool_calls_made", [])),
        "no_error": no_error,
    }


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    rows = []
    passed = 0
    total_hallucinated_numbers = 0
    total_answer_numbers = 0

    for case in eval_set:
        result = run_agent(case["question"], chat_history=[])
        answer = result.get("answer", "")
        tool_calls = result.get("tool_calls_made", [])

        ok, details = _case_accuracy(case, result)
        if ok:
            passed += 1

        hallucinated = _hallucinated_numbers(answer, tool_calls)
        answer_numbers = _extract_numbers(answer)
        total_hallucinated_numbers += len(hallucinated)
        total_answer_numbers += len(answer_numbers)

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "passed": ok,
            "tool_calls": details["tool_calls"],
            "keyword_hits": f"{details['keyword_hits']}/{details['keyword_total']}",
            "hallucinated_numbers": hallucinated,
            "critic_issues": result.get("critic_issues", []),
        })

    accuracy = (passed / len(eval_set)) * 100 if eval_set else 0.0
    hallucination_rate = (
        (total_hallucinated_numbers / total_answer_numbers) * 100
        if total_answer_numbers > 0
        else 0.0
    )

    lines = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Total questions: {len(eval_set)}")
    lines.append(f"Accuracy (rule-based pass rate): {accuracy:.2f}%")
    lines.append(
        f"Hallucination rate (numeric claims not found in tool outputs): {hallucination_rate:.2f}%"
    )
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| ID | Pass | Tool Calls | Keyword Hits | Hallucinated Numbers | Critic Issues |")
    lines.append("|---|---|---:|---:|---|---|")

    for r in rows:
        hall = ", ".join(str(x) for x in r["hallucinated_numbers"]) or "None"
        critic = "; ".join(r["critic_issues"]) or "None"
        lines.append(
            f"| {r['id']} | {'Yes' if r['passed'] else 'No'} | {r['tool_calls']} | {r['keyword_hits']} | {hall} | {critic} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Accuracy here is a practical QA pass rate (keywords + tool-use + no error).")
    lines.append("- Hallucination rate is computed only on numeric claims in final answers.")
    lines.append("- This eval set intentionally excludes graph/visualization prompts.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Eval complete. Report written to: {REPORT_PATH}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Hallucination rate: {hallucination_rate:.2f}%")


if __name__ == "__main__":
    main()
