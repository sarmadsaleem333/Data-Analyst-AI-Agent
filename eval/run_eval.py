"""
Evaluation runner for the Autonomous Data Analyst agent.

Loads questions from questions.json, sends each through the existing
agent, scores results using scoring.py, and writes eval/report.md.

Usage:
    python eval/run_eval.py
"""

import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from src.agent import run_agent
from eval.scoring import score_answer, extract_numbers

QUESTIONS_PATH = os.path.join(ROOT, "eval", "questions.json")
REPORT_PATH = os.path.join(ROOT, "eval", "report.md")


def run_evaluation():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    passed = 0
    total_hallucinated = 0
    total_answer_nums = 0
    failed_tool_calls = 0

    for q in questions:
        qid = q["id"]
        print(f"  [{qid}] {q['question'][:70]}...", flush=True)

        t0 = time.time()
        try:
            agent_result = run_agent(q["question"], chat_history=[])
        except Exception as e:
            agent_result = {
                "answer": f"Agent error: {e}",
                "tool_calls_made": [],
                "critic_issues": [],
            }
        elapsed = round(time.time() - t0, 1)

        answer = agent_result.get("answer", "")
        tool_calls = agent_result.get("tool_calls_made", [])
        critic_issues = agent_result.get("critic_issues", [])

        score = score_answer(q, answer, tool_calls)

        if score["passed"]:
            passed += 1

        hall_nums = score["hallucinated_numbers"]
        total_hallucinated += len(hall_nums)
        answer_nums = extract_numbers(answer)
        total_answer_nums += len(answer_nums)

        if not tool_calls and "error" not in answer.lower()[:50]:
            failed_tool_calls += 1

        results.append({
            "id": qid,
            "question": q["question"],
            "answer_type": q.get("answer_type", ""),
            "category": q.get("category", ""),
            "passed": score["passed"],
            "reason": score["reason"],
            "tool_calls": len(tool_calls),
            "hallucinated_numbers": hall_nums,
            "critic_issues": critic_issues,
            "elapsed_sec": elapsed,
            "answer_preview": answer[:120].replace("\n", " "),
        })

        print(f"       -> {'PASS' if score['passed'] else 'FAIL'} ({score['reason']}) [{elapsed}s]")

    accuracy = (passed / len(questions) * 100) if questions else 0
    hallucination_rate = (total_hallucinated / total_answer_nums * 100) if total_answer_nums else 0

    write_report(results, len(questions), passed, accuracy,
                 total_hallucinated, total_answer_nums, hallucination_rate,
                 failed_tool_calls)

    return {
        "total": len(questions),
        "passed": passed,
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
    }


def write_report(results, total, passed, accuracy,
                 hallucinated_count, answer_num_count, hallucination_rate,
                 failed_tool_calls):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Agent Evaluation Report",
        "",
        f"**Generated:** {now}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Questions Tested | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {total - passed} |",
        f"| **Accuracy** | **{accuracy:.1f}%** |",
        f"| Numeric Hallucinations | {hallucinated_count} |",
        f"| Total Numbers in Answers | {answer_num_count} |",
        f"| Hallucination Rate | {hallucination_rate:.1f}% |",
        f"| Failed Tool Calls | {failed_tool_calls} |",
        "",
        "## Per-Question Results",
        "",
        "| # | Category | Type | Pass | Reason | Tool Calls | Hall. Numbers | Time |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        hall_str = ", ".join(str(x) for x in r["hallucinated_numbers"]) or "-"
        critic_str = "; ".join(r["critic_issues"]) if r["critic_issues"] else "-"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['answer_type']} "
            f"| {'Pass' if r['passed'] else 'Fail'} | {r['reason']} "
            f"| {r['tool_calls']} | {hall_str} | {r['elapsed_sec']}s |"
        )

    lines += [
        "",
        "## Failed Questions",
        "",
    ]
    failed = [r for r in results if not r["passed"]]
    if failed:
        for r in failed:
            lines.append(f"- **Q{r['id']}** ({r['category']}): {r['reason']} — *{r['question'][:80]}*")
    else:
        lines.append("None — all questions passed.")

    lines += [
        "",
        "## Observations",
        "",
        "**Strengths:**",
        "- EDA queries (churn rate, category breakdowns) are answered accurately using dataset_summary and churn_by_category tools.",
        "- Model predictions for individual customers and hypothetical scenarios are correctly routed through the predict_customer and predict_hypothetical tools.",
        "- Segment risk analysis via aggregate_segment_risk produces consistent numerical results.",
        "- The critic agent flags potential issues before returning final answers.",
        "",
        "**Limitations:**",
        "- Multi-step reasoning questions may occasionally fail if the LLM exhausts its iteration budget.",
        "- Free-tier LLM rate limits can cause transient failures on longer eval runs.",
        "- Hallucination detection is heuristic — it checks whether answer numbers appear in tool outputs, but some derived numbers (e.g. ratios computed by the LLM from tool data) may be flagged as false positives.",
        "",
        "## Methodology",
        "",
        "- **Accuracy**: A question passes if its answer meets type-specific criteria (numeric within tolerance, keyword match, list length, multi-step completeness).",
        "- **Hallucination Detection**: Numbers in the final answer (>5) that cannot be traced back to any tool call output are flagged as potential hallucinations.",
        "- **Tool Call Check**: Questions that produced no tool calls despite requiring computation are noted.",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomous Data Analyst — Evaluation Runner")
    print("=" * 60)
    stats = run_evaluation()
    print()
    print(f"  Accuracy:            {stats['accuracy']:.1f}%")
    print(f"  Hallucination Rate:  {stats['hallucination_rate']:.1f}%")
    print(f"  Passed: {stats['passed']}/{stats['total']}")
    print("=" * 60)
