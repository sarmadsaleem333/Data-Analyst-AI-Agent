import re
import json


def extract_numbers(text):
    """Extract all numeric values from a text string."""
    nums = []
    for m in re.findall(r"-?\d+\.?\d*", text or ""):
        try:
            nums.append(float(m))
        except ValueError:
            continue
    return nums


def check_numeric_accuracy(answer, expected_value, tolerance):
    """Check if any number in the answer is within tolerance of the expected value."""
    numbers = extract_numbers(answer)
    for n in numbers:
        if abs(n - expected_value) <= tolerance:
            return True
    return False


def check_numeric_range(answer, low, high, tolerance):
    """Check if any number in the answer falls within [low-tol, high+tol]."""
    numbers = extract_numbers(answer)
    for n in numbers:
        if (low - tolerance) <= n <= (high + tolerance):
            return True
    return False


def check_category(answer, expected_keywords):
    """Check if the answer mentions at least one of the expected keywords."""
    answer_lower = answer.lower()
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            return True
    return False


def check_multi_step(answer, expected_keywords):
    """Multi-step questions need both keywords and reasonable length."""
    if len(answer) < 30:
        return False
    return check_category(answer, expected_keywords)


def check_list(answer, min_items):
    """Check that the answer mentions at least min_items distinct items (e.g. customer IDs)."""
    customer_ids = re.findall(r"[A-Z0-9]{6,10}", answer)
    return len(set(customer_ids)) >= min_items


def score_answer(question, answer, tool_calls_made):
    """
    Score a single answer against the question spec.

    Returns a dict:
      passed: bool
      reason: str
      hallucinated_numbers: list of floats that appear in answer but not in tool outputs
    """
    if not answer or "error" in answer.lower()[:50]:
        return {"passed": False, "reason": "No answer or error", "hallucinated_numbers": []}

    q_type = question.get("answer_type", "category")
    hallucinated = detect_hallucinations(answer, tool_calls_made)

    if q_type == "numeric":
        if "expected_value_range" in question:
            low, high = question["expected_value_range"]
            tol = question.get("tolerance", 0.1)
            ok = check_numeric_range(answer, low, high, tol)
        else:
            ev = question.get("expected_value", 0)
            tol = question.get("tolerance", 1)
            ok = check_numeric_accuracy(answer, ev, tol)
        reason = "numeric match" if ok else "numeric mismatch"
        return {"passed": ok, "reason": reason, "hallucinated_numbers": hallucinated}

    if q_type == "category":
        kws = question.get("expected_keywords", [])
        ok = check_category(answer, kws)
        reason = "keyword match" if ok else "keyword mismatch"
        return {"passed": ok, "reason": reason, "hallucinated_numbers": hallucinated}

    if q_type == "multi_step":
        kws = question.get("expected_keywords", [])
        ok = check_multi_step(answer, kws)
        reason = "multi-step pass" if ok else "multi-step fail"
        return {"passed": ok, "reason": reason, "hallucinated_numbers": hallucinated}

    if q_type == "list":
        min_items = question.get("expected_min_items", 1)
        ok = check_list(answer, min_items)
        reason = "list check pass" if ok else "list check fail"
        return {"passed": ok, "reason": reason, "hallucinated_numbers": hallucinated}

    return {"passed": False, "reason": "unknown question type", "hallucinated_numbers": []}


def detect_hallucinations(answer, tool_calls_made, tolerance=0.05):
    """
    Lightweight hallucination check:
    Extract numbers from the final answer, then check if each
    appears (within tolerance) in the tool call outputs.
    Numbers below 5 are skipped (likely category counts, not claims).
    """
    answer_numbers = extract_numbers(answer)

    tool_text = json.dumps(tool_calls_made, default=str)
    tool_numbers = extract_numbers(tool_text)

    hallucinated = []
    for n in answer_numbers:
        if abs(n) < 5:
            continue
        if not tool_numbers:
            hallucinated.append(n)
            continue
        matched = any(abs(n - t) <= tolerance for t in tool_numbers)
        if not matched:
            hallucinated.append(n)

    return hallucinated
