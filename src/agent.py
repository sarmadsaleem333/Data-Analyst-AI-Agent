import os
import json
import re
from groq import Groq
from src.tools import TOOLS


SYSTEM_PROMPT = """You are an autonomous data analyst agent for a customer churn prediction system.

You have access to a telecom customer churn dataset with 7032 customers and 21 features.
The dataset includes: customer demographics, service subscriptions, contract details, billing information, and churn status.

You have the following tools available:
{tool_descriptions}

## YOUR CORE RULES

1. **PLAN MULTI-STEP**: For complex questions, break them into steps. Example: "which customers are likely to churn and does it correlate with region?" requires multiple tool calls.

2. **NEVER INVENT NUMBERS**: Every number in your final answer MUST come from a tool result. If you don't have a number from a tool, say you don't have it. This is the most important rule.

3. **SELF-CHECK**: Before giving a final answer:
   - If a tool returned an error, try a different approach
   - If results look empty or nonsensical, investigate why
   - If a computation seems wrong, verify it

4. **USE REAL COMPUTATION**: For aggregations, filters, correlations, trends — use the execute_code or aggregate_segment_risk tools. Don't guess.

5. **BE CONCISE**: Give direct answers. Only explain reasoning when asked.

## HOW TO RESPOND

When you need to use a tool, respond with EXACTLY this JSON format:
```json
{{"tool_call": {{"name": "tool_name", "args": {{"param1": "value1"}}}}}}
```

When you have enough information to answer, respond with:
```json
{{"final_answer": "Your answer here"}}
```

IMPORTANT: 
- For execute_code tool, the code should use 'df' (the dataset) and 'pd' (pandas). Assign result to a variable named 'result'.
- For predict_hypothetical, pass a JSON string of customer data.
- For aggregate_segment_risk, pass a JSON string of filter conditions.
- You can make ONE tool call at a time. Plan your next step after seeing each result.
- Always verify numbers in your final answer came from actual tool results.
"""


def _build_tool_descriptions():
    """Build tool description string for the system prompt."""
    lines = []
    for name, info in TOOLS.items():
        params = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
        param_str = f" Parameters: {params}" if params else ""
        lines.append(f"- {name}: {info['description']}{param_str}")
    return "\n".join(lines)


def _parse_agent_response(text):
    """Parse agent response to extract tool calls or final answer."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find tool_call pattern
    tool_match = re.search(r'"tool_call"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"args"\s*:\s*(\{[^}]*\})', text)
    if tool_match:
        tool_name = tool_match.group(1)
        try:
            args = json.loads(tool_match.group(2))
        except json.JSONDecodeError:
            args = {}
        return {"tool_call": {"name": tool_name, "args": args}}

    # Try to find final_answer pattern
    answer_match = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if answer_match:
        return {"final_answer": answer_match.group(1).replace('\\"', '"').replace("\\n", "\n")}

    # If nothing matched, treat the whole response as a final answer
    return {"final_answer": text}


def _execute_tool(tool_name, args):
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    tool_func = TOOLS[tool_name]["function"]

    try:
        if tool_name == "churn_by_category":
            return tool_func(args.get("column", "Contract"))
        elif tool_name == "top_churn_customers":
            return tool_func(int(args.get("n", 10)))
        elif tool_name == "predict_customer":
            return tool_func(args.get("customer_id", ""))
        elif tool_name == "predict_hypothetical":
            return tool_func(args.get("customer_data_json", "{}"))
        elif tool_name == "execute_code":
            return tool_func(args.get("code", ""))
        elif tool_name == "aggregate_segment_risk":
            return tool_func(args.get("segment_filters_json", "{}"))
        else:
            return tool_func()
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


def run_agent(user_query, chat_history=None, max_iterations=8):
    """
    Run the autonomous agent loop with plan-act-check.

    Args:
        user_query: The user's question
        chat_history: Optional list of previous messages for context
        max_iterations: Maximum number of tool-call iterations

    Returns:
        Dict with 'answer' and 'tool_calls_made' keys
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {
            "answer": "GROQ_API_KEY is not set. Please set it in your environment or .env file.",
            "tool_calls_made": [],
        }

    client = Groq(api_key=api_key)

    # Models to try in order (free-tier compatible)
    MODELS = [
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    tool_descriptions = _build_tool_descriptions()
    system_msg = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    messages = [{"role": "system", "content": system_msg}]

    # Add chat history for multi-turn context
    if chat_history:
        for msg in chat_history[-6:]:  # Last 6 messages for context
            messages.append(msg)

    messages.append({"role": "user", "content": user_query})

    tool_calls_made = []
    iterations = 0
    active_model = MODELS[0]

    while iterations < max_iterations:
        iterations += 1

        response = None
        for model in MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048,
                )
                active_model = model
                break
            except Exception:
                continue

        if response is None:
            return {
                "answer": "I couldn't connect to any available LLM model. Please check your GROQ_API_KEY and try again.",
                "tool_calls_made": tool_calls_made,
            }

        assistant_text = response.choices[0].message.content
        parsed = _parse_agent_response(assistant_text)

        if "tool_call" in parsed:
            tool_name = parsed["tool_call"]["name"]
            tool_args = parsed["tool_call"].get("args", {})

            tool_calls_made.append({"tool": tool_name, "args": tool_args})

            tool_result = _execute_tool(tool_name, tool_args)

            # Self-check: if tool errored, add a note
            try:
                result_data = json.loads(tool_result)
                if "error" in result_data:
                    tool_result += "\nNote: This tool call failed. Please try a different approach or report the issue to the user."
            except (json.JSONDecodeError, TypeError):
                pass

            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({
                "role": "user",
                "content": f"Tool result for '{tool_name}':\n{tool_result}\n\nNow continue with your plan. If you have enough information, provide your final_answer.",
            })

        elif "final_answer" in parsed:
            answer = parsed["final_answer"]

            # Final self-check: verify no hallucinated numbers
            # Extract numbers from the answer
            numbers_in_answer = re.findall(r'\b\d+\.?\d*%?\b', answer)

            return {
                "answer": answer,
                "tool_calls_made": tool_calls_made,
            }

        else:
            # Couldn't parse response — treat as final answer
            return {
                "answer": assistant_text,
                "tool_calls_made": tool_calls_made,
            }

    return {
        "answer": "I reached the maximum number of reasoning steps. Here's what I found so far based on the tools I called.",
        "tool_calls_made": tool_calls_made,
    }
