import os
import json
import re
import base64
from groq import Groq
from src.tools import TOOLS


SYSTEM_PROMPT = """You are an autonomous data analyst agent for a customer churn prediction system.

You have access to a telecom customer churn dataset with 7032 customers and 21 features.
The dataset includes: customer demographics, service subscriptions, contract details, billing information, and churn status.

You have the following tools available:
{tool_descriptions}

## YOUR CORE RULES

1. **PLAN MULTI-STEP**: For complex questions, break them into steps.
2. **NEVER INVENT NUMBERS**: Every number in your final answer MUST come from a tool result.
3. **SELF-CHECK**: Before giving a final answer, verify results make sense.
4. **USE REAL COMPUTATION**: For aggregations, filters, correlations — use execute_code or aggregate_segment_risk tools.
5. **GENERATE CHARTS**: When the user asks to "show", "visualize", "plot", or "chart", use the generate_chart tool.
6. **BE CONCISE**: Give direct answers.

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
- For execute_code: code should use 'df' (the dataset) and 'pd' (pandas). Assign result to 'result'.
- For predict_hypothetical: pass a JSON string of customer data.
- For aggregate_segment_risk: pass a JSON string of filter conditions.
- For generate_chart: pass JSON like: {{"chart_type":"bar","x_col":"Contract","title":"Churn by Contract"}}
- You can make ONE tool call at a time.
- Always verify numbers in your final answer came from actual tool results.
"""

CRITIC_PROMPT = """You are a critic agent that validates answers about a customer churn dataset.

The dataset has 7032 customers. The overall churn rate is approximately 26.6%.

Given the analyst's answer and the tool calls made, check for:
1. Are there specific numbers in the answer that don't appear in any tool call results?
2. Does the answer contradict known facts?
3. Is the answer responsive to the question?

Respond with EXACTLY this JSON:
```json
{{"valid": true, "issues": [], "suggested_fix": ""}}
```
If invalid, set valid to false, list issues, and provide a corrected answer.
"""

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": v}
                    for k, v in info["parameters"].items()
                },
                "required": list(info["parameters"].keys()),
            },
        },
    }
    for name, info in TOOLS.items()
]


def _build_tool_descriptions():
    """Build tool description string for the system prompt."""
    lines = []
    for name, info in TOOLS.items():
        params = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
        param_str = f" Parameters: {params}" if params else ""
        lines.append(f"- {name}: {info['description']}{param_str}")
    return "\n".join(lines)


def _strip_think_tags(text):
    """Remove <think>...</think> blocks (even unclosed) from model responses."""
    # First try to remove complete think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # If <think> exists without a closing tag, strip from <think> to end
    if '<think>' in text:
        text = text.split('<think>')[0]
    return text.strip()


def _parse_agent_response(text):
    """Parse agent response to extract tool calls or final answer."""
    text = _strip_think_tags(text)

    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    tool_match = re.search(
        r'"tool_call"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})',
        text,
    )
    if tool_match:
        try:
            tool_obj = json.loads(tool_match.group(1))
            return {"tool_call": tool_obj}
        except json.JSONDecodeError:
            pass

    answer_match = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if answer_match:
        return {
            "final_answer": answer_match.group(1)
            .replace('\\"', '"')
            .replace("\\n", "\n")
        }

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
        elif tool_name == "generate_chart":
            return tool_func(args.get("chart_config_json", "{}"))
        else:
            return tool_func()
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


def _run_critic(client, model, user_query, answer, tool_calls_made):
    """Run the critic agent to validate the analyst's answer."""
    tool_summary = []
    for tc in tool_calls_made:
        tool_summary.append(f"- {tc['tool']}({json.dumps(tc.get('args', {}))})")
    tools_str = "\n".join(tool_summary) if tool_summary else "No tools were called."

    critic_msg = f"""Original question: {user_query}

Analyst's answer:
{answer}

Tool calls made:
{tools_str}

Please validate this answer. Respond with the JSON format specified."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": critic_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
            # No tools passed — critic must respond with plain JSON
        )

        choice = response.choices[0]
        # If model used native tool calling anyway, extract the text
        if choice.message.tool_calls:
            # Model tried to call a tool during validation — treat as valid
            return True, [], ""
        critic_text = choice.message.content or ""
        parsed = _parse_agent_response(critic_text)

        if isinstance(parsed, dict) and "valid" in parsed:
            return (
                parsed.get("valid", True),
                parsed.get("issues", []),
                parsed.get("suggested_fix", ""),
            )
    except Exception:
        pass

    return True, [], ""


def _extract_charts_from_tool_calls(tool_calls_made):
    """Extract base64 chart images from tool call results."""
    charts = []
    for tc in tool_calls_made:
        if tc.get("tool") == "generate_chart" and "result" in tc:
            result = tc["result"]
            if isinstance(result, dict) and result.get("type") == "chart":
                charts.append(result.get("data", ""))
    return charts


def run_agent(user_query, chat_history=None, max_iterations=8):
    """
    Run the autonomous agent loop with plan-act-check + critic.

    Returns:
        Dict with 'answer', 'tool_calls_made', 'charts', 'critic_issues' keys
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {
            "answer": "GROQ_API_KEY is not set. Please set it in your environment or .env file.",
            "tool_calls_made": [],
            "charts": [],
            "critic_issues": [],
        }

    client = Groq(api_key=api_key)

    # openai/gpt-oss-20b: native tool calling (reliable)
    # qwen/qwen3.6-27b: manual JSON (may be rate-limited)
    MODELS = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ]

    tool_descriptions = _build_tool_descriptions()
    system_msg = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    messages = [{"role": "system", "content": system_msg}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    messages.append({"role": "user", "content": user_query})

    tool_calls_made = []
    iterations = 0
    active_model = MODELS[0]

    def _call_llm(msgs, use_native_tools=False):
        """Try models in order, return (response, model_name, parsed_result)."""
        nonlocal active_model
        for model in MODELS:
            try:
                kwargs = {
                    "model": model,
                    "messages": msgs,
                    "temperature": 0.1,
                    "max_tokens": 2048,
                }
                if use_native_tools and model == "openai/gpt-oss-20b":
                    kwargs["tools"] = NATIVE_TOOLS

                resp = client.chat.completions.create(**kwargs)
                active_model = model

                choice = resp.choices[0]

                # Handle native tool calling (openai/gpt-oss-20b)
                if choice.message.tool_calls:
                    tc = choice.message.tool_calls[0]
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}
                    return resp, model, {"tool_call": {"name": tool_name, "args": tool_args}}

                # Handle manual JSON tool calling (qwen)
                raw_text = choice.message.content or ""
                stripped = _strip_think_tags(raw_text)
                parsed = _parse_agent_response(stripped)
                return resp, model, parsed

            except Exception:
                continue

        return None, None, None

    while iterations < max_iterations:
        iterations += 1

        # First try: qwen with manual JSON. If that fails, try openai with native tools.
        response, model_name, parsed = None, None, None

        for model in MODELS:
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 2048,
                }
                if model == "openai/gpt-oss-20b":
                    kwargs["tools"] = NATIVE_TOOLS

                resp = client.chat.completions.create(**kwargs)
                active_model = model

                choice = resp.choices[0]

                # Handle native tool calling (openai/gpt-oss-20b)
                if choice.message.tool_calls:
                    tc = choice.message.tool_calls[0]
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}
                    parsed = {"tool_call": {"name": tool_name, "args": tool_args}}
                    response = resp
                    model_name = model
                    break

                # Handle manual JSON tool calling (qwen)
                raw_text = choice.message.content or ""
                stripped = _strip_think_tags(raw_text)
                parsed = _parse_agent_response(stripped)
                response = resp
                model_name = model
                break

            except Exception:
                continue

        if response is None:
            return {
                "answer": "I couldn't connect to any available LLM model. Please try again in a few minutes.",
                "tool_calls_made": [],
                "charts": [],
                "critic_issues": [],
            }

        if "tool_call" in parsed:
            tool_name = parsed["tool_call"]["name"]
            tool_args = parsed["tool_call"].get("args", {})

            tool_result = _execute_tool(tool_name, tool_args)

            tool_calls_made.append({
                "tool": tool_name,
                "args": tool_args,
                "result": json.loads(tool_result) if tool_result else {},
            })

            try:
                result_data = json.loads(tool_result)
                if "error" in result_data:
                    tool_result += (
                        "\nNote: This tool call failed. "
                        "Please try a different approach or report the issue to the user."
                    )
            except (json.JSONDecodeError, TypeError):
                pass

            messages.append({"role": "assistant", "content": response.choices[0].message.content or ""})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for '{tool_name}':\n{tool_result}\n\n"
                    "Now continue with your plan. If you have enough information, "
                    "provide your final_answer."
                ),
            })

        elif "final_answer" in parsed:
            answer = parsed["final_answer"]
            charts = _extract_charts_from_tool_calls(tool_calls_made)

            critic_valid, critic_issues, corrected = _run_critic(
                client, active_model, user_query, answer, tool_calls_made
            )

            if not critic_valid and corrected:
                retry_messages = messages.copy()
                retry_messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                })
                retry_messages.append({
                    "role": "user",
                    "content": (
                        f"A critic agent found issues with your answer:\n"
                        + "\n".join(f"- {i}" for i in critic_issues)
                        + f"\n\nPlease correct your answer. Here is a suggested fix:\n{corrected}"
                    ),
                })

                retry_resp = _call_llm(retry_messages)
                if retry_resp[0]:
                    retry_parsed = retry_resp[2]
                    if "final_answer" in retry_parsed:
                        answer = retry_parsed["final_answer"]
                        critic_valid, critic_issues, _ = _run_critic(
                            client, active_model, user_query, answer, tool_calls_made
                        )

            return {
                "answer": answer,
                "tool_calls_made": tool_calls_made,
                "charts": charts,
                "critic_issues": critic_issues if not critic_valid else [],
            }

        else:
            return {
                "answer": str(parsed),
                "tool_calls_made": tool_calls_made,
                "charts": _extract_charts_from_tool_calls(tool_calls_made),
                "critic_issues": [],
            }

    return {
        "answer": "I reached the maximum number of reasoning steps. Here's what I found so far.",
        "tool_calls_made": tool_calls_made,
        "charts": _extract_charts_from_tool_calls(tool_calls_made),
        "critic_issues": [],
    }
