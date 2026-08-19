import os
import streamlit as st
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Churn Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Churn Analyst Agent")
st.caption("Ask questions about customer churn data and get real, computed answers.")

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please set it in your .env file or environment.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.markdown("""
    This agent analyzes a customer churn dataset with **7,032 customers**.

    **You can ask about:**
    - Dataset overview and EDA
    - Specific customer churn risk
    - Hypothetical customer scenarios
    - Segment-level churn analysis
    - Correlations and trends

    **Example questions:**
    - "Which customers are most likely to churn?"
    - "Does churn risk correlate with contract type?"
    - "Predict risk for customer 0376-YMCJC"
    - "What factors drive churn the most?"
    - "Show me segment analysis for fiber optic customers"
    """)

    st.divider()
    st.subheader("Dataset Summary")
    try:
        from src.tools import dataset_summary_tool
        summary_json = dataset_summary_tool()
        summary = json.loads(summary_json)
        st.metric("Total Customers", summary["total_customers"])
        st.metric("Churn Rate", f"{summary['churn_rate']}%")
        st.metric("Features", len(summary["columns"]))
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "critic_issues" in msg and msg["critic_issues"]:
            with st.expander("Critic feedback", expanded=False):
                for issue in msg["critic_issues"]:
                    st.warning(issue)

        if "tool_calls" in msg and msg["tool_calls"]:
            with st.expander(
                f"🔧 {len(msg['tool_calls'])} tool call(s)", expanded=False
            ):
                for tc in msg["tool_calls"]:
                    result_preview = json.dumps(
                        tc.get("result", {}), indent=2, default=str
                    )
                    if len(result_preview) > 500:
                        result_preview = result_preview[:500] + "..."
                    st.code(
                        f"Tool: {tc['tool']}\nArgs: {json.dumps(tc.get('args', {}), indent=2)}\nResult: {result_preview}",
                        language="python",
                    )

if prompt := st.chat_input("Ask a question about customer churn..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history = []
    for msg in st.session_state.messages[:-1]:
        chat_history.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Thinking and computing..."):
            try:
                from src.agent import run_agent
                result = run_agent(prompt, chat_history=chat_history)
            except Exception as e:
                result = {
                    "answer": f"Sorry, I encountered an error: {str(e)}. Please try again.",
                    "tool_calls_made": [],
                    "critic_issues": [],
                }

        st.markdown(result["answer"])

        if result.get("critic_issues"):
            with st.expander("Critic flagged issues", expanded=True):
                for issue in result["critic_issues"]:
                    st.warning(issue)

        if result["tool_calls_made"]:
            with st.expander(
                f"🔧 {len(result['tool_calls_made'])} tool call(s)", expanded=False
            ):
                for tc in result["tool_calls_made"]:
                    result_preview = json.dumps(
                        tc.get("result", {}), indent=2, default=str
                    )
                    if len(result_preview) > 500:
                        result_preview = result_preview[:500] + "..."
                    st.code(
                        f"Tool: {tc['tool']}\nArgs: {json.dumps(tc.get('args', {}), indent=2)}\nResult: {result_preview}",
                        language="python",
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool_calls": result["tool_calls_made"],
        "critic_issues": result.get("critic_issues", []),
    })
