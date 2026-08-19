"""Test all core modules."""
from src.data import load_clean_data
from src.model import predict_churn_risk
from src.tools import dataset_summary_tool, execute_code_tool, aggregate_segment_risk_tool

df = load_clean_data()
print("=== Data OK ===")
print(dataset_summary_tool()[:200])

print("\n=== Prediction OK ===")
cust = df[df["customerID"] == "0376-YMCJC"]
r = predict_churn_risk(cust)
print(r)

print("\n=== Code Exec OK ===")
code_result = execute_code_tool(
    'result = df.groupby("Contract")["Churn"].apply(lambda x: (x=="Yes").mean()*100)'
)
print(code_result[:300])

print("\n=== Segment Risk OK ===")
segment_result = aggregate_segment_risk_tool('{"Contract": "Month-to-month"}')
print(segment_result[:300])
