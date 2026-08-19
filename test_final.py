import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from src.agent import run_agent

# Test 1: Dataset overview
print("=== TEST 1: Dataset overview ===")
r = run_agent("What is this dataset about?", max_iterations=3)
print(f"Answer: {r['answer'][:300]}")
print(f"Tools: {[tc['tool'] for tc in r['tool_calls_made']]}")
print(f"Critic: {r['critic_issues']}")
print()

# Test 2: Prediction
print("=== TEST 2: Customer prediction ===")
r = run_agent("Predict risk for customer 0376-YMCJC", max_iterations=4)
print(f"Answer: {r['answer'][:300]}")
print(f"Tools: {[tc['tool'] for tc in r['tool_calls_made']]}")
print()

# Test 3: Top churn customers
print("=== TEST 3: Top churn customers ===")
r = run_agent("Which customers are most likely to churn?", max_iterations=4)
print(f"Answer: {r['answer'][:400]}")
print(f"Tools: {[tc['tool'] for tc in r['tool_calls_made']]}")
