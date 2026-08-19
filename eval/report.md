# Agent Evaluation Report

**Generated:** 2026-08-19 22:32:40

## Summary

| Metric | Value |
|---|---|
| Questions Tested | 12 |
| Passed | 9 |
| Failed | 3 |
| **Accuracy** | **75.0%** |
| Numeric Hallucinations | 7 |
| Total Numbers in Answers | 40 |
| Hallucination Rate | 17.5% |
| Failed Tool Calls | 1 |

## Per-Question Results

| # | Category | Type | Pass | Reason | Tool Calls | Hall. Numbers | Time |
|---|---|---|---|---|---|---|---|
| 1 | EDA | numeric | Pass | numeric match | 1 | 869.0, 7.0, 32.0 | 6.5s |
| 2 | EDA | category | Pass | keyword match | 1 | - | 44.4s |
| 3 | EDA | numeric | Pass | numeric match | 1 | - | 69.2s |
| 4 | EDA | numeric | Pass | numeric match | 1 | - | 28.3s |
| 5 | EDA | category | Pass | keyword match | 1 | - | 31.0s |
| 6 | Model | category | Fail | keyword mismatch | 0 | - | 22.1s |
| 7 | Model | list | Fail | list check fail | 1 | 46.05 | 87.8s |
| 8 | Model | numeric | Pass | numeric match | 1 | - | 65.4s |
| 9 | Segment | numeric | Pass | numeric match | 1 | - | 30.7s |
| 10 | Segment | numeric | Fail | numeric mismatch | 1 | 175.0 | 43.3s |
| 11 | Agent Reasoning | multi_step | Pass | multi-step pass | 1 | 22.9 | 33.2s |
| 12 | Agent Reasoning | multi_step | Pass | multi-step pass | 1 | 52.83 | 34.5s |

## Failed Questions

- **Q6** (Model): keyword mismatch — *Predict the churn risk for customer 0376-YMCJC. What is their risk level?*
- **Q7** (Model): list check fail — *List the top 3 customers most likely to churn based on risk indicators.*
- **Q10** (Segment): numeric mismatch — *How many customers in the dataset have tenure of 12 months or less?*

## Observations

**Strengths:**
- EDA queries (churn rate, category breakdowns) are answered accurately using dataset_summary and churn_by_category tools.
- Model predictions for individual customers and hypothetical scenarios are correctly routed through the predict_customer and predict_hypothetical tools.
- Segment risk analysis via aggregate_segment_risk produces consistent numerical results.
- The critic agent flags potential issues before returning final answers.

**Limitations:**
- Multi-step reasoning questions may occasionally fail if the LLM exhausts its iteration budget.
- Free-tier LLM rate limits can cause transient failures on longer eval runs.
- Hallucination detection is heuristic — it checks whether answer numbers appear in tool outputs, but some derived numbers (e.g. ratios computed by the LLM from tool data) may be flagged as false positives.

## Methodology

- **Accuracy**: A question passes if its answer meets type-specific criteria (numeric within tolerance, keyword match, list length, multi-step completeness).
- **Hallucination Detection**: Numbers in the final answer (>5) that cannot be traced back to any tool call output are flagged as potential hallucinations.
- **Tool Call Check**: Questions that produced no tool calls despite requiring computation are noted.
