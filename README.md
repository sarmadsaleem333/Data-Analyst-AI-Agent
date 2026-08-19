# Autonomous Data Analyst for Customer Churn

## 1) Project Overview

This repository contains my implementation of the Adept Tech Solutions AI Engineer assessment (Autonomous Data Analyst).

The goal is to combine:

- a trained churn prediction model,
- a tool-using agent that can perform real computations on the dataset,
- and a Streamlit chat interface that lets users ask natural-language churn and EDA questions.

The solution is built in three stages:

- Stage 1: Train a churn model and expose it as callable functions.
- Stage 2: Build a Streamlit chat app wired to model + agent.
- Stage 3: Implement an agent loop that plans, calls tools, validates outputs, and answers with computed facts.

## 2) What Has Been Built So Far

### Stage 1: Model as a Callable Tool

Completed:

- Data cleaning and preprocessing pipeline.
- Model training and evaluation workflow in notebook and retraining script.
- Saved model artifacts for runtime inference.
- Callable prediction functions for both single-customer and batch use.

Implemented callable functionality includes:

- `predict_churn_risk(customer_df)`
- `predict_batch(customer_df)`

Model artifacts saved in `models/`:

- `churn_decision_tree.pkl`
- `churn_threshold.pkl`
- `model_features.pkl`

### Stage 2: Streamlit Chat Interface

Completed:

- A working Streamlit chat app (`app.py`) with live wiring to agent + model.
- Sidebar dataset summary and usage examples.
- Multi-turn chat history handling.
- Tool-call transparency panel (shows tool arguments and result previews).
- Error handling for missing API key and runtime exceptions.
- Chart rendering support from agent-generated base64 image outputs.

### Stage 3: Agent Implementation

Completed:

- Tool-enabled agent loop using Groq.
- Multi-step behavior through iterative tool calls.
- Real computation tools (not only LLM text generation).
- Basic self-check and critic pass before final response.
- Support for both structured tool calling and JSON-formatted tool command fallback.

Implemented tool capabilities:

- Dataset summary and descriptive stats.
- Churn rate by category.
- Correlation analysis.
- Top churn-risk customer listing.
- Customer-level churn prediction.
- Hypothetical scenario prediction.
- Segment-level aggregate churn risk.
- Restricted dataframe code execution.
- Chart generation (bar, histogram, box, scatter, heatmap, pie).

## 3) Data Cleaning Findings and Decisions

The dataset issue identified and handled:

- `TotalCharges` contains blank string values that are not numeric.

Cleaning actions taken:

1. Convert `TotalCharges` to numeric with coercion.
2. Drop rows where conversion fails (null after coercion).

Observed impact:

- 11 rows dropped (new customers with `tenure = 0` and missing `TotalCharges`).
- Final modeling dataset is aligned and usable for training.

Why this is reasonable:

- These records do not contain enough billing history for a stable charge-based signal.
- Keeping malformed numeric values would introduce noise/errors in model features.

## 4) Model Choice and Why

Models were compared in the notebook workflow. The selected final model is:

- **Decision Tree Classifier** (`max_depth=5`, `random_state=42`)

Why this model was chosen:

- It achieved the strongest recall/F1 trade-off for the churn objective.
- It is interpretable and easy to inspect for feature-level influence.
- It works well for heterogeneous categorical/numeric telecom features after encoding.

## 5) Evaluation Metric Choice and Rationale

### Chosen decision metric

Primary metric focus: **Recall for churn class**, with **F1-score** as the balancing metric.

Why this metric strategy:

- In churn use-cases, missing true churners (false negatives) is usually costlier than over-flagging some non-churners.
- Recall ensures more at-risk customers are captured.
- F1 is used to avoid pushing recall so high that precision collapses.

Additional metrics tracked:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Threshold tuning

A custom probability threshold was tuned for business alignment.

Final threshold selected:

- **0.35**

Reason:

- It produced the best precision/recall balance in threshold tests.
- Notebook notes indicate approximately:
  - Recall: **71.66%**
  - F1-score: **62.40%**

This threshold is stored and used at inference time (`churn_threshold.pkl`) so training-time logic and runtime behavior stay consistent.

## 6) Agent Planning and Verification Approach

The agent is designed as a plan-act-check loop rather than single-shot QA.

### Planning

- The model receives a system prompt requiring multi-step decomposition for complex questions.
- It can call one tool at a time, inspect outputs, then decide next step.

### Acting via tools

- For computed facts (aggregations, group stats, trends), it calls data tools or restricted code execution.
- For customer risk answers, it calls model prediction tools.
- For visualization requests, it calls chart generation.

### Verification / self-check

Current safeguards:

- Tool errors are caught and surfaced to the loop.
- Empty/invalid outputs can trigger another tool attempt.
- A critic pass validates answer quality before display.
- Prompt-level rule enforces: "Never invent numbers."

Trade-off:

- This is lightweight and practical for free-tier API usage.
- It is not yet a fully formal verifier with guaranteed numeric citation provenance per sentence.

## 7) Repository Structure (High Level)

- `app.py`: Streamlit interface.
- `src/data.py`: loading, cleaning, EDA utilities, restricted execution, charting.
- `src/model.py`: model loading, preprocessing, prediction functions.
- `src/tools.py`: tool wrappers exposed to the agent.
- `src/agent.py`: orchestration loop, tool-calling, critic flow.
- `retrain_model.py`: local retraining and artifact generation.
- `notebooks/churn-analysis.ipynb`: EDA, model comparison, threshold tuning, training narrative.
- `Dockerfile`, `docker-compose.yml`: containerization.

## 8) Dockerization Status

Completed:

- Dockerfile for Streamlit app runtime.
- docker-compose setup with environment variable wiring for `GROQ_API_KEY`.

Run options:

- Local: `streamlit run app.py`
- Docker: `docker compose up --build`

## 9) What Is Done vs Pending (Assessment Checklist)

### Done

- Notebook with EDA + model training/evaluation workflow.
- Trained model and callable inference functions.
- Working Streamlit app wired to model and agent.
- Agent with multi-step tool use and computation tools.
- Modular project structure.
- Dockerization.

### In progress / to finalize for submission

- Final README refinement (this file can still be tightened before submission).
- Short reflection section (half-page max).
- Honest time-spent breakdown section.
- Public repo URL + hosted app URL for final submission package.
- Optional: stronger hard guarantees for numeric grounding in final answers.

## 10) Reflection (Draft)

Hardest part:

- The most challenging part was balancing agent flexibility with factual reliability. Free-tier models are fast but may vary in tool-calling reliability, so the loop had to support both native tool-calling and JSON fallback while still producing grounded answers.

What I learned:

- Good agent behavior comes from orchestration and safeguards, not prompt wording alone.
- Threshold tuning can be more important than changing algorithms when business objectives prioritize recall.
- Restricting execution context is essential when exposing "code as a tool".

What I would improve with more time:

- Add strict numeric citation checks from final answer back to tool outputs.
- Add an eval set (10-15 benchmark questions) with hallucination scoring.
- Add stronger memory for follow-up context and better critic retry policies.

## 11) Honest Time Note (Draft)

Approximate time allocation (target 8-10 hours):

- 2.0h: Data understanding, cleaning decisions, and EDA.
- 2.0h: Model comparison, threshold tuning, and artifact packaging.
- 2.5h: Agent and tool integration (including restricted execution).
- 1.5h: Streamlit chat UX, error handling, and chart display.
- 1.0h: Dockerization, testing, and cleanup.

Total: ~9.0 hours focused effort.

## 12) AI Tool Usage Disclosure

AI tools were used to accelerate implementation and documentation, including:

- code structure suggestions,
- bug-fix support,
- prompt/system-message refinement,
- and README drafting.

All final code decisions, testing, and integration were reviewed and validated manually in this project context.

## 13) Quick Start

### Prerequisites

- Python 3.11+
- A valid Groq API key

### Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Create environment file:
   - copy `.env.example` to `.env`
   - set `GROQ_API_KEY=...`
3. Run app:
   - `streamlit run app.py`

### Optional: retrain model

- `python retrain_model.py`

---

This README documents what has been implemented so far and where the project currently stands against the Adept Tech Solutions assessment requirements.
