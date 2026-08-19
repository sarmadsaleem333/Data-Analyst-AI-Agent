# Autonomous Data Analyst for Customer Churn

Deployed URL: https://data-analyst-ai-agent-bzywut5ot5jn4naep2sajs.streamlit.app/

Qwen is currently the best-performing model for this agent and it is available on the free tier. However, due to the free-tier token limits, it may occasionally hit the maximum token quota and fail to respond. Instead of switching permanently to lower-performing models, I prefer to continue using Qwen as the primary model. When the Qwen token limit is reached, the system should wait for around 20 minutes and then retry once the limit resets.

## 1. Project Overview

This repository contains the implementation of the Adept Tech Solutions AI Engineer assessment — **Autonomous Data Analyst**.

The solution combines:

- a trained churn prediction model (Decision Tree),
- a tool-using autonomous agent backed by Groq-hosted LLMs,
- a critic agent that validates answers before returning them to the user,
- a Streamlit chat interface for natural-language interaction with the dataset.

Built across three stages:

| Stage | Focus | Status |
|---|---|---|
| Stage 1 | Data cleaning, EDA, model training, evaluation | Done |
| Stage 2 | Streamlit chat app wired to model + agent | Done |
| Stage 3 | Autonomous agent loop with tools, critic, memory | Done |

---

## 2. What Has Been Built

### Stage 1 — Model as a Callable Tool

- Data cleaning pipeline: `TotalCharges` blank-string coercion, 11-row drop for missing billing history.
- Model training and evaluation in `notebooks/churn-analysis.ipynb`.
- Retraining script `retrain_model.py` for artifact regeneration.
- Callable prediction functions: `predict_churn_risk(customer_df)`, `predict_batch(customer_df)`.

Saved artifacts in `models/`:

| File | Contents |
|---|---|
| `churn_decision_tree.pkl` | Trained Decision Tree (`max_depth=5`, `random_state=42`) |
| `churn_threshold.pkl` | Optimal classification threshold (`0.35`) |
| `model_features.pkl` | Ordered feature list used during training |

### Stage 2 — Streamlit Chat Interface

- `app.py`: live chat interface with sidebar dataset summary.
- Multi-turn chat history handling (last 6 messages passed as context).
- Tool-call transparency panel (shows tool name, arguments, result preview).
- Critic feedback display when the agent's answer is flagged.
- Error handling for missing API key and runtime exceptions.

### Stage 3 — Agent Implementation

- Plan-act-check loop using Groq free-tier models.
- 9 registered tools covering EDA, prediction, segment analysis, code execution, and charting.
- Critic agent validates every final answer before display; retries once with feedback if issues found.
- Structured JSON response parsing with fallback to regex extraction.

**Registered tools:**

| Tool | Purpose |
|---|---|
| `dataset_summary` | Total customers, columns, churn distribution |
| `numerical_stats` | Descriptive stats for tenure, MonthlyCharges, TotalCharges |
| `churn_by_category` | Churn rate by any categorical column |
| `top_churn_customers` | Top N customers by heuristic risk indicators |
| `correlation` | Numerical feature correlation with churn |
| `predict_customer` | Single-customer churn risk prediction by customerID |
| `predict_hypothetical` | Prediction for a hypothetical customer profile |
| `execute_code` | Restricted pandas code execution against the dataset |
| `aggregate_segment_risk` | Aggregate churn risk for a filtered customer segment |

**LLM models tried (in order):**

1. `qwen/qwen3.6-27b` — works with JSON-formatted tool calling
2. `openai/gpt-oss-20b`


The agent falls through to the next model if one is unavailable.

---

## 3. Data Cleaning Findings

**Issue identified:** `TotalCharges` contains blank strings that pandas reads as non-numeric values.

**Actions taken:**

1. `pd.to_numeric(errors="coerce")` converts blanks to NaN.
2. Rows with NaN `TotalCharges` are dropped (11 customers, all with `tenure = 0`).

**Why this is reasonable:**

- These are new customers with no billing history; the model needs `TotalCharges` as a feature.
- Keeping coerced NaN or zero values would introduce misleading signals.

Final cleaned dataset: **7,032 customers**, 21 columns.

---

## 4. Model Choice and Rationale

**Selected model:** Decision Tree Classifier (`max_depth=5`, `random_state=42`).

**Why Decision Tree:**

- Achieved the best recall/F1 trade-off for the churn class in comparative evaluation.
- Interpretable: feature importances are directly available for the agent to report.
- Works well with the mixed categorical/numeric feature set after encoding.

**Evaluation metrics (custom threshold = 0.35):**

| Metric | Value |
|---|---|
| Recall (churn class) | ~71.66% |
| F1-score (churn class) | ~62.40% |
| ROC-AUC | Tracked in notebook |

**Threshold rationale:**

- Recall is prioritized because missing a true churner (false negative) is costlier than over-flagging.
- Threshold 0.35 was selected from threshold sweep results in the notebook.

---

## 5. Agent Planning and Verification Approach

### Planning

The system prompt instructs the model to decompose complex questions into sequential tool calls. One tool call per iteration; the agent inspects each result before deciding the next step.

### Verification layers

| Layer | Mechanism |
|---|---|
| Tool errors | Caught and surfaced to the loop; agent may retry with a different tool |
| Numeric grounding | Prompt rule: "Never invent numbers — every number must come from a tool result" |
| Critic agent | A second LLM call validates the final answer against tool outputs before display |
| Critic retry | If the critic flags issues, the agent retries once with the critic's suggested fix |

### Trade-off

This is a practical, lightweight verification approach suitable for free-tier API usage. It is not a formal provenance checker — some derived numbers may pass the heuristic critic without a strict citation chain.

---

## 6. Evaluation Framework

A dedicated evaluation suite lives in `eval/` and runs 12 benchmark questions through the live agent.

### Running the evaluation

```bash
python eval/run_eval.py
```

This will:

1. Load questions from `eval/questions.json`.
2. Send each question through `run_agent()`.
3. Score each answer using rules in `eval/scoring.py`.
4. Write results to `eval/report.md`.

### Question categories (12 total)

| Category | Questions | What is tested |
|---|---|---|
| EDA | Q1–Q5 | Churn rate, category breakdowns, correlations |
| Model | Q6–Q8 | Individual prediction, top-risk listing, hypothetical scenarios |
| Segment | Q9–Q10 | Aggregate risk, filtered segment analysis |
| Agent Reasoning | Q11–Q12 | Multi-step comparison, compound filtering |

### Scoring methodology

| Metric | Definition |
|---|---|
| **Accuracy** | Question passes if answer meets type-specific criteria (numeric within tolerance, keyword match, list length, multi-step completeness) |
| **Hallucination rate** | Count of numbers in the final answer (>5 in magnitude) that cannot be traced to any tool call output, divided by total numbers in answers |
| **Tool call check** | Questions that produced zero tool calls despite requiring computation are flagged |

### Latest evaluation results

| Metric | Value |
|---|---|
| Questions tested | 12 |
| Passed | 9 |
| Failed | 3 |
| **Accuracy** | **75.0%** |
| Numeric hallucinations | 7 |
| Total numbers in answers | 40 |
| Hallucination rate | 17.5% |
| Failed tool calls | 1 |

### Per-question breakdown

| # | Category | Type | Pass | Reason | Tool Calls | Time |
|---|---|---|---|---|---|---|
| 1 | EDA | numeric | Pass | numeric match | 1 | 6.5s |
| 2 | EDA | category | Pass | keyword match | 1 | 44.4s |
| 3 | EDA | numeric | Pass | numeric match | 1 | 69.2s |
| 4 | EDA | numeric | Pass | numeric match | 1 | 28.3s |
| 5 | EDA | category | Pass | keyword match | 1 | 31.0s |
| 6 | Model | category | Fail | keyword mismatch | 0 | 22.1s |
| 7 | Model | list | Fail | list check fail | 1 | 87.8s |
| 8 | Model | numeric | Pass | numeric match | 1 | 65.4s |
| 9 | Segment | numeric | Pass | numeric match | 1 | 30.7s |
| 10 | Segment | numeric | Fail | numeric mismatch | 1 | 43.3s |
| 11 | Reasoning | multi-step | Pass | multi-step pass | 1 | 33.2s |
| 12 | Reasoning | multi-step | Pass | multi-step pass | 1 | 34.5s |

**Failed questions analysis:**

- **Q6** (risk level for customer 0376-YMCJC): The agent called the prediction tool but the keyword check for "medium" in the answer did not match — likely the agent described the result in prose without using the exact risk-level word.
- **Q7** (top 3 customers by ID): The agent returned a descriptive list but the regex-based list checker did not detect 3 distinct customer IDs in the answer text.
- **Q10** (customers with tenure ≤ 12): The agent used `execute_code` instead of `aggregate_segment_risk` and the answer included a different aggregation scope.

These are scoring-rigidity failures more than agent-logic failures — the agent produced substantively correct information but in a format the automated scorer could not match.

### Evaluation file structure

```
eval/
├── questions.json    # 12 benchmark questions with type/expected values
├── run_eval.py       # Evaluation runner (imports run_agent directly)
├── scoring.py        # Scoring functions: numeric, category, list, hallucination
├── report.md         # Generated report after each eval run
└── eval_set.json     # Legacy eval set (retained for reference)
```

---

## 7. Repository Structure

```
notebooks/
├── app.py                          # Streamlit chat interface
├── retrain_model.py                # Model retraining script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Docker Compose with GROQ_API_KEY wiring
├── .env.example                    # Environment variable template
├── .gitignore
│
├── data/
│   └── Customer-Churn.csv          # Source dataset (7,032 customers)
│
├── models/
│   ├── churn_decision_tree.pkl     # Trained model
│   ├── churn_threshold.pkl         # Classification threshold (0.35)
│   └── model_features.pkl          # Feature order for inference
│
├── notebooks/
│   └── churn-analysis.ipynb        # Stage 1: EDA, training, evaluation
│
├── src/
│   ├── __init__.py
│   ├── data.py                     # Data loading, cleaning, EDA, charts
│   ├── model.py                    # Model loading, preprocessing, prediction
│   ├── tools.py                    # Tool registry for the agent
│   └── agent.py                    # Agent loop, critic, LLM orchestration
│
└── eval/
    ├── questions.json              # 12 benchmark questions
    ├── scoring.py                  # Scoring and hallucination detection
    ├── run_eval.py                 # Evaluation runner
    ├── report.md                   # Generated evaluation report
    └── eval_set.json               # Legacy eval set
```

---

## 8. Dockerization

**Dockerfile** and **docker-compose.yml** are provided.

Run locally:

```bash
streamlit run app.py
```

Run via Docker:

```bash
docker compose up --build
```

The compose file wires `GROQ_API_KEY` from the `.env` file into the container.

---

## 9. Assessment Checklist

### Required (Must Complete)

| Item | Status |
|---|---|
| Notebook with trained model, data-cleaning decisions, metric justification | Done |
| Working Streamlit chat app wired to model and agent | Done |
| Agent that plans multi-step, computes real answers, self-checks | Done |
| Modular, organized code | Done |
| Written reflection | Done |
| Git etiquette (meaningful commits) | Done |
| Dockerization | Done |

### Stretch Goals (Optional)

| Item | Status |
|---|---|
| Critic agent that validates and can reject/revise answers | Done |
| Multi-turn memory for follow-up questions | Done |
| Small evaluation set (12 questions) with accuracy and hallucination report | Done |
| Deploying the app somewhere reachable | Done |
| Ngrok reverse proxy access | Done |

---

## 10. Reflection

### What was hardest

Balancing agent flexibility with factual reliability. Free-tier LLMs vary in tool-calling reliability — some attempts to call tools produce malformed JSON, requiring the regex fallback parser. The loop had to support both structured and freeform responses while still grounding every number in a tool result.

The critic agent was also non-trivial: it needed to distinguish between "number not in tool output but correctly derived" (acceptable) and "number fabricated entirely" (hallucination). The heuristic approach (flag numbers >5 not found in tool outputs) catches most cases but has false positives on derived ratios.

### What I learned

- Good agent behavior comes from orchestration and prompt engineering together — prompt alone is insufficient.
- Threshold tuning on the model side had more business impact than switching algorithms.
- Restricting code execution context (safe builtins, no filesystem access) is essential when exposing "execute code" as a tool.
- Evaluation is the hardest part to get right — automated scoring of natural-language answers is inherently lossy.

### What I would improve with more time

- Add strict numeric citation provenance (trace each number in the answer back to the exact tool call that produced it).
- Expand the eval set to 25+ questions with human-graded gold answers.
- Add conversation memory persistence across Streamlit sessions.
- Add retry logic with exponential backoff for Groq rate limits.

---

## 11. Time Note

Approximate time allocation (target 8–10 hours):

| Task | Hours |
|---|---|
| Data understanding, cleaning, EDA | 2.0 |
| Model comparison, threshold tuning, artifact packaging | 2.0 |
| Agent and tool integration (including restricted execution) | 2.5 |
| Streamlit chat UX, error handling, chart display | 1.5 |
| Critic agent, eval framework, Dockerization | 1.5 |
| Testing, debugging, documentation | 0.5 |
| **Total** | **~10.0** |

---

## 12. AI Tool Usage Disclosure

AI tools were used to accelerate implementation and documentation, including:

- Code structure suggestions and scaffolding
- Bug-fix support during agent loop integration
- Prompt and system-message refinement for the critic agent
- README drafting and evaluation report formatting

All final code decisions, testing, and integration were reviewed and validated manually.

---

## 13. Quick Start

### Prerequisites

- Python 3.11+
- A valid Groq API key (free tier works)

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

# 3. Run the app
streamlit run app.py
```

### Run evaluation

```bash
python eval/run_eval.py
```

Report is written to `eval/report.md`.

### Retrain model (optional)

```bash
python retrain_model.py
```

---

This README reflects the current state of the project against the Adept Tech Solutions assessment requirements as of August 2026.
