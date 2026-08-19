import json
import pandas as pd
from src.data import (
    load_clean_data,
    get_dataset_summary,
    describe_numerical,
    get_churn_by_category,
    get_top_churn_customers,
    get_correlation_with_churn,
    execute_restricted_code,
    validate_answer_against_data,
)
from src.model import predict_churn_risk, predict_batch


# Global dataset (loaded once)
_DF = None


def get_df():
    global _DF
    if _DF is None:
        _DF = load_clean_data()
    return _DF


def dataset_summary_tool():
    """Get overview of the dataset: columns, row count, churn distribution."""
    df = get_df()
    summary = get_dataset_summary(df)
    return json.dumps(summary, indent=2, default=str)


def numerical_stats_tool():
    """Get descriptive statistics for tenure, MonthlyCharges, TotalCharges."""
    df = get_df()
    stats = describe_numerical(df)
    return json.dumps(stats, indent=2, default=str)


def churn_by_category_tool(column):
    """Get churn rate broken down by a categorical column."""
    df = get_df()
    result = get_churn_by_category(df, column)
    return json.dumps(result, indent=2, default=str)


def top_churn_customers_tool(n=10):
    """Get the N customers with highest risk indicators."""
    df = get_df()
    customers = get_top_churn_customers(df, n)
    return json.dumps(customers, indent=2, default=str)


def correlation_tool():
    """Get correlation of numerical features with churn."""
    df = get_df()
    corr = get_correlation_with_churn(df)
    return json.dumps(corr, indent=2)


def predict_customer_tool(customer_id):
    """Predict churn risk for a specific customer by their ID."""
    df = get_df()
    customer = df[df["customerID"] == customer_id]
    if customer.empty:
        return json.dumps({"error": f"Customer '{customer_id}' not found."})
    result = predict_churn_risk(customer)
    result["customer_id"] = customer_id
    customer_info = customer.iloc[0].to_dict()
    result["customer_info"] = customer_info
    return json.dumps(result, indent=2, default=str)


def predict_hypothetical_tool(customer_data_json):
    """
    Predict churn risk for a hypothetical customer.
    Input: JSON string with customer features.
    Example: '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No",
               "tenure":5,"PhoneService":"Yes","MultipleLines":"No",
               "InternetService":"Fiber optic","OnlineSecurity":"No",
               "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
               "StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month",
               "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
               "MonthlyCharges":85.0,"TotalCharges":425.0}'
    """
    try:
        customer_dict = json.loads(customer_data_json)
        customer_df = pd.DataFrame([customer_dict])
        result = predict_churn_risk(customer_df)
        result["input_data"] = customer_dict
        return json.dumps(result, indent=2, default=str)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {str(e)}"})


def execute_code_tool(code):
    """
    Execute a restricted pandas query against the dataset.
    The code can use 'df' (the cleaned dataset) and 'pd' (pandas).
    Assign your result to a variable named 'result' to see its output.
    """
    df = get_df()
    output = execute_restricted_code(df, code)
    return json.dumps(output, indent=2, default=str)


def aggregate_segment_risk_tool(segment_filters_json):
    """
    Compute aggregate churn risk for a segment of customers.
    Input: JSON with filter conditions, e.g.:
    '{"Contract": "Month-to-month", "InternetService": "Fiber optic"}'
    or for multiple filters: '{"tenure_max": 12, "MonthlyCharges_min": 70}'
    """
    try:
        filters = json.loads(segment_filters_json)
        df = get_df()
        filtered = df.copy()

        for key, value in filters.items():
            if key.endswith("_max"):
                col = key.replace("_max", "")
                if col in filtered.columns:
                    filtered = filtered[filtered[col] <= value]
            elif key.endswith("_min"):
                col = key.replace("_min", "")
                if col in filtered.columns:
                    filtered = filtered[filtered[col] >= value]
            elif key in filtered.columns:
                filtered = filtered[filtered[key] == value]

        if filtered.empty:
            return json.dumps({"error": "No customers match the given filters."})

        predictions = predict_batch(filtered)

        churn_count = sum(1 for p in predictions if p["prediction"] == "Likely to Churn")
        stay_count = len(predictions) - churn_count
        avg_risk = round(sum(p["risk_score"] for p in predictions) / len(predictions), 4)
        high_risk = sum(1 for p in predictions if p["risk_level"] == "High")
        medium_risk = sum(1 for p in predictions if p["risk_level"] == "Medium")
        low_risk = sum(1 for p in predictions if p["risk_level"] == "Low")

        return json.dumps({
            "segment_size": len(filtered),
            "filters_applied": filters,
            "predicted_churn": churn_count,
            "predicted_stay": stay_count,
            "churn_rate_percent": round(churn_count / len(filtered) * 100, 2),
            "average_risk_score": avg_risk,
            "risk_distribution": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk,
            },
        }, indent=2)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})
    except Exception as e:
        return json.dumps({"error": f"Segment analysis failed: {str(e)}"})


# Tool registry for the agent
TOOLS = {
    "dataset_summary": {
        "function": dataset_summary_tool,
        "description": "Get an overview of the customer churn dataset: total customers, columns, churn distribution.",
        "parameters": {},
    },
    "numerical_stats": {
        "function": numerical_stats_tool,
        "description": "Get descriptive statistics (mean, std, min, max, quartiles) for numerical features: tenure, MonthlyCharges, TotalCharges.",
        "parameters": {},
    },
    "churn_by_category": {
        "function": churn_by_category_tool,
        "description": "Get churn rate breakdown for a specific categorical column. Valid columns: gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod.",
        "parameters": {"column": "string"},
    },
    "top_churn_customers": {
        "function": top_churn_customers_tool,
        "description": "Get the top N customers most likely to churn based on risk indicators.",
        "parameters": {"n": "integer (default 10)"},
    },
    "correlation": {
        "function": correlation_tool,
        "description": "Get correlation of numerical features (tenure, MonthlyCharges, TotalCharges) with churn.",
        "parameters": {},
    },
    "predict_customer": {
        "function": predict_customer_tool,
        "description": "Predict churn risk for a specific customer by their customerID. Returns risk score, risk level, prediction, and top contributing factors.",
        "parameters": {"customer_id": "string"},
    },
    "predict_hypothetical": {
        "function": predict_hypothetical_tool,
        "description": "Predict churn risk for a hypothetical customer. Pass a JSON object with customer features as they appear in the raw dataset.",
        "parameters": {"customer_data_json": "JSON string"},
    },
    "execute_code": {
        "function": execute_code_tool,
        "description": "Execute a restricted pandas query against the dataset. Use 'df' for the DataFrame and 'pd' for pandas. Assign output to 'result'. Example: result = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100)",
        "parameters": {"code": "string (Python pandas code)"},
    },
    "aggregate_segment_risk": {
        "function": aggregate_segment_risk_tool,
        "description": "Compute aggregate churn risk for a customer segment. Pass JSON with filters like {\"Contract\": \"Month-to-month\"} or numeric ranges like {\"tenure_max\": 12}.",
        "parameters": {"segment_filters_json": "JSON string"},
    },
}
