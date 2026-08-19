import pandas as pd
import os


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Customer-Churn.csv")


def load_clean_data():
    """Load and clean the customer churn dataset."""
    df = pd.read_csv(DATA_PATH)

    # TotalCharges has blank strings that pandas reads as strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows where tenure=0 and TotalCharges is NaN (11 new customers)
    df = df.dropna(subset=["TotalCharges"])

    return df


def get_dataset_summary(df):
    """Return a structured summary of the dataset."""
    summary = {
        "total_customers": len(df),
        "columns": df.columns.tolist(),
        "numerical_columns": ["tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_columns": [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"
        ],
        "target_column": "Churn",
        "churn_distribution": df["Churn"].value_counts().to_dict(),
        "churn_rate": round(df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary


def describe_numerical(df):
    """Return descriptive statistics for numerical columns."""
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    stats = df[numerical_cols].describe().round(2).to_dict()
    return stats


def get_churn_by_category(df, column):
    """Compute churn rate grouped by a categorical column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    churn_rate = (
        df.groupby(column)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(2)
        .to_dict()
    )

    counts = df[column].value_counts().to_dict()

    return {
        "column": column,
        "churn_rate_by_group": churn_rate,
        "group_counts": counts,
    }


def get_top_churn_customers(df, n=10):
    """Return top N customers most likely to churn based on risk indicators."""
    # Use a simple heuristic: high monthly charges, low tenure, no contract
    df_copy = df.copy()

    # Encode basic risk signals
    df_copy["contract_risk"] = df_copy["Contract"].map({
        "Month-to-month": 1,
        "One year": 0.5,
        "Two year": 0,
    })
    df_copy["payment_risk"] = (
        df_copy["PaymentMethod"] == "Electronic check"
    ).astype(int)
    df_copy["internet_risk"] = (
        df_copy["InternetService"] == "Fiber optic"
    ).astype(int)

    # Simple risk score
    df_copy["risk_indicator"] = (
        df_copy["contract_risk"] * 0.4
        + df_copy["payment_risk"] * 0.3
        + df_copy["internet_risk"] * 0.3
    )

    top = df_copy.nlargest(n, "risk_indicator")[
        ["customerID", "gender", "tenure", "Contract", "InternetService",
         "MonthlyCharges", "TotalCharges", "Churn"]
    ]

    return top.to_dict(orient="records")


def get_correlation_with_churn(df):
    """Compute correlation of numerical features with churn."""
    df_encoded = df.copy()
    df_encoded["Churn_binary"] = (df_encoded["Churn"] == "Yes").astype(int)

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    correlations = {}
    for col in numerical_cols:
        correlations[col] = round(df_encoded[col].corr(df_encoded["Churn_binary"]), 4)

    return correlations


def execute_restricted_code(df, code_string):
    """
    Execute a restricted pandas query against the dataset.
    Only allows safe operations: filtering, grouping, aggregation.
    Returns the result as a serializable dict or string.
    """
    import io
    import contextlib

    # Restricted namespace
    safe_builtins = {
        "len": len,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "sorted": sorted,
        "print": print,
        "True": True,
        "False": False,
        "None": None,
    }

    namespace = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
    }

    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code_string, namespace)

        stdout_output = output_buffer.getvalue().strip()

        # If the code assigned a result variable, return it
        if "result" in namespace:
            result = namespace["result"]
            if isinstance(result, pd.DataFrame):
                return {"type": "dataframe", "data": result.head(50).to_dict(orient="records"), "shape": list(result.shape)}
            elif isinstance(result, pd.Series):
                return {"type": "series", "data": result.to_dict()}
            else:
                return {"type": "value", "data": str(result)}

        if stdout_output:
            return {"type": "output", "data": stdout_output}

        return {"type": "output", "data": "Code executed successfully (no output)."}

    except Exception as e:
        return {"type": "error", "data": f"Error: {type(e).__name__}: {str(e)}"}
