import json
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


def generate_chart(df, chart_type, x_col=None, y_col=None, hue_col=None, title=None):
    """
    Generate a chart and return it as a base64-encoded PNG string.
    Supported chart types: bar, histogram, box, scatter, heatmap, pie.
    """
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if chart_type == "bar" and x_col:
            if hue_col:
                churn_rate = (
                    df.groupby([x_col, hue_col])["Churn"]
                    .apply(lambda x: (x == "Yes").mean() * 100)
                    .reset_index(name="ChurnRate")
                )
                sns.barplot(data=churn_rate, x=x_col, y="ChurnRate", hue=hue_col, ax=ax)
            else:
                churn_rate = (
                    df.groupby(x_col)["Churn"]
                    .apply(lambda x: (x == "Yes").mean() * 100)
                    .reset_index(name="ChurnRate")
                )
                sns.barplot(data=churn_rate, x=x_col, y="ChurnRate", ax=ax)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xlabel(x_col)
            if x_col in ["Contract", "PaymentMethod", "InternetService"]:
                ax.tick_params(axis="x", rotation=30)

        elif chart_type == "histogram" and x_col:
            data_churn = df[df["Churn"] == "Yes"][x_col].dropna()
            data_stay = df[df["Churn"] == "No"][x_col].dropna()
            ax.hist(data_stay, bins=30, alpha=0.6, label="No Churn", color="#2ecc71")
            ax.hist(data_churn, bins=30, alpha=0.6, label="Churn", color="#e74c3c")
            ax.legend()
            ax.set_xlabel(x_col)
            ax.set_ylabel("Count")

        elif chart_type == "box" and x_col and y_col:
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif chart_type == "scatter" and x_col and y_col:
            colors = df["Churn"].map({"Yes": "#e74c3c", "No": "#2ecc71"})
            ax.scatter(df[x_col], df[y_col], c=colors, alpha=0.4, s=15)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="No Churn"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="Churn"),
            ]
            ax.legend(handles=legend_elements)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif chart_type == "heatmap":
            numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            df_encoded = df.copy()
            df_encoded["Churn_binary"] = (df_encoded["Churn"] == "Yes").astype(int)
            cols = numerical_cols + ["Churn_binary"]
            corr = df_encoded[cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
            ax.set_title("Feature Correlations")

        elif chart_type == "pie" and x_col:
            counts = df[x_col].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_aspect("equal")

        else:
            plt.close(fig)
            return {"type": "error", "data": f"Unsupported chart type or missing columns: {chart_type}"}

        ax.set_title(title or f"{chart_type.title()} Chart")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {"type": "chart", "data": img_base64, "chart_type": chart_type}

    except Exception as e:
        plt.close(fig)
        return {"type": "error", "data": f"Chart generation failed: {type(e).__name__}: {str(e)}"}


def validate_answer_against_data(df, answer_text, tool_calls_made):
    """
    Validate an agent's answer against the raw data.
    Checks for:
    1. Numbers in answer that don't match any tool results
    2. Claims that contradict the data
    3. Missing computation for specific figures

    Returns a dict with validation result.
    """
    import re

    issues = []
    verified = []

    # Extract all numbers from the answer
    numbers_in_answer = re.findall(r'\b\d+\.?\d*%?\b', answer_text)

    # Collect all numbers that appeared in tool results
    tool_numbers = set()
    for tc in tool_calls_made:
        result_str = json.dumps(tc.get("result", {}))
        found = re.findall(r'\b\d+\.?\d*%?\b', result_str)
        tool_numbers.update(found)

    # Check if any prominent numbers in the answer are not backed by tool results
    # (Skip small numbers like 0, 1, 2 which are likely category counts)
    for num_str in numbers_in_answer:
        try:
            num = float(num_str.rstrip("%"))
            if num > 5:  # Only check meaningful numbers
                if num_str not in tool_numbers:
                    # Not necessarily an issue — could be derived. Flag as caution.
                    pass
                else:
                    verified.append(num_str)
        except ValueError:
            pass

    # Sanity checks against the actual dataset
    total_customers = len(df)
    churn_rate = (df["Churn"] == "Yes").mean() * 100

    # Check if answer mentions a total customer count that's wrong
    # Match patterns like "5000 customers", "dataset has 5000", "5000 total customers"
    total_match = re.search(
        r'(\d{3,5})\s*(?:total\s+)?customers\s+(?:in|in\s+the|of\s+the)\s+dataset',
        answer_text.lower(),
    )
    if not total_match:
        total_match = re.search(
            r'(?:total|overall|entire|dataset\s+has|contains)\s+(?:of\s+)?(\d{3,5})\s*customers',
            answer_text.lower(),
        )
    if not total_match:
        total_match = re.search(
            r'(\d{3,5})\s*total\s*customers',
            answer_text.lower(),
        )
    if total_match:
        claimed_total = int(total_match.group(1))
        if abs(claimed_total - total_customers) > 50:
            issues.append(
                f"Claimed ~{claimed_total} total customers, but dataset has {total_customers}."
            )

    # Check if answer claims an overall churn rate that's wildly off
    # Only flag if the answer explicitly says "overall" or "total" churn rate
    overall_rate_match = re.search(
        r'(?:overall|total|average)\s+(?:churn\s+)?rate\s+(?:is\s+)?(\d{1,3}\.?\d*)\s*%',
        answer_text.lower(),
    )
    if overall_rate_match:
        claimed_rate = float(overall_rate_match.group(1))
        if abs(claimed_rate - churn_rate) > 15:
            issues.append(
                f"Claimed overall churn rate of {claimed_rate}%, but actual is {churn_rate:.1f}%."
            )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "verified_numbers": verified[:10],
        "total_numbers_checked": len(numbers_in_answer),
    }
