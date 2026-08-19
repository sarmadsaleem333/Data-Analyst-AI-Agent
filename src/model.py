import os
import joblib
import pandas as pd


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model():
    """Load the trained decision tree model and its artifacts."""
    model = joblib.load(os.path.join(MODEL_DIR, "churn_decision_tree.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "churn_threshold.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    return model, threshold, features


def preprocess_customer(customer_df, model_features):
    """Preprocess a customer record into model-ready features."""
    data = customer_df.copy()

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])
    if "Churn" in data.columns:
        data = data.drop(columns=["Churn"])

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({"Yes": 1, "No": 0})

    if "gender" in data.columns:
        data["gender"] = data["gender"].map({"Female": 0, "Male": 1})

    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        if col in data.columns:
            data[col] = data[col].map({
                "Yes": 1, "No": 0,
                "No phone service": 0, "No internet service": 0,
            })

    if "Contract" in data.columns:
        data["Contract"] = data["Contract"].map({
            "Month-to-month": 0, "One year": 1, "Two year": 2,
        })

    if "InternetService" in data.columns:
        data["InternetService_Fiber optic"] = (
            data["InternetService"] == "Fiber optic"
        ).astype(int)
        data["InternetService_No"] = (
            data["InternetService"] == "No"
        ).astype(int)
        data = data.drop(columns=["InternetService"])

    if "PaymentMethod" in data.columns:
        data["PaymentMethod_Credit card (automatic)"] = (
            data["PaymentMethod"] == "Credit card (automatic)"
        ).astype(int)
        data["PaymentMethod_Electronic check"] = (
            data["PaymentMethod"] == "Electronic check"
        ).astype(int)
        data["PaymentMethod_Mailed check"] = (
            data["PaymentMethod"] == "Mailed check"
        ).astype(int)
        data = data.drop(columns=["PaymentMethod"])

    bool_cols = data.select_dtypes(include="bool").columns
    data[bool_cols] = data[bool_cols].astype(int)

    for feature in model_features:
        if feature not in data.columns:
            data[feature] = 0

    data = data[model_features]

    if data.isnull().sum().sum() > 0:
        raise ValueError("Preprocessing produced missing values.")

    return data


def get_top_factors(model, customer_processed, model_features, top_n=3):
    """Get top N contributing factors for a prediction."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": model_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    factors = []
    for _, row in importance_df.iterrows():
        feature = row["feature"]
        if feature in customer_processed.columns:
            value = customer_processed.iloc[0][feature]
            if value != 0:
                readable_name = feature
                for prefix in ["InternetService_", "PaymentMethod_"]:
                    if feature.startswith(prefix):
                        readable_name = feature.replace(prefix, "")
                factors.append({
                    "feature": readable_name,
                    "importance": round(float(row["importance"]), 4),
                })
        if len(factors) >= top_n:
            break

    if len(factors) < top_n:
        for _, row in importance_df.iterrows():
            feature = row["feature"]
            readable_name = feature
            for prefix in ["InternetService_", "PaymentMethod_"]:
                if feature.startswith(prefix):
                    readable_name = feature.replace(prefix, "")
            if not any(f["feature"] == readable_name for f in factors):
                factors.append({
                    "feature": readable_name,
                    "importance": round(float(row["importance"]), 4),
                })
            if len(factors) >= top_n:
                break

    return factors[:top_n]


def predict_churn_risk(customer_df):
    """
    Predict churn risk for a customer (or multiple customers).
    Accepts a DataFrame with customer data.
    Returns a dict with prediction details.
    """
    model, threshold, model_features = load_model()

    customer_processed = preprocess_customer(customer_df, model_features)
    risk_score = model.predict_proba(customer_processed)[0][1]
    prediction_value = int(risk_score >= threshold)

    prediction = "Likely to Churn" if prediction_value == 1 else "Likely to Stay"

    if risk_score >= 0.70:
        risk_level = "High"
    elif risk_score >= threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    top_factors = get_top_factors(model, customer_processed, model_features, top_n=3)

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_level": risk_level,
        "prediction": prediction,
        "threshold_used": threshold,
        "top_factors": top_factors,
    }


def predict_batch(customer_df):
    """Predict churn risk for multiple customers. Returns list of results."""
    model, threshold, model_features = load_model()
    customer_processed = preprocess_customer(customer_df, model_features)
    risk_scores = model.predict_proba(customer_processed)[:, 1]

    results = []
    for i, score in enumerate(risk_scores):
        prediction_value = int(score >= threshold)
        prediction = "Likely to Churn" if prediction_value == 1 else "Likely to Stay"
        if score >= 0.70:
            risk_level = "High"
        elif score >= threshold:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        results.append({
            "index": i,
            "risk_score": round(float(score), 4),
            "risk_level": risk_level,
            "prediction": prediction,
        })

    return results
