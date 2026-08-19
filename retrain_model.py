"""Retrain the model with the current scikit-learn version to avoid version mismatch."""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# 1. Load and clean data
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Customer-Churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

print(f"Data loaded: {df.shape}")

# ============================================================
# 2. Encode features (same logic as notebook)
# ============================================================
df_model = df.copy()
df_model = df_model.drop(columns=["customerID"])

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0})

df_model["gender"] = df_model["gender"].map({"Female": 0, "Male": 1})
df_model["Churn"] = df_model["Churn"].map({"No": 0, "Yes": 1})

service_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
for col in service_cols:
    df_model[col] = df_model[col].map({
        "Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0,
    })

df_model["Contract"] = df_model["Contract"].map({
    "Month-to-month": 0, "One year": 1, "Two year": 2,
})

df_model = pd.get_dummies(df_model, columns=["InternetService", "PaymentMethod"], drop_first=True)
bool_cols = df_model.select_dtypes(include="bool").columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

print(f"Encoded shape: {df_model.shape}")

# ============================================================
# 3. Split data
# ============================================================
from sklearn.model_selection import train_test_split

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# 4. Train final model
# ============================================================
FINAL_THRESHOLD = 0.35

final_model = DecisionTreeClassifier(max_depth=5, random_state=42)
final_model.fit(X_train, y_train)

# Evaluate
y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= FINAL_THRESHOLD).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nFinal Model Evaluation (threshold={FINAL_THRESHOLD}):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# ============================================================
# 5. Save artifacts
# ============================================================
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(final_model, os.path.join(MODEL_DIR, "churn_decision_tree.pkl"))
joblib.dump(FINAL_THRESHOLD, os.path.join(MODEL_DIR, "churn_threshold.pkl"))
joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, "model_features.pkl"))

print(f"\nModel artifacts saved to {MODEL_DIR}")
print(f"Features: {X_train.columns.tolist()}")
