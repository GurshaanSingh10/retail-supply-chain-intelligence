
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "train_analysis.csv"

df = pd.read_csv(DATA)

train = df[df["Validation"] == 0].copy()
valid = df[df["Validation"] == 1].copy()

target = "went_on_backorder"
excluded = [target, "sku", "Validation"]

X_train = train.drop(columns=excluded, errors="ignore")
X_valid = valid.drop(columns=excluded, errors="ignore")

X_train = X_train.select_dtypes(include="number")
X_valid = X_valid[X_train.columns]
y_train = train[target]
y_valid = valid[target]

feature_sets = {
    "A - Full baseline": list(X_train.columns),
    "B - Excluding potential timing risks": [
        c for c in X_train.columns
        if c not in ["local_bo_qty", "pieces_past_due"]
    ],
}

for name, features in feature_sets.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train[features], y_train)

    predictions = model.predict(X_valid[features])
    probabilities = model.predict_proba(X_valid[features])[:, 1]

    print(f"\n{'=' * 60}")
    print(name)
    print("Number of features:", len(features))
    print("ROC-AUC:", round(roc_auc_score(y_valid, probabilities), 4))
    print("PR-AUC:", round(average_precision_score(y_valid, probabilities), 4))
    print("Precision:", round(precision_score(y_valid, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_valid, predictions, zero_division=0), 4))
    print("F1:", round(f1_score(y_valid, predictions, zero_division=0), 4))
