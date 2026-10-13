
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "train_analysis.csv"

df = pd.read_csv(DATA)

target = "went_on_backorder"

train = df[df["Validation"] == 0].copy()
valid = df[df["Validation"] == 1].copy()

X_train = train.drop(columns=[target, "sku", "Validation"])
X_valid = valid.drop(columns=[target, "sku", "Validation"])

# Use numeric features only for the initial baseline.
X_train = X_train.select_dtypes(include="number")
X_valid = X_valid[X_train.columns]

y_train = train[target]
y_valid = valid[target]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_valid)
prob = pipeline.predict_proba(X_valid)[:, 1]

print("TRAIN ROWS:", len(X_train))
print("VALIDATION ROWS:", len(X_valid))
print("PREDICTORS:", len(X_train.columns))
print("VALIDATION POSITIVE RATE:", round(y_valid.mean() * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_valid, pred, digits=4, zero_division=0))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_valid, pred))

print("ROC-AUC:", round(roc_auc_score(y_valid, prob), 4))
print("PR-AUC / Average Precision:",
      round(average_precision_score(y_valid, prob), 4))
