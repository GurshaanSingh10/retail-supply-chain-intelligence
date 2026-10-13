
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(DATA_PATH)

train = df[df["Validation"] == 0].copy()
valid = df[df["Validation"] == 1].copy()

target = "went_on_backorder"

# Use only original numeric columns.
# Exclude target, SKU identifier, validation metadata,
# and fields whose timing may create leakage risk.
excluded = {
    target,
    "sku",
    "Validation",
    "local_bo_qty",
    "pieces_past_due",
}

X_train = train.drop(columns=list(excluded)).select_dtypes(
    include="number"
)
X_valid = valid[X_train.columns]

y_train = train[target].astype(int)
y_valid = valid[target].astype(int)

# Explicitly treat -99 supplier-performance values as missing.
for col in ["perf_6_month_avg", "perf_12_month_avg"]:
    if col in X_train.columns:
        X_train.loc[X_train[col] == -99, col] = float("nan")
        X_valid.loc[X_valid[col] == -99, col] = float("nan")

imputer = SimpleImputer(strategy="median", add_indicator=True)
X_train_imp = imputer.fit_transform(X_train)
X_valid_imp = imputer.transform(X_valid)

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_imp, y_train)

probabilities = model.predict_proba(X_valid_imp)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("Train rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Features:", X_train.shape[1])
print("Validation backorder rate:", round(y_valid.mean(), 4))
print("ROC-AUC:", round(roc_auc_score(y_valid, probabilities), 4))
print("PR-AUC:", round(average_precision_score(y_valid, probabilities), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_valid, predictions))
print("\nClassification report:")
print(classification_report(y_valid, predictions, zero_division=0))
