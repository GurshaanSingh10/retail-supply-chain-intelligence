
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "validation_risk_predictions.csv"

df = pd.read_csv(DATA_PATH)

train = df[df["Validation"] == 0].copy()
valid = df[df["Validation"] == 1].copy()

target = "went_on_backorder"
excluded = {
    target,
    "sku",
    "Validation",
    "local_bo_qty",
    "pieces_past_due",
}

X_train = train.drop(columns=list(excluded)).select_dtypes(include="number")
X_valid = valid[X_train.columns].copy()

y_train = train[target].astype(int)
y_valid = valid[target].astype(int)

for col in ["perf_6_month_avg", "perf_12_month_avg"]:
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

valid["predicted_backorder_probability"] = model.predict_proba(
    X_valid_imp
)[:, 1]

valid["risk_category"] = pd.cut(
    valid["predicted_backorder_probability"],
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low", "Medium", "High"],
)

valid["actual_backorder"] = y_valid

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
valid.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("\nRisk category counts:")
print(valid["risk_category"].value_counts().sort_index())
print("\nAverage predicted probability by category:")
print(
    valid.groupby("risk_category", observed=False)[
        "predicted_backorder_probability"
    ].mean().round(3)
)
