
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "train_analysis.csv"

df = pd.read_csv(DATA)

train = df[df["Validation"] == 0].copy()
valid = df[df["Validation"] == 1].copy()

target = "went_on_backorder"
excluded = [
    target, "sku", "Validation",
    "local_bo_qty", "pieces_past_due",
]

X_train = train.drop(columns=excluded, errors="ignore")
X_valid = valid.drop(columns=excluded, errors="ignore")

X_train = X_train.select_dtypes(include="number")
X_valid = X_valid[X_train.columns]

y_train = train[target]
y_valid = valid[target]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

pred = model.predict(X_valid)
prob = model.predict_proba(X_valid)[:, 1]

print("TRAIN ROWS:", len(X_train))
print("VALIDATION ROWS:", len(X_valid))
print("FEATURES:", len(X_train.columns))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_valid, pred, digits=4, zero_division=0))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_valid, pred))

print("ROC-AUC:", round(roc_auc_score(y_valid, prob), 4))
print("PR-AUC:", round(average_precision_score(y_valid, prob), 4))


# Feature names after missing-value indicators are added.
transformed_names = model.named_steps[
    "imputer"
].get_feature_names_out(X_train.columns)

importances = pd.Series(
    model.named_steps["classifier"].feature_importances_,
    index=transformed_names,
).sort_values(ascending=False)

print("\nTOP 10 FEATURE IMPORTANCES:")
print(importances.head(10).to_string())
