
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train_analysis.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "went_on_backorder"

# Use the official validation split
train_df = df[df["Validation"] == 0].copy()
valid_df = df[df["Validation"] == 1].copy()

# Exclude identifiers, target, validation metadata,
# and potentially leakage-prone operational fields.
excluded = {
    TARGET,
    "sku",
    "Validation",
    "local_bo_qty",
    "pieces_past_due",
}

# Numeric features only, consistent with the earlier model
X_train_base = train_df.drop(
    columns=list(excluded), errors="ignore"
).select_dtypes(include="number")

X_valid_base = valid_df[X_train_base.columns]

y_train = train_df[TARGET].astype(int)
y_valid = valid_df[TARGET].astype(int)

# Check that the split is valid
assert len(X_train_base) == len(y_train)
assert len(X_valid_base) == len(y_valid)

# Version 1: Original engineered feature, including its missingness
# when forecast_3_month <= 0.
features_v1 = list(X_train_base.columns)

# Version 2: Remove the engineered feature completely.
features_v2 = [
    c for c in features_v1
    if c != "inventory_below_3m_forecast"
]

# Version 3: Use an explicit, non-missing flag plus an availability flag.
# This makes the meaning of the feature engineering clearer.
X_train_v3 = X_train_base.drop(
    columns=["inventory_below_3m_forecast"], errors="ignore"
).copy()

X_valid_v3 = X_valid_base.drop(
    columns=["inventory_below_3m_forecast"], errors="ignore"
).copy()

for frame in (X_train_v3, X_valid_v3):
    if "forecast_3_month" in frame.columns:
        frame["forecast_3m_available"] = (
            frame["forecast_3_month"] > 0
        ).astype(int)

        frame["inventory_below_3m_forecast_flag"] = (
            (frame["forecast_3_month"] > 0)
            & (frame["national_inv"] < frame["forecast_3_month"])
        ).astype(int)

# Evaluate a feature configuration using the same model and split
def evaluate(name, X_train, X_valid):
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

    result = {
        "version": name,
        "features": X_train.shape[1],
        "ROC-AUC": roc_auc_score(y_valid, probabilities),
        "PR-AUC": average_precision_score(y_valid, probabilities),
        "precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1": f1_score(y_valid, predictions, zero_division=0),
    }

    print(f"\n--- {name} ---")
    print(pd.Series(result).to_string())
    print("Confusion matrix:")
    print(confusion_matrix(y_valid, predictions))

    return result


results = []

# V1: Original engineered feature
results.append(
    evaluate(
        "V1_original_engineered_feature",
        X_train_base[features_v1],
        X_valid_base[features_v1],
    )
)

# V2: No engineered inventory-vs-forecast feature
results.append(
    evaluate(
        "V2_feature_removed",
        X_train_base[features_v2],
        X_valid_base[features_v2],
    )
)

# V3: Explicit flags, without the NA-based engineered feature
results.append(
    evaluate(
        "V3_explicit_flags",
        X_train_v3,
        X_valid_v3,
    )
)

comparison = pd.DataFrame(results).sort_values(
    "PR-AUC", ascending=False
)

print("\n\n========== FINAL COMPARISON ==========")
print(comparison.to_string(index=False))

output_path = ROOT / "data" / "processed" / "inventory_feature_comparison.csv"
comparison.to_csv(output_path, index=False)
print(f"\nSaved comparison to: {output_path}")
