
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
PREDICTIONS_FILE = (
    ROOT / "data" / "processed" / "validation_risk_predictions.csv"
)

TARGET = "went_on_backorder"

# Features excluded from the conservative Random Forest model.
EXCLUDED_FROM_MODEL = {
    TARGET,
    "sku",
    "Validation",
    "local_bo_qty",
    "pieces_past_due",
}

# Features that deserve special review because their timing or meaning
# may be important for a future prediction task.
REVIEW_FEATURES = [
    "local_bo_qty",
    "pieces_past_due",
    "national_inv",
    "in_transit_qty",
    "forecast_3_month",
    "forecast_6_month",
    "forecast_9_month",
    "sales_1_month",
    "sales_3_month",
    "sales_6_month",
    "sales_9_month",
    "lead_time",
    "perf_6_month_avg",
    "perf_12_month_avg",
]


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    df = pd.read_csv(TRAIN_FILE)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    print("=" * 70)
    print("SUPPLY CHAIN MODEL — TARGET LEAKAGE AUDIT")
    print("=" * 70)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Target: {TARGET}")

    print("\nTarget distribution:")
    print(df[TARGET].value_counts(dropna=False).to_string())

    print("\nTarget prevalence:")
    print(f"{df[TARGET].mean():.2%}")

    # Check whether any columns are exact copies of the target.
    print("\n" + "=" * 70)
    print("1. POSSIBLE DIRECT TARGET COPIES")
    print("=" * 70)

    exact_copies = []

    for col in df.columns:
        if col == TARGET:
            continue

        comparable = df[[col, TARGET]].dropna()

        if (
            len(comparable) > 0
            and comparable[col].nunique() <= 2
            and comparable[col].equals(comparable[TARGET])
        ):
            exact_copies.append(col)

    if exact_copies:
        print("Potential exact target copies:", exact_copies)
    else:
        print("No exact target-copy columns found.")

    # Examine feature relationships without treating them as proof of leakage.
    print("\n" + "=" * 70)
    print("2. FEATURE ASSOCIATIONS")
    print("=" * 70)

    associations = []

    for col in REVIEW_FEATURES:
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.notna()

        if valid.sum() == 0:
            continue

        associations.append({
            "feature": col,
            "missing_pct": round(numeric.isna().mean() * 100, 2),
            "unique_values": int(numeric.nunique()),
            "correlation_with_target": round(
                numeric[valid].corr(df.loc[valid, TARGET]), 4
            ),
            "excluded_from_conservative_model": (
                col in EXCLUDED_FROM_MODEL
            ),
        })

    association_df = pd.DataFrame(associations)

    if not association_df.empty:
        print(
            association_df.sort_values(
                "correlation_with_target",
                key=lambda s: s.abs(),
                ascending=False,
            ).to_string(index=False)
        )

    # Look for features that are nearly perfect target proxies.
    print("\n" + "=" * 70)
    print("3. FEATURES WITH VERY STRONG ASSOCIATIONS")
    print("=" * 70)

    if not association_df.empty:
        strong = association_df[
            association_df["correlation_with_target"].abs() >= 0.80
        ]

        if strong.empty:
            print(
                "No reviewed numeric feature has absolute correlation "
                "of 0.80 or higher."
            )
        else:
            print(strong.to_string(index=False))
            print(
                "\nImportant: strong association alone does not prove "
                "target leakage."
            )

    # Check whether the validation partition was stratified.
    print("\n" + "=" * 70)
    print("4. VALIDATION PARTITION")
    print("=" * 70)

    if "Validation" in df.columns:
        print(
            df.groupby("Validation", dropna=False)[TARGET]
            .agg(["count", "mean"])
            .rename(columns={"mean": "backorder_rate"})
            .to_string()
        )
        print(
            "\nThe Validation column appears to identify partitions. "
            "Confirm its meaning in the dataset documentation."
        )
    else:
        print("No Validation column found.")

    # Inspect prediction file for evaluation context.
    print("\n" + "=" * 70)
    print("5. PREDICTION FILE CHECK")
    print("=" * 70)

    if PREDICTIONS_FILE.exists():
        pred = pd.read_csv(PREDICTIONS_FILE)

        print(f"Prediction rows: {len(pred):,}")
        print("Prediction columns:", list(pred.columns))

        if TARGET in pred.columns:
            print(
                "The prediction file contains actual target labels. "
                "That is expected for validation analysis, but these "
                "records must not be described as unseen production data."
            )
        else:
            print(
                "No target column in prediction file. Check how validation "
                "metrics and category summaries were generated."
            )
    else:
        print("Prediction file not found; skipping this check.")

    print("\n" + "=" * 70)
    print("6. REQUIRED MANUAL CHECKS")
    print("=" * 70)

    print(
        """
This script can flag suspicious patterns, but it cannot prove that
the model is leakage-free.

Before claiming future-backorder prediction, verify:

1. What exact moment is the prediction supposed to be made?
2. Were inventory, forecasts, sales and supplier metrics recorded
   before that moment?
3. Do local_bo_qty and pieces_past_due describe information that
   may only be known after a backorder occurs?
4. Does the dataset documentation define the Validation column?
5. Were preprocessing and model fitting performed on training rows
   only, with validation rows used only for evaluation?

Do not remove features solely because they correlate with the target.
Investigate their definitions and timing first.
"""
    )


if __name__ == "__main__":
    main()
