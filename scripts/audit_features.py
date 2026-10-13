
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
train_path = ROOT / "data" / "raw" / "train.csv"
processed_path = ROOT / "data" / "processed" / "train_analysis.csv"

raw = pd.read_csv(train_path)
processed = pd.read_csv(processed_path)

print("\n=== RAW DATA ===")
print("Shape:", raw.shape)
print("Duplicate rows:", raw.duplicated().sum())
print("Duplicate SKUs:", raw["sku"].duplicated().sum())
print("\nValidation counts:")
print(raw["Validation"].value_counts(dropna=False).sort_index())
print("\nTarget rates by validation partition:")
print(raw.groupby("Validation")["went_on_backorder"].agg(["count", "mean"]))

print("\n=== FEATURE AUDIT ===")
for col in raw.columns:
    if col in ["sku", "Validation", "went_on_backorder"]:
        continue

    print(
        f"{col:25} "
        f"dtype={str(raw[col].dtype):10} "
        f"missing={raw[col].isna().sum():5} "
        f"unique={raw[col].nunique(dropna=False):5}"
    )

print("\n=== TARGET ASSOCIATIONS ===")
for col in [
    "national_inv",
    "lead_time",
    "in_transit_qty",
    "forecast_3_month",
    "forecast_6_month",
    "forecast_9_month",
    "sales_1_month",
    "sales_3_month",
    "sales_6_month",
    "sales_9_month",
    "min_bank",
    "potential_issue",
    "pieces_past_due",
    "perf_6_month_avg",
    "perf_12_month_avg",
    "local_bo_qty",
    "deck_risk",
    "oe_constraint",
    "ppap_risk",
    "stop_auto_buy",
    "rev_stop",
]:
    if col not in raw.columns:
        continue

    print(f"\n{col}")
    print(
        raw.groupby("went_on_backorder")[col]
        .agg(["count", "mean", "median"])
        .round(3)
    )

print("\n=== ENGINEERED COLUMNS ===")
print("Raw columns:", len(raw.columns))
print("Processed columns:", len(processed.columns))
print("Added columns:", sorted(set(processed.columns) - set(raw.columns)))
print("\nProcessed shape:", processed.shape)
