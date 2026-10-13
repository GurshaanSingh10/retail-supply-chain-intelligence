
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
OUT = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

# Preserve the original data and create a separate analysis copy.
df = df.copy()

# Treat -99 as a special missing/sentinel code.
# Retain an indicator so we can investigate whether missingness
# itself is informative.
for col in ["perf_6_month_avg", "perf_12_month_avg"]:
    df[f"{col}_sentinel"] = (df[col] == -99).astype("int8")
    df[col] = df[col].mask(df[col] == -99)

# Flag missing lead times without inventing replacement values.
df["lead_time_missing"] = df["lead_time"].isna().astype("int8")

# Descriptive inventory indicators.
df["inventory_status"] = "Positive"
df.loc[df["national_inv"] == 0, "inventory_status"] = "Zero"
df.loc[df["national_inv"] < 0, "inventory_status"] = "Negative"

# Only compare inventory with the forecast where forecast > 0.
df["inventory_below_3m_forecast"] = pd.NA
valid_forecast = df["forecast_3_month"] > 0
df.loc[valid_forecast, "inventory_below_3m_forecast"] = (
    df.loc[valid_forecast, "national_inv"]
    < df.loc[valid_forecast, "forecast_3_month"]
).astype("int8")

# Exclude split metadata from the analytical feature set later.
# Keep it in this file for traceability.
out_path = OUT / "train_analysis.csv"
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))

print("\nInventory status:")
print(
    df.groupby("inventory_status", observed=True)
      .agg(
          sku_count=("sku", "count"),
          backorder_rate=("went_on_backorder", "mean")
      )
      .assign(backorder_rate_pct=lambda x: (x["backorder_rate"] * 100).round(2))
      .drop(columns="backorder_rate")
      .to_string()
)

print("\nMissing lead time:", df["lead_time"].isna().sum())
print("\nPerformance sentinel counts:")
print(df[["perf_6_month_avg_sentinel",
          "perf_12_month_avg_sentinel"]].sum())

print("\nInventory below 3-month forecast among positive forecasts:")
print(df.loc[valid_forecast, "inventory_below_3m_forecast"].value_counts())
