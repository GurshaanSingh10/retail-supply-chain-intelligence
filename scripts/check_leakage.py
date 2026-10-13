
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "train_analysis.csv")

target = "went_on_backorder"

print("=== TARGET CORRELATIONS ===")
corr = (
    df.select_dtypes(include="number")
      .corr()[target]
      .drop(target)
      .sort_values(key=lambda s: s.abs(), ascending=False)
)
print(corr.to_string())

print("\n=== BACKORDER RATE BY LOCAL BACKORDER QUANTITY ===")
print(
    df.assign(local_bo_positive=df["local_bo_qty"] > 0)
      .groupby("local_bo_positive")[target]
      .agg(["count", "mean"])
      .to_string()
)

print("\n=== BACKORDER RATE BY PAST-DUE QUANTITY ===")
print(
    df.assign(past_due_positive=df["pieces_past_due"] > 0)
      .groupby("past_due_positive")[target]
      .agg(["count", "mean"])
      .to_string()
)

print("\n=== BACKORDER RATE BY INVENTORY STATUS ===")
print(
    df.assign(
        inventory_status=df["national_inv"].apply(
            lambda x: "Negative" if x < 0 else
            ("Zero" if x == 0 else "Positive")
        )
    )
    .groupby("inventory_status")[target]
    .agg(["count", "mean"])
    .to_string()
)

