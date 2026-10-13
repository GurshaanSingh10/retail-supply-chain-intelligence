
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

train = pd.read_csv(DATA / "train.csv", low_memory=False)
test = pd.read_csv(DATA / "test.csv", low_memory=False)
sample = pd.read_csv(DATA / "sample_Submission.csv", low_memory=False)

print("=== SHAPE ===")
print("Train:", train.shape)
print("Test:", test.shape)
print("Sample submission:", sample.shape)

print("\n=== ALL TRAINING COLUMNS ===")
for col in train.columns:
    print(f"{col}: {train[col].dtype}")

print("\n=== TARGET COUNTS ===")
print(train["went_on_backorder"].value_counts(dropna=False))
print("\nTarget rate (%):")
print(round(train["went_on_backorder"].mean() * 100, 2))

print("\n=== SKU CHECKS ===")
for name, df in [("train", train), ("test", test)]:
    print(f"\n{name}")
    print("Unique SKUs:", df["sku"].nunique())
    print("Duplicate SKU rows:", df["sku"].duplicated().sum())
    print("Missing SKUs:", df["sku"].isna().sum())

overlap = set(train["sku"]) & set(test["sku"])
print("\nTrain/test SKU overlap:", len(overlap))

print("\n=== MISSINGNESS: TRAIN ===")
missing = train.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n=== NUMERIC FEATURE SUMMARY ===")
print(train.select_dtypes(include="number").describe().T.to_string())

print("\n=== NEGATIVE VALUES ===")
numeric = train.select_dtypes(include="number")
for col in numeric.columns:
    if col not in ["went_on_backorder"]:
        count = (numeric[col] < 0).sum()
        if count:
            print(f"{col}: {count:,} negative values")

print("\n=== TRAIN/TEST COLUMN DIFFERENCES ===")
print("Only in train:", sorted(set(train.columns) - set(test.columns)))
print("Only in test:", sorted(set(test.columns) - set(train.columns)))

print("\n=== SAMPLE SUBMISSION ===")
print("Columns:", sample.columns.tolist())
print("Rows:", len(sample))
