
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

for filename in ["train.csv", "test.csv", "sample_Submission.csv"]:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)

    print(f"\n{'=' * 55}")
    print(f"FILE: {filename}")
    print(f"ROWS: {len(df):,} | COLUMNS: {len(df.columns)}")

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nMISSING VALUES:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string()
          if missing.any() else "None")

    print("\nEXACT DUPLICATE ROWS:", df.duplicated().sum())

    if "went_on_backorder" in df.columns:
        print("\nBACKORDER DISTRIBUTION:")
        print(df["went_on_backorder"].value_counts(dropna=False))
        print("\nBACKORDER PERCENTAGES:")
        print(
            (df["went_on_backorder"].value_counts(normalize=True) * 100)
            .round(2)
        )

    print("\nFIRST 3 RECORDS:")
    print(df.head(3).to_string(index=False))
