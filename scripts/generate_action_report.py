
from pathlib import Path

import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "validation_risk_predictions.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
CSV_OUTPUT = OUTPUT_DIR / "high_risk_action_report.csv"
EXCEL_OUTPUT = OUTPUT_DIR / "high_risk_action_report.xlsx"


def build_recommendation(row):
    """Generate rule-based review recommendations for each SKU."""
    actions = []

    inventory = row["national_inv"]
    forecast = row["forecast_3_month"]
    risk = row["predicted_backorder_probability"]
    category = row["risk_category"]

    if inventory <= 0:
        actions.append(
            "URGENT REVIEW: Verify zero or negative inventory"
        )

    if forecast > 0 and inventory < forecast:
        actions.append(
            "Review stock coverage against 3-month forecast"
        )

    if category == "High":
        actions.append(
            "Prioritise supplier and replenishment review"
        )
    elif category == "Medium":
        actions.append(
            "Monitor demand and inventory position"
        )

    if risk >= 0.90:
        actions.append(
            "Very high model score: verify underlying inventory data"
        )

    if not actions:
        actions.append("Routine inventory monitoring")

    return " | ".join(actions)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find prediction file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "sku",
        "risk_category",
        "predicted_backorder_probability",
        "national_inv",
        "forecast_3_month",
        "lead_time",
        "in_transit_qty",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Ensure probability is represented on a 0-1 scale.
    df["predicted_backorder_probability"] = pd.to_numeric(
        df["predicted_backorder_probability"],
        errors="coerce",
    )

    df["national_inv"] = pd.to_numeric(
        df["national_inv"], errors="coerce"
    )
    df["forecast_3_month"] = pd.to_numeric(
        df["forecast_3_month"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "predicted_backorder_probability",
            "national_inv",
            "forecast_3_month",
        ]
    )

    # Select SKUs for review using the existing risk categories.
    report = df[
        df["risk_category"].isin(["High", "Medium"])
        | (df["national_inv"] <= 0)
    ].copy()

    report["predicted_risk_percent"] = (
        report["predicted_backorder_probability"] * 100
    ).round(2)

    report["inventory_below_forecast"] = (
        (report["forecast_3_month"] > 0)
        & (report["national_inv"] < report["forecast_3_month"])
    )

    report["recommended_action"] = report.apply(
        build_recommendation, axis=1
    )

    # Prioritise by risk, then flag stock at or below zero.
    report["inventory_priority"] = report["national_inv"].apply(
        lambda value: (
            "Inventory at/below zero" if value <= 0
            else "Positive inventory"
        )
    )

    report = report.sort_values(
        by=[
            "predicted_backorder_probability",
            "inventory_below_forecast",
        ],
        ascending=[False, False],
    )

    # Select and rename columns for a clean business report.
    output_columns = [
        "sku",
        "risk_category",
        "predicted_risk_percent",
        "national_inv",
        "forecast_3_month",
        "inventory_below_forecast",
        "lead_time",
        "in_transit_qty",
        "inventory_priority",
        "recommended_action",
    ]

    report = report[output_columns].rename(columns={
        "sku": "SKU",
        "risk_category": "Risk Category",
        "predicted_risk_percent": "Predicted Risk (%)",
        "national_inv": "Current Inventory",
        "forecast_3_month": "3-Month Forecast",
        "inventory_below_forecast": "Inventory Below Forecast",
        "lead_time": "Lead Time",
        "in_transit_qty": "In-Transit Quantity",
        "inventory_priority": "Inventory Status",
        "recommended_action": "Recommended Review Action",
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(CSV_OUTPUT, index=False)

    try:
        with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
            report.to_excel(
                writer,
                sheet_name="Action Report",
                index=False,
            )

            summary = (
                report.groupby("Risk Category")
                .agg(
                    SKUs=("SKU", "count"),
                    Average_Predicted_Risk=("Predicted Risk (%)", "mean"),
                    Inventory_At_or_Below_Zero=(
                        "Inventory Status",
                        lambda s: (s == "Inventory at/below zero").sum(),
                    ),
                )
                .reset_index()
            )

            summary.to_excel(
                writer,
                sheet_name="Risk Summary",
                index=False,
            )

            for sheet in writer.sheets.values():
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions

                for column_cells in sheet.columns:
                    max_length = max(
                        len(str(cell.value or ""))
                        for cell in column_cells
                    )
                    sheet.column_dimensions[
                        column_cells[0].column_letter
                    ].width = min(max_length + 2, 55)

    except ImportError:
        print(
            "openpyxl is not installed, so the Excel file was not created."
        )
        print("Install it with: python -m pip install openpyxl")

    print("\nACTION REPORT CREATED")
    print(f"Rows included: {len(report):,}")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Excel: {EXCEL_OUTPUT}")

    if len(report):
        print("\nRisk category counts:")
        print(report["Risk Category"].value_counts().to_string())

        print("\nInventory at or below zero:")
        print(
            (report["Inventory Status"] == "Inventory at/below zero")
            .sum()
        )


if __name__ == "__main__":
    main()
