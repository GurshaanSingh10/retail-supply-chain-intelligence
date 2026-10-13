
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "validation_risk_predictions.csv"

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL LIGHT THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f7faff 0%, #ffffff 48%, #f7faff 100%);
    color: #152238;
}

.block-container {
    padding: 1.8rem 2rem 3rem;
    max-width: 1550px;
}

section[data-testid="stSidebar"] {
    background: #f0f5fc;
    border-right: 1px solid #e0e8f4;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #243650 !important;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

h1, h2, h3 {
    color: #152238 !important;
    letter-spacing: -0.7px;
}

.stApp p, .stApp label {
    color: #34445c;
}

/* Inputs, dropdowns and multiselects */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #ffffff !important;
    border-color: #d9e3f1 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] input {
    color: #243650 !important;
}

div[data-baseweb="popover"],
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: #ffffff !important;
}

div[data-baseweb="menu"] {
    background: #ffffff !important;
}

div[data-baseweb="menu"] li,
div[data-baseweb="menu"] span {
    color: #243650 !important;
}

/* Hero */
.hero {
    padding: 30px 32px;
    margin-bottom: 20px;
    border: 1px solid #dfe9f8;
    border-radius: 24px;
    background:
        radial-gradient(circle at 90% 10%, #dceaff 0, transparent 38%),
        linear-gradient(120deg, #ffffff, #f1f7ff);
    animation: fadeIn .65s ease-out both;
}

.eyebrow {
    display: inline-block;
    background: #e6f0ff;
    color: #155eef;
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 12px;
    letter-spacing: .5px;
}

.hero-subtitle {
    color: #526581 !important;
    font-size: 16px;
    line-height: 1.7;
    margin-bottom: 0;
}

/* KPI cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e3eaf5;
    border-radius: 18px;
    padding: 19px;
    min-height: 126px;
    box-shadow: 0 5px 18px rgba(33,68,120,.035);
    transition: transform .22s ease, box-shadow .22s ease,
                border-color .22s ease;
    animation: fadeIn .55s ease-out both;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(33,68,120,.10);
    border-color: #b9d2ff;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #101828;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
}

.metric-note {
    color: #64748b;
    font-size: 12px;
    margin-top: 7px;
}

.section-caption {
    color: #64748b !important;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 12px;
}

/* Panels */
.panel {
    background: #ffffff;
    border: 1px solid #e3eaf5;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 20px rgba(33,68,120,.035);
    margin-bottom: 14px;
}

.detail-card {
    background: #ffffff;
    border: 1px solid #dfe8f5;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 26px rgba(33,68,120,.055);
    margin: 8px 0 18px;
}

.risk-pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
}

.risk-track {
    height: 9px;
    width: 100%;
    background: #edf2f8;
    border-radius: 20px;
    overflow: hidden;
    margin: 12px 0 8px;
}

.risk-fill {
    height: 100%;
    border-radius: 20px;
}

.insight-card {
    background: #ffffff;
    border: 1px solid #e3eaf5;
    border-radius: 15px;
    padding: 17px;
    min-height: 120px;
    transition: transform .2s ease, box-shadow .2s ease;
}

.insight-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(33,68,120,.07);
}

div[data-testid="stDataFrame"] {
    border: 1px solid #e3eaf4;
    border-radius: 12px;
    overflow: hidden;
}

.stDownloadButton button {
    border-radius: 10px;
    border: 1px solid #d7e4f5;
    background: #ffffff;
    color: #155eef;
    font-weight: 600;
    transition: all .2s ease;
}

.stDownloadButton button:hover {
    background: #edf4ff;
    border-color: #9bbcff;
    color: #174ea6;
}

div.stButton > button {
    border-radius: 10px;
    transition: all .2s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
}

.js-plotly-plot .plotly text {
    fill: #34445c !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
            

/* Improve Streamlit dropdown contrast */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #d9e3f1 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] [data-testid="stMarkdownContainer"] {
    color: #243650 !important;
}

[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid #e3eaf5 !important;
}

[data-baseweb="menu"] li,
[data-baseweb="menu"] li * {
    color: #243650 !important;
}

/* Make the SKU detail view more compact */
.detail-card {
    padding: 20px 24px;
    margin: 8px 0 14px;
}

[data-testid="stMetricValue"] {
    color: #152238;
}

/* More compact chart */
div[data-testid="stPlotlyChart"] {
    margin-bottom: 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================
if not DATA_PATH.exists():
    st.error(f"Prediction file not found: {DATA_PATH}")
    st.info("Run: python scripts/export_risk_predictions.py")
    st.stop()

df = pd.read_csv(DATA_PATH)

required = {
    "sku",
    "went_on_backorder",
    "predicted_backorder_probability",
    "risk_category",
    "national_inv",
    "forecast_3_month",
}

missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

df["sku"] = df["sku"].astype(str)
df["risk_category"] = df["risk_category"].fillna("Unclassified")
df["predicted_backorder_probability"] = (
    df["predicted_backorder_probability"].clip(0, 1)
)

for col in ["lead_time", "in_transit_qty"]:
    if col not in df.columns:
        df[col] = float("nan")

risk_options = ["Low", "Medium", "High", "Unclassified"]

risk_colors = {
    "Low": "#3985ed",
    "Medium": "#e9a23b",
    "High": "#e85e69",
    "Unclassified": "#8796aa",
}

risk_backgrounds = {
    "Low": "#eaf3ff",
    "Medium": "#fff4df",
    "High": "#ffebed",
    "Unclassified": "#edf1f6",
}


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 📦 Supply Chain")
st.sidebar.caption("INTELLIGENCE PLATFORM")
st.sidebar.divider()
st.sidebar.markdown("### Explore inventory")

available = [r for r in risk_options if r in df["risk_category"].unique()]

selected_risks = st.sidebar.multiselect(
    "Risk category",
    options=available,
    default=available,
)

min_risk = st.sidebar.slider(
    "Minimum predicted risk (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
)

search_sku = st.sidebar.text_input(
    "Search SKU",
    placeholder="Enter SKU ID...",
)

top_n = st.sidebar.slider(
    "Maximum SKUs in table",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
)

st.sidebar.divider()
st.sidebar.caption(
    "Held-out validation records. Predictions are experimental, "
    "not live inventory alerts."
)


# =========================================================
# FILTER RECORDS
# =========================================================
filtered = df[
    df["risk_category"].isin(selected_risks)
    & (df["predicted_backorder_probability"] >= min_risk / 100)
].copy()

if search_sku.strip():
    filtered = filtered[
        filtered["sku"].str.contains(
            search_sku.strip(), case=False, regex=False
        )
    ]

total = len(filtered)
actual_backorders = int(filtered["went_on_backorder"].sum())

actual_rate = (
    filtered["went_on_backorder"].mean() * 100 if total else 0
)
avg_risk = (
    filtered["predicted_backorder_probability"].mean() * 100
    if total else 0
)
high_count = int((filtered["risk_category"] == "High").sum())


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">● &nbsp; SUPPLY CHAIN ANALYTICS</div>
    <h1>Supply Chain Risk Intelligence</h1>
    <p class="hero-subtitle">
        Understand backorder risk. Explore inventory signals.
        Make better-informed supply chain decisions.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Validation dashboard · Actual outcomes are known for these records. "
    "Predictions are experimental."
)


# =========================================================
# KPI CARDS
# =========================================================
k1, k2, k3, k4 = st.columns(4)

cards = [
    (k1, "📦", "SKUs in view", f"{total:,}", "After applying filters"),
    (
        k2, "⚠️", "Actual backorders", f"{actual_backorders:,}",
        f"{actual_rate:.1f}% of filtered records",
    ),
    (
        k3, "◉", "Average predicted risk", f"{avg_risk:.1f}%",
        "Mean model probability",
    ),
    (
        k4, "↗", "High-risk SKUs", f"{high_count:,}",
        f"{high_count / total * 100:.1f}% of view" if total else "No records",
    ),
]

for col, icon, label, value, note in cards:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{icon} &nbsp; {label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# INTERACTIVE CHARTS
# =========================================================
left, middle, right = st.columns([1, 1.2, 1])

counts = (
    filtered["risk_category"]
    .value_counts()
    .reindex(risk_options, fill_value=0)
    .rename_axis("Risk category")
    .reset_index(name="SKUs")
)

with left:
    st.markdown("### Risk distribution")
    st.markdown(
        '<div class="section-caption">SKUs by predicted risk band</div>',
        unsafe_allow_html=True,
    )

    fig = px.bar(
        counts,
        x="Risk category",
        y="SKUs",
        color="Risk category",
        color_discrete_map=risk_colors,
        text="SKUs",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(l=5, r=5, t=20, b=5),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#34445c"),
        xaxis_title=None,
        yaxis_title="Number of SKUs",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf1f7", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

with middle:
    st.markdown("### Probability distribution")
    st.markdown(
        '<div class="section-caption">Predicted backorder probabilities</div>',
        unsafe_allow_html=True,
    )

    fig = px.histogram(
        filtered,
        x="predicted_backorder_probability",
        nbins=30,
        color_discrete_sequence=["#76a9fa"],
    )
    fig.update_layout(
        height=320,
        margin=dict(l=5, r=5, t=20, b=5),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#34445c"),
        xaxis_title="Predicted probability",
        yaxis_title="Number of SKUs",
        bargap=0.08,
    )
    fig.update_xaxes(tickformat=".0%", gridcolor="#edf1f7")
    fig.update_yaxes(gridcolor="#edf1f7")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Risk mix")
    st.markdown(
        '<div class="section-caption">Share of filtered SKUs</div>',
        unsafe_allow_html=True,
    )

    pie_data = counts[counts["SKUs"] > 0]

    if not pie_data.empty:
        fig = px.pie(
            pie_data,
            names="Risk category",
            values="SKUs",
            hole=0.64,
            color="Risk category",
            color_discrete_map=risk_colors,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "%{label}<br>SKUs: %{value:,}"
                "<br>Share: %{percent}<extra></extra>"
            ),
        )
        fig.update_layout(
            height=320,
            margin=dict(l=5, r=5, t=20, b=5),
            paper_bgcolor="white",
            font=dict(color="#34445c"),
            legend=dict(orientation="h", y=-0.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match the current filters.")


# =========================================================
# CATEGORY PERFORMANCE
# =========================================================
st.write("")
st.markdown("### Risk category performance")
st.markdown(
    '<div class="section-caption">'
    'Compare predicted risk with observed validation outcomes.'
    '</div>',
    unsafe_allow_html=True,
)

if total:
    performance = (
        filtered.groupby("risk_category", observed=False)
        .agg(
            SKUs=("sku", "count"),
            Actual_Backorders=("went_on_backorder", "sum"),
            Average_Predicted_Risk=(
                "predicted_backorder_probability", "mean"
            ),
            Actual_Backorder_Rate=("went_on_backorder", "mean"),
        )
        .reset_index()
    )

    performance["Average_Predicted_Risk"] *= 100
    performance["Actual_Backorder_Rate"] *= 100

    performance = performance.rename(columns={
        "risk_category": "Risk category",
        "Actual_Backorders": "Actual backorders",
        "Average_Predicted_Risk": "Avg predicted risk (%)",
        "Actual_Backorder_Rate": "Actual backorder rate (%)",
    })

    st.dataframe(
        performance.round(2),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# INVENTORY INSIGHTS
# =========================================================
st.write("")
st.markdown("### Inventory insights")

if total:
    high_df = filtered[filtered["risk_category"] == "High"]
    low_inventory = filtered[filtered["national_inv"] <= 0]
    positive_forecast = filtered[filtered["forecast_3_month"] > 0]

    high_rate = (
        high_df["went_on_backorder"].mean() * 100 if len(high_df) else 0
    )

    i1, i2, i3 = st.columns(3)

    insight_data = [
        (
            i1, "🎯 High-risk group", f"{len(high_df):,} SKUs",
            f"Observed backorder rate: {high_rate:.1f}% in this validation group.",
        ),
        (
            i2, "📉 Inventory at or below zero",
            f"{len(low_inventory):,} SKUs",
            "Review stock position, forecast and replenishment policies.",
        ),
        (
            i3, "📈 Positive 3-month forecast",
            f"{len(positive_forecast):,} SKUs",
            "Compare inventory with expected demand before making decisions.",
        ),
    ]

    for col, title, metric, description in insight_data:
        with col:
            st.markdown(
                f"""
                <div class="insight-card">
                    <strong>{title}</strong>
                    <h2 style="margin:10px 0;color:#152238 !important">{metric}</h2>
                    <p style="font-size:12px;color:#64748b !important">
                        {description}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# SKU RISK REGISTER
# =========================================================
st.write("")
st.markdown("### SKU risk explorer")
st.markdown(
    '<div class="section-caption">'
    'Search, sort, inspect, and export filtered SKU records.'
    '</div>',
    unsafe_allow_html=True,
)

if total:
    display_cols = [
        c for c in [
            "sku",
            "risk_category",
            "predicted_backorder_probability",
            "national_inv",
            "forecast_3_month",
            "lead_time",
            "in_transit_qty",
            "went_on_backorder",
        ]
        if c in filtered.columns
    ]

    table = filtered[display_cols].copy()
    table["predicted_backorder_probability"] *= 100

    table = table.rename(columns={
        "sku": "SKU",
        "risk_category": "Risk category",
        "predicted_backorder_probability": "Predicted risk (%)",
        "national_inv": "Current inventory",
        "forecast_3_month": "3-month forecast",
        "lead_time": "Lead time",
        "in_transit_qty": "In transit",
        "went_on_backorder": "Actual backorder",
    })

    table = table.sort_values("Predicted risk (%)", ascending=False)

    st.dataframe(
        table.head(top_n).round(2),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.download_button(
        "⬇ Download filtered SKU register",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="supply_chain_risk_register.csv",
        mime="text/csv",
    )

    # Download the generated action report
    st.markdown("### 📋 High-risk SKU action report")
    st.caption(
        "Download a prioritised review report generated from the "
        "experimental validation predictions."
    )

    action_report_path = (
        ROOT / "data" / "processed" / "high_risk_action_report.xlsx"
    )
    action_csv_path = (
        ROOT / "data" / "processed" / "high_risk_action_report.csv"
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        if action_report_path.exists():
            with open(action_report_path, "rb") as file:
                st.download_button(
                    "⬇ Download Excel action report",
                    data=file.read(),
                    file_name="high_risk_action_report.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="download_action_excel",
                )
        else:
            st.info("Excel action report not found. Run the report script.")

    with download_col2:
        if action_csv_path.exists():
            with open(action_csv_path, "rb") as file:
                st.download_button(
                    "⬇ Download CSV action report",
                    data=file.read(),
                    file_name="high_risk_action_report.csv",
                    mime="text/csv",
                    key="download_action_csv",
                )
        else:
            st.info("CSV action report not found. Run the report script.")


    # =====================================================
    # SKU DETAIL EXPLORER
    # =====================================================
    st.write("")
    st.markdown("### SKU detail explorer")

    sku_options = filtered.sort_values(
        "predicted_backorder_probability", ascending=False
    )["sku"].tolist()

    selected_sku = st.selectbox(
        "Select a SKU to inspect",
        options=sku_options,
        key="selected_sku",
    )

    details = filtered[filtered["sku"] == selected_sku].iloc[0]

    probability = float(details["predicted_backorder_probability"])
    category = str(details["risk_category"])
    inventory = float(details["national_inv"])
    forecast = float(details["forecast_3_month"])
    lead_time = details["lead_time"]
    in_transit = details["in_transit_qty"]

    risk_color = risk_colors.get(category, "#8796aa")
    risk_bg = risk_backgrounds.get(category, "#edf1f6")

    st.markdown(
        f"""
        <div class="detail-card">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;gap:20px;flex-wrap:wrap">
                <div>
                    <div style="font-size:12px;color:#64748b;
                                font-weight:600;letter-spacing:.5px">
                        SELECTED INVENTORY ITEM
                    </div>
                    <h2 style="margin:8px 0 4px;
                               color:#152238 !important">
                        SKU {selected_sku}
                    </h2>
                    <span class="risk-pill"
                          style="background:{risk_bg};color:{risk_color}">
                        {category} risk
                    </span>
                </div>
                <div style="min-width:210px">
                    <div style="font-size:12px;color:#64748b">
                        Predicted backorder probability
                    </div>
                    <div style="font-size:34px;font-weight:800;
                                color:{risk_color};letter-spacing:-1px">
                        {probability:.1%}
                    </div>
                    <div class="risk-track">
                        <div class="risk-fill"
                             style="width:{probability * 100:.1f}%;
                                    background:{risk_color}">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Current inventory", f"{inventory:,.0f}")
    d2.metric("3-month forecast", f"{forecast:,.0f}")
    d3.metric(
        "Lead time",
        f"{lead_time:g}" if pd.notna(lead_time) else "Unavailable",
    )
    d4.metric(
        "In transit",
        f"{in_transit:,.0f}" if pd.notna(in_transit) else "Unavailable",
    )

    # Inventory vs forecast visual
    st.markdown("#### Inventory versus forecast")

    comparison = pd.DataFrame({
        "Measure": ["Current inventory", "3-month forecast"],
        "Quantity": [inventory, forecast],
    })

    fig = px.bar(
        comparison,
        x="Measure",
        y="Quantity",
        color="Measure",
        color_discrete_map={
            "Current inventory": "#3985ed",
            "3-month forecast": "#a7c8ff",
        },
        text="Quantity",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        height=260,
        margin=dict(l=5, r=15, t=15, b=5),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#34445c"),
        xaxis_title=None,
        yaxis_title="Quantity",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf1f7", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    # Rule-based review recommendations
    st.markdown("#### Suggested review points")

    if category == "High":
        st.warning(
            "High model-predicted risk. Review inventory, demand forecasts, "
            "supplier lead time, and replenishment policy."
        )
    elif category == "Medium":
        st.info(
            "Moderate model-predicted risk. Monitor forecast changes "
            "and review stock coverage."
        )
    else:
        st.success(
            "Low model-predicted risk under the current model. "
            "Continue routine monitoring."
        )

    if inventory <= 0:
        st.warning(
            "Inventory is zero or negative. Verify the stock position "
            "and investigate replenishment requirements."
        )

    if forecast > 0 and inventory < forecast:
        st.info(
            "Inventory is below the 3-month forecast. Review demand timing, "
            "incoming supply, and replenishment requirements. This is a "
            "screening signal, not proof of a future stockout."
        )

    if pd.notna(lead_time) and lead_time > 0:
        st.caption(
            f"Recorded lead time: {lead_time:g}. Confirm units and "
            "expected delivery timing before making decisions."
        )

    st.caption(
        "Review prompts are rule-based and are not causal explanations "
        "of the model prediction. Actual validation outcomes are known "
        "for the records shown."
    )

else:
    st.info("No SKUs match the current filters. Adjust the sidebar controls.")

st.divider()
st.caption(
    "Supply Chain Risk Intelligence · Experimental analytics prototype. "
    "Risk bands are provisional; predictions are not guarantees. "
    "Validation outcomes are known for the displayed records."
)
