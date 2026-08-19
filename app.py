"""
Bean Counter — Specialty Coffee Expansion & Unit-Economics Simulator
======================================================================
A live, interactive proof-of-work project: model whether a specialty coffee
roaster should grow through B2B wholesale accounts or company-owned retail
cafes, using the same unit-economics vocabulary (CAC, ACV, churn, payback,
LTV:CAC, TAM/SAM/SOM) that any B2B SaaS or consumer-retail startup uses.

Run locally:      streamlit run app.py
Deploy:            push this folder to a public GitHub repo, then deploy
                    free on Streamlit Community Cloud (see README.md).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    REGION_DEFAULTS,
    cac_payback_months,
    cohort_retention_curve,
    generate_takeaway,
    ltv_to_cac,
    market_sizing,
    retail_projection,
    retail_unit_economics,
    wholesale_cohort_projection,
)

# ---------------------------------------------------------------------------
# Palette — warm coffee-roaster tones, validated for colorblind-safe
# separation with scripts/validate_palette.js (dataviz method). All three
# categorical checks pass at surface #f7f1e8: chroma floor, CVD adjacent
# separation, and normal-vision floor.
# ---------------------------------------------------------------------------
AMBER = "#c9821f"       # scenario: Conservative
RUST = "#a8442b"        # scenario: Base / primary accent
GREEN = "#1baf7a"       # scenario: Aggressive
INK = "#2b1810"
SECONDARY_INK = "#6b5645"
MUTED = "#9c8874"
GRID = "#e6dcc8"
SURFACE = "#f7f1e8"

SCENARIO_COLORS = {"Conservative": AMBER, "Base": RUST, "Aggressive": GREEN}

st.set_page_config(
    page_title="Bean Counter — Coffee Expansion Simulator",
    page_icon="☕",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("☕ Bean Counter")
st.caption(
    "A unit-economics model for expanding a specialty coffee business — "
    "wholesale accounts vs. company-owned cafes. Adjust assumptions in the "
    "sidebar; the metrics, charts, and takeaway below update immediately."
)
st.info(
    "Inputs are planning assumptions grounded in specialty coffee industry "
    "norms — informed by multi-year independent research (20+ industry "
    "contacts, 5 producing regions, 10+ cities) — not claimed market data. "
    "Every value is adjustable."
)

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Assumptions")

with st.sidebar.expander("Wholesale / B2B channel", expanded=True):
    acv = st.slider("Average Contract Value — ACV ($/yr)", 2000, 30000, 8000, step=500)
    cac = st.slider("Customer Acquisition Cost — CAC ($)", 300, 5000, 1200, step=100)
    gross_margin_ws = st.slider("Wholesale gross margin (%)", 30, 75, 58, step=1) / 100
    monthly_churn = st.slider("Monthly logo churn (%)", 0.5, 8.0, 2.5, step=0.1) / 100
    sales_cycle = st.slider("Sales cycle length (months)", 0.5, 6.0, 2.0, step=0.5)
    new_accounts = st.slider("New accounts signed / month", 1, 20, 4, step=1)

with st.sidebar.expander("Retail channel", expanded=False):
    build_out = st.slider("New cafe build-out cost ($)", 100_000, 500_000, 220_000, step=10_000)
    rent = st.slider("Monthly rent ($)", 3_000, 25_000, 11_000, step=500)
    avg_ticket = st.slider("Average ticket ($)", 3.0, 12.0, 6.25, step=0.25)
    daily_tx = st.slider("Daily transactions", 50, 600, 220, step=10)
    cogs_pct = st.slider("COGS (%)", 15, 45, 28, step=1) / 100
    labor_pct = st.slider("Labor (% of revenue)", 15, 45, 32, step=1) / 100
    cafes_per_year = st.slider("New cafes opened / year", 0.0, 6.0, 1.0, step=0.5)

with st.sidebar.expander("Market sizing", expanded=False):
    region = st.selectbox("Target region", list(REGION_DEFAULTS.keys()), index=0)
    region_defaults = REGION_DEFAULTS[region]
    addressable_accounts = st.number_input(
        "Addressable business accounts in region (cafes, offices, restaurants)",
        min_value=200,
        max_value=20000,
        value=region_defaults["addressable_accounts"],
        step=100,
    )
    qualified_pct = st.slider(
        "% qualified as specialty-relevant (SAM)",
        5,
        60,
        int(region_defaults["qualified_pct"] * 100),
        step=1,
    ) / 100

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
payback = cac_payback_months(cac, acv, gross_margin_ws)
ltc = ltv_to_cac(acv, gross_margin_ws, monthly_churn, cac)
ws_proj = wholesale_cohort_projection(acv, gross_margin_ws, monthly_churn, new_accounts, months=36)
retention = cohort_retention_curve(monthly_churn, months=24)
retail_unit = retail_unit_economics(build_out, rent, avg_ticket, daily_tx, cogs_pct, labor_pct)
retail_proj = retail_projection(
    build_out, rent, avg_ticket, daily_tx, cogs_pct, labor_pct, cafes_per_year, months=36
)
mkt = market_sizing(addressable_accounts, qualified_pct, acv, new_accounts, months=36)

ws_24mo_profit = ws_proj[(ws_proj.scenario == "Base") & (ws_proj.month == 24)]["cum_gross_profit"].iloc[0]
retail_24mo_cash = retail_proj[retail_proj.month == 24]["cum_cash_position"].iloc[0]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.subheader("Headline metrics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Wholesale CAC payback", f"{payback:.1f} mo" if payback != float("inf") else "never")
k2.metric("LTV : CAC", f"{ltc:.1f}x" if ltc != float("inf") else "∞")
k3.metric(
    "Retail build-out payback",
    f"{retail_unit['payback_months']:.1f} mo" if retail_unit["payback_months"] != float("inf") else "never",
)
k4.metric("Modeled 36-mo SAM penetration", f"{mkt['penetration_of_sam']:.0%}")

st.divider()

# ---------------------------------------------------------------------------
# Executive takeaway
# ---------------------------------------------------------------------------
st.subheader("Executive takeaway")
takeaways = generate_takeaway(
    cac_payback=payback,
    ltv_cac=ltc,
    sales_cycle_months=sales_cycle,
    retail_payback=retail_unit["payback_months"],
    wholesale_24mo_profit=ws_24mo_profit,
    retail_24mo_cash=retail_24mo_cash,
    som_penetration=mkt["penetration_of_sam"],
)
for line in takeaways:
    st.markdown(f"- {line}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Wholesale growth", "Retail economics", "Market sizing"])

with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("**36-month MRR by scenario** (Conservative / Base / Aggressive)")
        fig = go.Figure()
        for scenario, color in SCENARIO_COLORS.items():
            d = ws_proj[ws_proj.scenario == scenario]
            fig.add_trace(
                go.Scatter(
                    x=d["month"],
                    y=d["mrr"],
                    mode="lines",
                    name=scenario,
                    line=dict(color=color, width=2),
                )
            )
        fig.update_layout(
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK),
            xaxis=dict(title="Month", gridcolor=GRID, showline=True, linecolor=MUTED),
            yaxis=dict(title="MRR ($)", gridcolor=GRID, tickformat="$,.0f"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=30, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Cohort retention** (one signup class, 24 mo)")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=retention["month"],
                y=retention["pct_retained"],
                mode="lines",
                line=dict(color=RUST, width=2),
                fill="tozeroy",
                fillcolor="rgba(168,68,43,0.12)",
            )
        )
        fig2.update_layout(
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK),
            xaxis=dict(title="Month", gridcolor=GRID),
            yaxis=dict(title="% of cohort remaining", gridcolor=GRID, range=[0, 100]),
            margin=dict(t=30, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        f"At {monthly_churn:.1%} monthly churn, half of any signup cohort has "
        f"churned out by month {int(np.log(0.5) / np.log(1 - monthly_churn)) if monthly_churn > 0 else '∞'}."
    )

with tab2:
    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**Cumulative cash position** — opening {cafes_per_year:.1f} cafes/year")
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=retail_proj["month"],
                y=retail_proj["cum_cash_position"],
                mode="lines",
                line=dict(color=RUST, width=2),
            )
        )
        fig3.add_hline(y=0, line_dash="dot", line_color=MUTED)
        fig3.update_layout(
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=dict(color=INK),
            xaxis=dict(title="Month", gridcolor=GRID),
            yaxis=dict(title="Cumulative cash ($)", gridcolor=GRID, tickformat="$,.0f"),
            margin=dict(t=30, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)
    with right:
        st.markdown("**Per-store monthly economics**")
        st.metric("Monthly revenue", f"${retail_unit['monthly_revenue']:,.0f}")
        st.metric("Monthly gross profit", f"${retail_unit['monthly_gross_profit']:,.0f}")
        st.metric("Annual gross profit / store", f"${retail_unit['annual_gross_profit']:,.0f}")

with tab3:
    st.markdown(f"**TAM → SAM → SOM** for {region} (36-month horizon)")
    funnel_labels = ["TAM (all addressable accounts)", "SAM (specialty-qualified)", "SOM (modeled capture)"]
    funnel_values = [mkt["tam_accounts"], mkt["sam_accounts"], mkt["som_accounts"]]
    funnel_dollars = [mkt["tam_dollars"], mkt["sam_dollars"], mkt["som_dollars"]]
    fig4 = go.Figure(
        go.Funnel(
            y=funnel_labels,
            x=funnel_values,
            marker=dict(color=[AMBER, RUST, GREEN]),
            textinfo="value",
        )
    )
    fig4.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK),
        margin=dict(t=30, b=10, l=10, r=10),
        height=360,
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        st.metric("TAM", f"${mkt['tam_dollars']:,.0f}")
        st.metric("SAM", f"${mkt['sam_dollars']:,.0f}")
        st.metric("SOM (36 mo)", f"${mkt['som_dollars']:,.0f}")

st.divider()
st.caption(
    "Built by Ashlyn Chi Garcia (Tepper MS Strategy, CMU). Model logic in "
    "model.py; interface in Streamlit and Plotly. All assumptions above "
    "are adjustable."
)