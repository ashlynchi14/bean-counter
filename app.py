"""
Bean Counter — Specialty Coffee Expansion & Unit-Economics Simulator
======================================================================
Strategic question: a specialty coffee company wants to grow. Should it
deploy capital toward acquiring wholesale accounts, or toward opening its
own cafes? This app structures that decision, models the unit economics of
both channels, and stress-tests which assumptions would flip the answer.

Run locally:      streamlit run app.py
Deploy:            push this folder to a public GitHub repo, then deploy
                    free on Streamlit Community Cloud (see README.md).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    COMPARISON_HORIZON_MONTHS,
    REGION_DEFAULTS,
    cac_payback_months,
    compute_breakevens,
    generate_strategic_summary,
    ltv_cac_sensitivity_to_churn,
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
# separation, and normal-vision floor. Amber/Rust/Green identify the three
# growth *scenarios* everywhere in the app; the TAM/SAM/SOM chart uses a
# separate single-hue (rust) sequential ramp, since those are nested market
# sets, not distinct categories.
# ---------------------------------------------------------------------------
AMBER = "#c9821f"       # scenario: Conservative
RUST = "#a8442b"        # scenario: Base / primary accent
GREEN = "#1baf7a"       # scenario: Aggressive
INK = "#2b1810"
MUTED = "#9c8874"
GRID = "#e6dcc8"
SURFACE = "#f7f1e8"
RUST_RAMP = ["#e3b49b", "#c17f57", "#a8442b"]  # light -> dark, for TAM/SAM/SOM

SCENARIO_COLORS = {"Conservative": AMBER, "Base": RUST, "Aggressive": GREEN}
MARKET_SIZING_MONTHS = 36  # horizon for the SOM projection and MRR/cash charts


def esc(text: str) -> str:
    """Escape '$' before passing a generated string into st.markdown/st.info
    — Streamlit's markdown renderer treats a pair of '$' as inline LaTeX,
    which mangles ordinary sentences that happen to contain two dollar
    amounts."""
    return text.replace("$", r"\$")


def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


st.set_page_config(
    page_title="Bean Counter — Coffee Expansion Simulator",
    page_icon="☕",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header — lead with the decision, not the dashboard
# ---------------------------------------------------------------------------
st.title("☕ Bean Counter")
st.subheader(
    "A specialty coffee company wants to grow. Should it put capital into "
    "acquiring wholesale accounts, or into opening its own cafes?"
)
st.caption(
    "Adjust assumptions in the sidebar. The metrics, the recommendation, "
    "and the break-even thresholds below all update immediately."
)
st.info(
    "Inputs are planning assumptions grounded in specialty coffee industry "
    "norms — informed by multi-year independent research (20+ industry "
    "contacts, 5 producing regions, 10+ cities) — not claimed market data. "
    "Every value is adjustable."
)

# ---------------------------------------------------------------------------
# Sidebar — inputs, grouped by decision area
# ---------------------------------------------------------------------------
st.sidebar.header("Assumptions")

with st.sidebar.expander("Wholesale economics", expanded=True):
    acv = st.slider("Average Contract Value — ACV ($/yr)", 2000, 30000, 8000, step=500, format="dollar")
    cac = st.slider("Customer Acquisition Cost — CAC ($)", 300, 5000, 1200, step=100, format="dollar")
    gross_margin_ws = st.slider("Wholesale gross margin (%)", 30, 75, 58, step=1, format="%d%%") / 100
    monthly_churn = st.slider("Monthly logo churn (%)", 0.5, 8.0, 2.5, step=0.1, format="%.1f%%") / 100

with st.sidebar.expander("Sales capacity", expanded=False):
    sales_cycle = st.slider("Sales cycle length (months)", 0.5, 6.0, 2.0, step=0.5)
    new_accounts = st.slider("New accounts signed / month", 1, 20, 4, step=1)

with st.sidebar.expander("Retail economics", expanded=False):
    st.caption("Build-out & occupancy")
    build_out = st.slider("New cafe build-out cost ($)", 100_000, 500_000, 220_000, step=10_000, format="dollar")
    rent = st.slider("Monthly rent ($)", 3_000, 25_000, 11_000, step=500, format="dollar")
    st.caption("Revenue drivers")
    avg_ticket = st.slider("Average ticket ($)", 3.0, 12.0, 6.25, step=0.25, format="dollar")
    daily_tx = st.slider("Daily transactions", 50, 600, 220, step=10)
    st.caption("Margins & costs")
    cogs_pct = st.slider("COGS (%)", 15, 45, 28, step=1, format="%d%%") / 100
    labor_pct = st.slider("Labor (% of revenue)", 15, 45, 32, step=1, format="%d%%") / 100
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
        5, 60, int(region_defaults["qualified_pct"] * 100), step=1, format="%d%%",
    ) / 100

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
payback = cac_payback_months(cac, acv, gross_margin_ws)
ltc = ltv_to_cac(acv, gross_margin_ws, monthly_churn, cac)
ws_proj = wholesale_cohort_projection(acv, gross_margin_ws, monthly_churn, cac, new_accounts, months=MARKET_SIZING_MONTHS)
sensitivity = ltv_cac_sensitivity_to_churn(acv, gross_margin_ws, cac)
retail_unit = retail_unit_economics(build_out, rent, avg_ticket, daily_tx, cogs_pct, labor_pct)
retail_proj = retail_projection(build_out, rent, avg_ticket, daily_tx, cogs_pct, labor_pct, cafes_per_year, months=MARKET_SIZING_MONTHS)
mkt = market_sizing(addressable_accounts, qualified_pct, acv, new_accounts, months=MARKET_SIZING_MONTHS)

h = COMPARISON_HORIZON_MONTHS
wholesale_net_h = ws_proj[(ws_proj.scenario == "Base") & (ws_proj.month == h)]["cum_net_cash_position"].iloc[0]
retail_net_h = retail_proj[retail_proj.month == h]["cum_cash_position"].iloc[0]

summary = generate_strategic_summary(
    ltc, payback, retail_unit["payback_months"], wholesale_net_h, retail_net_h, sales_cycle
)
breakevens = compute_breakevens(
    acv, gross_margin_ws, monthly_churn, cac, new_accounts,
    build_out, rent, avg_ticket, daily_tx, cogs_pct, labor_pct, cafes_per_year,
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Wholesale CAC payback", f"{payback:.1f} mo" if payback != float("inf") else "never")
k2.metric("LTV : CAC", f"{ltc:.2f}x" if ltc != float("inf") else "∞")
k3.metric(
    "Retail build-out payback",
    f"{retail_unit['payback_months']:.1f} mo" if retail_unit["payback_months"] != float("inf") else "never",
)
k4.metric(f"Modeled {MARKET_SIZING_MONTHS}-mo SAM penetration", fmt_pct(mkt["penetration_of_sam"]))

st.divider()

# ---------------------------------------------------------------------------
# Strategic summary: metric -> interpretation -> decision
# ---------------------------------------------------------------------------
st.subheader("Strategic summary")

st.markdown(f"**Wholesale capital efficiency** — {summary['wholesale']['metric']}")
st.markdown(esc(summary["wholesale"]["interpretation"]))

st.markdown(f"**Retail economics** — {summary['retail']['metric']}")
st.markdown(esc(summary["retail"]["interpretation"]))

st.markdown(f"**Recommendation: {summary['recommendation']['choice']}**")
st.markdown(esc(summary["recommendation"]["reason"]))

st.divider()

# ---------------------------------------------------------------------------
# What would change the recommendation?
# ---------------------------------------------------------------------------
st.subheader("What would change the recommendation?")
st.caption(
    "The recommendation above isn't absolute — it holds under the current "
    "assumptions. Here's how far each key driver would have to move, on its "
    "own, to flip it."
)

FORMATTERS = {
    "Wholesale monthly churn": fmt_pct,
    "Wholesale CAC": fmt_money,
    "Cafe monthly revenue": fmt_money,
    "Cafe build-out cost": fmt_money,
}

rows = []
for entry in breakevens.values():
    fmt = FORMATTERS[entry["label"]]
    if entry["in_range"]:
        rows.append(
            {
                "Driver": entry["label"],
                "Current": fmt(entry["current"]),
                "Break-even": fmt(entry["breakeven"]),
                "What it means": entry["what_it_means"],
            }
        )
    else:
        rows.append(
            {
                "Driver": entry["label"],
                "Current": fmt(entry["current"]),
                "Break-even": "not within a realistic range",
                "What it means": "This driver alone doesn't flip the recommendation.",
            }
        )
st.table(pd.DataFrame(rows).set_index("Driver"))

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Wholesale growth", "Retail economics", "Market sizing"])

with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"**{MARKET_SIZING_MONTHS}-month MRR by scenario**")
        fig = go.Figure()
        ending_values = []
        for scenario, color in SCENARIO_COLORS.items():
            d = ws_proj[ws_proj.scenario == scenario]
            fig.add_trace(
                go.Scatter(
                    x=d["month"], y=d["mrr"], mode="lines", name=scenario,
                    line=dict(color=color, width=2),
                )
            )
            ending_values.append((scenario, d["mrr"].iloc[-1]))
        fig.update_layout(
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
            xaxis=dict(title="Month", gridcolor=GRID, showline=True, linecolor=MUTED),
            yaxis=dict(title="MRR ($)", gridcolor=GRID, tickformat="$,.0f"),
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=30, b=10, l=10, r=10),
            height=360,
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            esc(f"Month {MARKET_SIZING_MONTHS} MRR — " + "  ·  ".join(f"{s}: {fmt_money(v)}" for s, v in ending_values))
        )

    with right:
        st.markdown("**LTV:CAC sensitivity to churn**")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=sensitivity["monthly_churn"] * 100, y=sensitivity["ltv_cac"],
                mode="lines", line=dict(color=RUST, width=2),
            )
        )
        fig2.add_hline(y=1, line_dash="dot", line_color=MUTED)
        fig2.add_trace(
            go.Scatter(
                x=[monthly_churn * 100], y=[ltc], mode="markers",
                marker=dict(color=RUST, size=9), showlegend=False,
            )
        )
        fig2.update_layout(
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
            xaxis=dict(title="Monthly churn (%)", gridcolor=GRID),
            yaxis=dict(title="LTV : CAC", gridcolor=GRID),
            margin=dict(t=30, b=10, l=10, r=10),
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig2, width='stretch')
        st.caption("Dot marks the current churn assumption. Dotted line is LTV:CAC = 1.")

with tab2:
    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"**Cumulative net cash position** — opening {cafes_per_year:.1f} cafes/year")
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=retail_proj["month"], y=retail_proj["cum_cash_position"],
                mode="lines", line=dict(color=RUST, width=2),
            )
        )
        fig3.add_hline(y=0, line_dash="dot", line_color=MUTED)
        fig3.update_layout(
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
            xaxis=dict(title="Month", gridcolor=GRID),
            yaxis=dict(title="Cumulative net cash ($)", gridcolor=GRID, tickformat="$,.0f"),
            margin=dict(t=30, b=10, l=10, r=10),
            height=340,
        )
        st.plotly_chart(fig3, width='stretch')

    with right:
        st.markdown("**Where the numbers come from** (per store, per month)")
        st.caption(f"${avg_ticket:.2f} avg ticket × {daily_tx} transactions/day × 30 days")
        breakdown = pd.DataFrame(
            [
                {"Line item": "Revenue", "Monthly $": fmt_money(retail_unit["monthly_revenue"])},
                {"Line item": "− COGS", "Monthly $": fmt_money(-retail_unit["cogs_amount"])},
                {"Line item": "− Labor", "Monthly $": fmt_money(-retail_unit["labor_amount"])},
                {"Line item": "− Rent", "Monthly $": fmt_money(-retail_unit["rent"])},
                {"Line item": "= Gross profit", "Monthly $": fmt_money(retail_unit["monthly_gross_profit"])},
            ]
        ).set_index("Line item")
        st.table(breakdown)
        st.metric("Build-out payback", f"{retail_unit['payback_months']:.1f} mo" if retail_unit["payback_months"] != float("inf") else "never")

with tab3:
    st.markdown(f"**TAM → SAM → SOM** for {region}")
    st.caption("Nested market sets, not a conversion funnel — SOM is a subset of SAM, which is a subset of TAM.")
    labels = ["TAM — all addressable accounts", "SAM — specialty-qualified", f"SOM — modeled, {MARKET_SIZING_MONTHS}mo"]
    dollar_values = [mkt["tam_dollars"], mkt["sam_dollars"], mkt["som_dollars"]]
    account_values = [mkt["tam_accounts"], mkt["sam_accounts"], mkt["som_accounts"]]
    text_labels = [
        f"{acct:,.0f} accounts | ${dollars/1e6:.2f}M"
        for acct, dollars in zip(account_values, dollar_values)
    ]
    fig4 = go.Figure(
        go.Bar(
            y=labels, x=dollar_values, orientation="h",
            marker=dict(color=RUST_RAMP),
            text=text_labels, textposition="outside",
        )
    )
    fig4.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
        xaxis=dict(title="Dollar value ($)", gridcolor=GRID, tickformat="$.2s"),
        yaxis=dict(gridcolor=GRID),
        margin=dict(t=20, b=10, l=10, r=80),
        height=300,
    )
    st.plotly_chart(fig4, width='stretch')
    st.metric(f"Modeled {MARKET_SIZING_MONTHS}-month SAM penetration", fmt_pct(mkt["penetration_of_sam"]))
    st.caption("Sanity-checks the new-accounts assumption against how much of the addressable market it implies capturing.")

st.divider()
st.caption(
    "Built by Ashlyn Chi Garcia (Tepper MS Strategy, CMU). Model logic in "
    "model.py; interface in Streamlit and Plotly. All assumptions above "
    "are adjustable."
)