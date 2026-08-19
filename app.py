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
    LTV_HORIZON_MONTHS,
    PRESETS,
    REGION_DEFAULTS,
    RETAIL_MARKET_DEFAULTS,
    breakeven_robustness_summary,
    cac_payback_months,
    classify_breakeven_robustness,
    compute_breakevens,
    generate_strategic_summary,
    interpret_sam_penetration,
    ltv_cac_sensitivity_to_churn,
    ltv_to_cac,
    ltv_to_cac_finite,
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

with st.expander("How the model works", expanded=False):
    st.markdown(
        f"""
**Wholesale LTV.** Each account generates monthly gross profit (ACV ÷ 12 ×
gross margin). The headline LTV:CAC uses a **{LTV_HORIZON_MONTHS}-month
window** — it sums the gross profit an average account is expected to
generate, discounted only by the chance it's still active each month, and
divides by CAC. A separate steady-state figure (shown alongside it) assumes
an account stays forever, which is the textbook formula but can look
artificially large at low churn — the finite-horizon number is the more
defensible one to lead with.

**CAC payback.** How many months of one account's gross profit it takes to
recoup its acquisition cost — independent of the LTV horizon.

**Wholesale cash position.** New accounts are added each month and existing
ones churn off (a "bathtub" model: this month's total = last month's
survivors + new signups). Cumulative cash position is gross profit collected
so far, *minus* CAC spent acquiring every signup — so it's a true net
position, not just revenue.

**Retail payback.** A store's monthly gross profit (revenue minus COGS,
labor, and rent) divided into its build-out cost. The cash-position chart
opens new stores at the modeled pace and nets build-out capex against
cumulative gross profit across all open stores.

**TAM / SAM / SOM.** TAM is every addressable account in the region. SAM is
the share judged specialty-relevant. SOM is how many accounts the modeled
new-accounts pace could realistically sign within the chart's window,
capped at SAM.

**Comparison horizon.** The strategic summary and break-even table compare
wholesale vs. retail net cash position at month {COMPARISON_HORIZON_MONTHS}
— a fixed point far enough out to separate the two channels, but soon
enough to still be a near-term capital decision. Charts extend further (to
{MARKET_SIZING_MONTHS} months) to show the fuller trajectory.
        """
    )

# ---------------------------------------------------------------------------
# Sidebar — inputs, grouped by decision area
# ---------------------------------------------------------------------------
st.sidebar.header("Assumptions")

_ALL_DEFAULTS = {**PRESETS["Base case"], **RETAIL_MARKET_DEFAULTS}
for _key, _val in _ALL_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val
if "preset_choice" not in st.session_state:
    st.session_state["preset_choice"] = "Base case"


def _apply_preset():
    for k, v in PRESETS[st.session_state["preset_choice"]].items():
        st.session_state[k] = v


def _reset_to_base_case():
    for k, v in _ALL_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["preset_choice"] = "Base case"


def _on_region_change():
    region_defaults = REGION_DEFAULTS[st.session_state["region_select"]]
    st.session_state["addressable_accounts"] = region_defaults["addressable_accounts"]
    st.session_state["qualified_pct"] = int(region_defaults["qualified_pct"] * 100)


st.sidebar.selectbox(
    "Wholesale assumption preset",
    list(PRESETS.keys()),
    key="preset_choice",
    on_change=_apply_preset,
    help="Loads a defensible starting set of wholesale sliders below. Not fabricated data — an illustrative planning scenario, adjustable after loading.",
)
st.sidebar.button("↺ Reset to Base Case", on_click=_reset_to_base_case, width='stretch')

with st.sidebar.expander("Wholesale economics", expanded=True):
    acv = st.slider("Average Contract Value — ACV ($/yr)", 2000, 30000, step=500, format="dollar", key="acv")
    cac = st.slider("Customer Acquisition Cost — CAC ($)", 300, 5000, step=100, format="dollar", key="cac")
    gross_margin_ws = st.slider("Wholesale gross margin (%)", 30, 75, step=1, format="%d%%", key="gross_margin_ws") / 100
    monthly_churn = st.slider("Monthly logo churn (%)", 0.5, 8.0, step=0.1, format="%.1f%%", key="monthly_churn") / 100

with st.sidebar.expander("Sales capacity", expanded=False):
    sales_cycle = st.slider("Sales cycle length (months)", 0.5, 6.0, step=0.5, key="sales_cycle")
    new_accounts = st.slider("New accounts signed / month", 1, 20, step=1, key="new_accounts")

with st.sidebar.expander("Retail economics", expanded=False):
    st.caption("Build-out & occupancy")
    build_out = st.slider("New cafe build-out cost ($)", 100_000, 500_000, step=10_000, format="dollar", key="build_out")
    rent = st.slider("Monthly rent ($)", 3_000, 25_000, step=500, format="dollar", key="rent")
    st.caption("Revenue drivers")
    avg_ticket = st.slider("Average ticket ($)", 3.0, 12.0, step=0.25, format="dollar", key="avg_ticket")
    daily_tx = st.slider("Daily transactions", 50, 600, step=10, key="daily_tx")
    st.caption("Margins & costs")
    cogs_pct = st.slider("COGS (%)", 15, 45, step=1, format="%d%%", key="cogs_pct") / 100
    labor_pct = st.slider("Labor (% of revenue)", 15, 45, step=1, format="%d%%", key="labor_pct") / 100
    cafes_per_year = st.slider("New cafes opened / year", 0.0, 6.0, step=0.5, key="cafes_per_year")

with st.sidebar.expander("Market sizing", expanded=False):
    region = st.selectbox(
        "Target region", list(REGION_DEFAULTS.keys()), key="region_select", on_change=_on_region_change
    )
    addressable_accounts = st.number_input(
        "Addressable business accounts in region (cafes, offices, restaurants)",
        min_value=200,
        max_value=20000,
        step=100,
        key="addressable_accounts",
    )
    qualified_pct = st.slider(
        "% qualified as specialty-relevant (SAM)",
        5, 60, step=1, format="%d%%", key="qualified_pct",
    ) / 100

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
payback = cac_payback_months(cac, acv, gross_margin_ws)
ltc = ltv_to_cac_finite(acv, gross_margin_ws, monthly_churn, cac, months=LTV_HORIZON_MONTHS)
ltc_steady_state = ltv_to_cac(acv, gross_margin_ws, monthly_churn, cac)
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
k2.metric(f"{LTV_HORIZON_MONTHS}-mo LTV : CAC", f"{ltc:.2f}x" if ltc != float("inf") else "∞")
k2.caption(
    f"Steady-state (assumes indefinite retention): {ltc_steady_state:.2f}x"
    if ltc_steady_state != float("inf") else "Steady-state: ∞ (zero churn assumed)"
)
k3.metric(
    "Retail build-out payback",
    f"{retail_unit['payback_months']:.1f} mo" if retail_unit["payback_months"] != float("inf") else "never",
)
k4.metric(f"Modeled {MARKET_SIZING_MONTHS}-mo SAM penetration", fmt_pct(mkt["penetration_of_sam"]))
k4.caption(interpret_sam_penetration(mkt["penetration_of_sam"]))

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
                "How robust is this?": classify_breakeven_robustness(entry),
            }
        )
    else:
        rows.append(
            {
                "Driver": entry["label"],
                "Current": fmt(entry["current"]),
                "Break-even": "not within a realistic range",
                "What it means": "This driver alone doesn't flip the recommendation.",
                "How robust is this?": classify_breakeven_robustness(entry),
            }
        )
st.table(pd.DataFrame(rows).set_index("Driver"))
st.caption(esc(breakeven_robustness_summary(breakevens)))

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
        st.markdown(f"**{LTV_HORIZON_MONTHS}-mo LTV:CAC sensitivity to churn**")
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
    st.caption(interpret_sam_penetration(mkt["penetration_of_sam"]))
    st.caption(
        "Rough planning bands, not a sourced benchmark: <10% conservative · 10–20% plausible "
        "· 20–30% ambitious · >30% aggressive. Also worth checking against the acquisition "
        "capacity implied by the new-accounts-per-month assumption in the sidebar."
    )

st.divider()
st.caption(
    "Built by Ashlyn Chi Garcia (Tepper MS Strategy, CMU). Model logic in "
    "model.py; interface in Streamlit and Plotly. All assumptions above "
    "are adjustable."
)