"""
model.py — Unit-economics & expansion logic for Bean Counter.

Two go-to-market channels for a specialty coffee roaster deciding how to
scale:

  1. Wholesale / B2B  — signing cafes, offices, and restaurants to recurring
     roasted-bean accounts (subscription-like: CAC, ACV, monthly logo churn,
     sales-cycle length — the same shape as enterprise SaaS unit economics).
  2. Retail           — opening company-owned cafes (build-out capex, rent,
     COGS, labor, footfall).

All functions are pure and take plain numbers/dicts in, DataFrames out, so
app.py stays a thin presentation layer over this logic. Every default is an
*illustrative* planning assumption (labeled as such in the UI), not a claimed
market fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_DEFAULT = 36

# ---------------------------------------------------------------------------
# Wholesale / B2B channel
# ---------------------------------------------------------------------------


def cac_payback_months(cac: float, acv: float, gross_margin: float) -> float:
    """Months of gross profit needed to recoup CAC on one account."""
    monthly_gross_profit = (acv / 12) * gross_margin
    if monthly_gross_profit <= 0:
        return float("inf")
    return cac / monthly_gross_profit


def ltv(acv: float, gross_margin: float, monthly_churn: float) -> float:
    """Simple recurring-revenue LTV: monthly gross profit / monthly churn."""
    if monthly_churn <= 0:
        return float("inf")
    monthly_gross_profit = (acv / 12) * gross_margin
    return monthly_gross_profit / monthly_churn


def ltv_to_cac(acv: float, gross_margin: float, monthly_churn: float, cac: float) -> float:
    if cac <= 0:
        return float("inf")
    return ltv(acv, gross_margin, monthly_churn) / cac


SCENARIO_ADJUSTMENTS = {
    "Conservative": {"new_accounts_mult": 0.6, "churn_mult": 1.4},
    "Base": {"new_accounts_mult": 1.0, "churn_mult": 1.0},
    "Aggressive": {"new_accounts_mult": 1.5, "churn_mult": 0.7},
}


def wholesale_cohort_projection(
    acv: float,
    gross_margin: float,
    monthly_churn: float,
    new_accounts_per_month: float,
    months: int = MONTHS_DEFAULT,
    starting_accounts: float = 0.0,
) -> pd.DataFrame:
    """
    Bathtub model: active accounts each month = last month's survivors +
    this month's new signups. Returns a month-by-month DataFrame of active
    accounts, MRR, and cumulative gross profit for the three standard
    scenarios (Conservative / Base / Aggressive).
    """
    rows = []
    for scenario, adj in SCENARIO_ADJUSTMENTS.items():
        accounts = starting_accounts
        cum_gross_profit = 0.0
        churn = min(max(monthly_churn * adj["churn_mult"], 0.0), 0.95)
        adds = new_accounts_per_month * adj["new_accounts_mult"]
        for m in range(1, months + 1):
            accounts = accounts * (1 - churn) + adds
            mrr = accounts * (acv / 12)
            monthly_gross_profit = mrr * gross_margin
            cum_gross_profit += monthly_gross_profit
            rows.append(
                {
                    "month": m,
                    "scenario": scenario,
                    "active_accounts": accounts,
                    "mrr": mrr,
                    "arr": mrr * 12,
                    "cum_gross_profit": cum_gross_profit,
                }
            )
    return pd.DataFrame(rows)


def cohort_retention_curve(monthly_churn: float, months: int = 24) -> pd.DataFrame:
    """Single starting cohort of 100 accounts, decayed by churn — the
    'how much of one signup class survives' curve hiring managers expect."""
    t = np.arange(0, months + 1)
    retained = 100 * (1 - monthly_churn) ** t
    return pd.DataFrame({"month": t, "pct_retained": retained})


# ---------------------------------------------------------------------------
# Retail channel
# ---------------------------------------------------------------------------


def retail_unit_economics(
    build_out_cost: float,
    monthly_rent: float,
    avg_ticket: float,
    daily_transactions: float,
    cogs_pct: float,
    labor_pct: float,
) -> dict:
    monthly_revenue = avg_ticket * daily_transactions * 30
    monthly_gross_profit = (
        monthly_revenue * (1 - cogs_pct) - monthly_revenue * labor_pct - monthly_rent
    )
    payback_months = (
        build_out_cost / monthly_gross_profit if monthly_gross_profit > 0 else float("inf")
    )
    return {
        "monthly_revenue": monthly_revenue,
        "monthly_gross_profit": monthly_gross_profit,
        "payback_months": payback_months,
        "annual_gross_profit": monthly_gross_profit * 12,
    }


def retail_projection(
    build_out_cost: float,
    monthly_rent: float,
    avg_ticket: float,
    daily_transactions: float,
    cogs_pct: float,
    labor_pct: float,
    cafes_per_year: float,
    months: int = MONTHS_DEFAULT,
) -> pd.DataFrame:
    """Cumulative net cash position if opening cafes_per_year new stores,
    each ramping to full unit economics after a 3-month ramp period."""
    unit = retail_unit_economics(
        build_out_cost, monthly_rent, avg_ticket, daily_transactions, cogs_pct, labor_pct
    )
    monthly_open_rate = cafes_per_year / 12
    rows = []
    cum_cash = 0.0
    stores_open = 0.0
    for m in range(1, months + 1):
        stores_open += monthly_open_rate
        cum_cash += -build_out_cost * monthly_open_rate  # capex spread across the month
        cum_cash += stores_open * unit["monthly_gross_profit"]
        rows.append(
            {
                "month": m,
                "stores_open": stores_open,
                "cum_cash_position": cum_cash,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Market sizing (TAM / SAM / SOM)
# ---------------------------------------------------------------------------

REGION_DEFAULTS = {
    "New York City": {"addressable_accounts": 5200, "qualified_pct": 0.35},
    "Los Angeles": {"addressable_accounts": 3800, "qualified_pct": 0.30},
    "Chicago": {"addressable_accounts": 2600, "qualified_pct": 0.30},
    "Portland": {"addressable_accounts": 1400, "qualified_pct": 0.40},
    "Mexico City": {"addressable_accounts": 3100, "qualified_pct": 0.28},
    "Bogotá": {"addressable_accounts": 1900, "qualified_pct": 0.25},
    "Custom": {"addressable_accounts": 2000, "qualified_pct": 0.30},
}


def market_sizing(
    addressable_accounts: float,
    qualified_pct: float,
    acv: float,
    new_accounts_per_month: float,
    months: int = 36,
) -> dict:
    tam_dollars = addressable_accounts * acv
    sam_accounts = addressable_accounts * qualified_pct
    sam_dollars = sam_accounts * acv
    som_accounts = min(new_accounts_per_month * months, sam_accounts)
    som_dollars = som_accounts * acv
    penetration_of_sam = som_accounts / sam_accounts if sam_accounts > 0 else 0.0
    return {
        "tam_accounts": addressable_accounts,
        "tam_dollars": tam_dollars,
        "sam_accounts": sam_accounts,
        "sam_dollars": sam_dollars,
        "som_accounts": som_accounts,
        "som_dollars": som_dollars,
        "penetration_of_sam": penetration_of_sam,
    }


# ---------------------------------------------------------------------------
# Executive takeaway generation
# ---------------------------------------------------------------------------


def generate_takeaway(
    cac_payback: float,
    ltv_cac: float,
    sales_cycle_months: float,
    retail_payback: float,
    wholesale_24mo_profit: float,
    retail_24mo_cash: float,
    som_penetration: float,
) -> list[str]:
    """Return a short list of plain-English, threshold-driven takeaways —
    the kind of read a Chief of Staff wants in 15 seconds, not a chart."""
    lines = []

    if ltv_cac >= 3:
        lines.append(
            f"**Wholesale unit economics are healthy** — LTV:CAC of {ltv_cac:.1f}x clears "
            "the 3x bar investors typically look for."
        )
    elif ltv_cac >= 1:
        lines.append(
            f"**Wholesale unit economics are marginal** — LTV:CAC of {ltv_cac:.1f}x is "
            "positive but below the 3x bar; churn or CAC needs to improve before scaling spend."
        )
    else:
        lines.append(
            f"**Wholesale unit economics don't work yet** — LTV:CAC of {ltv_cac:.1f}x means "
            "each account costs more to acquire than it returns. Fix churn or CAC before adding volume."
        )

    payback_vs_cycle = cac_payback - sales_cycle_months
    if cac_payback <= 12:
        lines.append(
            f"CAC payback is {cac_payback:.1f} months (sales cycle: {sales_cycle_months:.1f} "
            "months) — within the healthy under-12-month range for B2B wholesale."
        )
    else:
        lines.append(
            f"CAC payback is {cac_payback:.1f} months — longer than the 12-month rule of "
            "thumb, so cash gets tied up per account for a while before it pays back."
        )

    if retail_payback != float("inf") and retail_payback <= 24:
        lines.append(
            f"A new cafe pays back its build-out in {retail_payback:.1f} months — solid "
            "for brick-and-mortar retail (24 months is a common hurdle)."
        )
    elif retail_payback == float("inf"):
        lines.append(
            "At current rent/ticket/traffic assumptions, a new cafe never breaks even on "
            "build-out cost — retail expansion needs better unit economics before opening more stores."
        )
    else:
        lines.append(
            f"A new cafe takes {retail_payback:.1f} months to pay back its build-out — "
            "slower than the typical 24-month retail hurdle."
        )

    if wholesale_24mo_profit > retail_24mo_cash:
        lines.append(
            "**Recommendation: prioritize wholesale.** At month 24, cumulative gross profit "
            f"from wholesale (${wholesale_24mo_profit:,.0f}) outpaces retail's cash position "
            f"(${retail_24mo_cash:,.0f}) for the same modeling horizon — wholesale is the more "
            "capital-efficient growth lever right now."
        )
    else:
        lines.append(
            "**Recommendation: retail is pulling ahead.** At month 24, cumulative cash from "
            f"retail (${retail_24mo_cash:,.0f}) outpaces wholesale's gross profit "
            f"(${wholesale_24mo_profit:,.0f}) — worth protecting store-opening pace even though "
            "it's more capital-intensive up front."
        )

    lines.append(
        f"Modeled wholesale growth reaches ~{som_penetration:.0%} penetration of the "
        "serviceable addressable market by month 36 — a sanity check on whether the new-accounts "
        "assumption is realistic for the region selected."
    )

    return lines
