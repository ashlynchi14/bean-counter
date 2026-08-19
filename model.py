"""
model.py — Unit-economics, break-even, and strategic-summary logic for Bean
Counter.

The strategic question this model exists to answer: a specialty coffee
company wants to grow. Should it deploy capital toward acquiring wholesale
accounts (cafes, offices, restaurants — subscription-like economics: CAC,
ACV, monthly logo churn), or toward opening its own retail cafes (build-out
capex, rent, COGS, labor, footfall)?

Everything in this file is pure and UI-free — app.py is a thin presentation
layer over it. Every default is an *illustrative* planning assumption
(labeled as such in the UI), not a claimed market fact.

Design note on the wholesale/retail comparison: both channels' cumulative
figures are tracked NET of the capital spent to get there — wholesale nets
out CAC spend on new signups each month, retail nets out build-out capex as
stores open — so the "which channel wins" comparison is apples-to-apples.
Earlier versions of this model tracked wholesale gross profit without
subtracting CAC, which overstated wholesale's advantage; that's fixed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_DEFAULT = 36
COMPARISON_HORIZON_MONTHS = 24  # the month at which channels are compared
LTV_HORIZON_MONTHS = 36  # finite window for the headline LTV:CAC — see ltv_finite_horizon

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


def ltv_finite_horizon(
    acv: float, gross_margin: float, monthly_churn: float, months: int = LTV_HORIZON_MONTHS
) -> float:
    """LTV over a fixed window instead of assuming indefinite retention.

    `ltv()` above is a perpetuity (monthly gross profit / churn) — at low
    churn that produces a near-infinite "lifetime" that overstates what a
    real cohort actually returns. This sums each month's expected gross
    profit, discounted only by the probability the account is still active
    that month (a straightforward geometric series), over a bounded
    horizon — a more defensible number to lead with.
    """
    monthly_gross_profit = (acv / 12) * gross_margin
    if monthly_churn <= 0:
        return monthly_gross_profit * months
    retention_sum = (1 - (1 - monthly_churn) ** months) / monthly_churn
    return monthly_gross_profit * retention_sum


def ltv_to_cac_finite(
    acv: float,
    gross_margin: float,
    monthly_churn: float,
    cac: float,
    months: int = LTV_HORIZON_MONTHS,
) -> float:
    if cac <= 0:
        return float("inf")
    return ltv_finite_horizon(acv, gross_margin, monthly_churn, months) / cac


SCENARIO_ADJUSTMENTS = {
    "Conservative": {"new_accounts_mult": 0.6, "churn_mult": 1.4},
    "Base": {"new_accounts_mult": 1.0, "churn_mult": 1.0},
    "Aggressive": {"new_accounts_mult": 1.5, "churn_mult": 0.7},
}


def wholesale_cohort_projection(
    acv: float,
    gross_margin: float,
    monthly_churn: float,
    cac: float,
    new_accounts_per_month: float,
    months: int = MONTHS_DEFAULT,
    starting_accounts: float = 0.0,
) -> pd.DataFrame:
    """
    Bathtub model: active accounts each month = last month's survivors +
    this month's new signups, each new signup costing CAC up front. Returns
    a month-by-month DataFrame with active accounts, MRR, cumulative gross
    profit, and cumulative *net* cash position (gross profit minus CAC
    spend) for the three standard scenarios.
    """
    rows = []
    for scenario, adj in SCENARIO_ADJUSTMENTS.items():
        accounts = starting_accounts
        cum_gross_profit = 0.0
        cum_net_cash = 0.0
        churn = min(max(monthly_churn * adj["churn_mult"], 0.0), 0.95)
        adds = new_accounts_per_month * adj["new_accounts_mult"]
        for m in range(1, months + 1):
            accounts = accounts * (1 - churn) + adds
            mrr = accounts * (acv / 12)
            monthly_gross_profit = mrr * gross_margin
            monthly_cac_spend = adds * cac
            cum_gross_profit += monthly_gross_profit
            cum_net_cash += monthly_gross_profit - monthly_cac_spend
            rows.append(
                {
                    "month": m,
                    "scenario": scenario,
                    "active_accounts": accounts,
                    "mrr": mrr,
                    "arr": mrr * 12,
                    "cum_gross_profit": cum_gross_profit,
                    "cum_net_cash_position": cum_net_cash,
                }
            )
    return pd.DataFrame(rows)


def ltv_cac_sensitivity_to_churn(
    acv: float,
    gross_margin: float,
    cac: float,
    churn_min: float = 0.005,
    churn_max: float = 0.08,
    n: int = 24,
    months: int = LTV_HORIZON_MONTHS,
) -> pd.DataFrame:
    """Finite-horizon LTV:CAC across a range of monthly churn, holding
    ACV/margin/CAC fixed — the decision-relevant sensitivity: how much air
    is in the number the user is already looking at, not just a picture of
    the churn input. Uses the same finite-horizon definition as the
    headline metric so the chart and the number it's next to never
    disagree."""
    churns = np.linspace(churn_min, churn_max, n)
    ratios = [ltv_to_cac_finite(acv, gross_margin, c, cac, months) for c in churns]
    return pd.DataFrame({"monthly_churn": churns, "ltv_cac": ratios})


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
    """Revenue built from operating drivers (ticket x transactions x days)
    and gross profit broken into its named cost lines, so the numbers are
    traceable rather than opaque."""
    monthly_revenue = avg_ticket * daily_transactions * 30
    cogs_amount = monthly_revenue * cogs_pct
    labor_amount = monthly_revenue * labor_pct
    monthly_gross_profit = monthly_revenue - cogs_amount - labor_amount - monthly_rent
    payback_months = (
        build_out_cost / monthly_gross_profit if monthly_gross_profit > 0 else float("inf")
    )
    return {
        "monthly_revenue": monthly_revenue,
        "cogs_amount": cogs_amount,
        "labor_amount": labor_amount,
        "rent": monthly_rent,
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
    each ramping to full unit economics immediately (a simplification —
    real stores ramp over a few months, but that adds a parameter for
    limited decision value here)."""
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
# Strategic summary: metric -> interpretation -> decision
# ---------------------------------------------------------------------------


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def generate_strategic_summary(
    ltv_cac: float,
    cac_payback: float,
    retail_payback_months: float,
    wholesale_net_24: float,
    retail_net_24: float,
    sales_cycle_months: float = 0.0,
    ltv_horizon_months: int = LTV_HORIZON_MONTHS,
) -> dict:
    """Three short, structured conclusions — not a memo. Benchmarks are
    described as general planning heuristics, not asserted as universal
    rules, since they aren't sourced to any specific investor or market.
    `ltv_cac` is expected to be the finite-horizon ratio (see
    ltv_to_cac_finite) — the headline number this app leads with — not the
    steady-state perpetuity value."""

    payback_str = f"{cac_payback:.1f} months" if cac_payback != float("inf") else "never"
    if ltv_cac >= 3:
        wholesale_interp = "Indicates strong modeled unit economics under the current assumptions."
    elif ltv_cac >= 1:
        wholesale_interp = (
            "Positive but thin under the current assumptions — a modest rise in churn or "
            "CAC would erode this margin."
        )
    else:
        wholesale_interp = (
            "Under the current assumptions, each account costs more to acquire than it "
            f"returns within the {ltv_horizon_months}-month window modeled."
        )
    if cac_payback != float("inf") and sales_cycle_months > 0:
        total_cycle = cac_payback + sales_cycle_months
        wholesale_interp += (
            f" Add the {sales_cycle_months:.1f}-month sales cycle, and cash isn't fully "
            f"recovered until about {total_cycle:.1f} months after outreach starts."
        )
    wholesale = {
        "metric": f"{ltv_horizon_months}-mo LTV:CAC {ltv_cac:.2f}x  ·  CAC payback {payback_str}",
        "interpretation": wholesale_interp,
    }

    if retail_payback_months == float("inf"):
        retail_interp = (
            "Under the current rent, ticket size, and traffic assumptions, a new cafe does "
            "not earn back its build-out cost within the modeled horizon."
        )
    elif retail_payback_months <= 24:
        retail_interp = (
            "Under the current assumptions, payback lands within a typical multi-year "
            "retail investment window."
        )
    else:
        retail_interp = (
            "Under the current assumptions, payback runs longer than a typical multi-year "
            "retail investment window."
        )
    retail_payback_str = (
        f"{retail_payback_months:.1f} months" if retail_payback_months != float("inf") else "beyond the modeled horizon"
    )
    retail = {
        "metric": f"Build-out payback: {retail_payback_str}",
        "interpretation": retail_interp,
    }

    wholesale_viable = ltv_cac >= 1
    retail_viable = retail_payback_months != float("inf")

    if not wholesale_viable and not retail_viable:
        recommendation = {
            "choice": "Neither",
            "reason": (
                "Under the current assumptions, neither channel's unit economics work as "
                "modeled — fix wholesale CAC/churn or retail rent/traffic assumptions before "
                "committing further capital to either."
            ),
        }
    elif wholesale_viable and (not retail_viable or wholesale_net_24 >= retail_net_24):
        recommendation = {
            "choice": "Wholesale",
            "reason": (
                f"Under the current assumptions, wholesale is the more capital-efficient "
                f"growth path: at month {COMPARISON_HORIZON_MONTHS}, its net cash position "
                f"({_fmt_money(wholesale_net_24)}) is ahead of retail's ({_fmt_money(retail_net_24)})."
            ),
        }
    else:
        recommendation = {
            "choice": "Retail",
            "reason": (
                f"Under the current assumptions, retail is the more capital-efficient growth "
                f"path: at month {COMPARISON_HORIZON_MONTHS}, its net cash position "
                f"({_fmt_money(retail_net_24)}) is ahead of wholesale's ({_fmt_money(wholesale_net_24)})."
            ),
        }

    return {"wholesale": wholesale, "retail": retail, "recommendation": recommendation}


# ---------------------------------------------------------------------------
# "What would change the recommendation?" — break-even thresholds
# ---------------------------------------------------------------------------


def _linear_crossover(f, x_low: float, x_high: float, target: float = 0.0):
    """Two-point solve for the x where f(x) == target. Exact when f is
    linear in x (true for CAC, build-out cost, and revenue drivers in this
    model over a fixed horizon — each only scales one additive term), and a
    reasonable first-order estimate otherwise. Returns None if f doesn't
    depend on x at all (the two points come out equal)."""
    y_low, y_high = f(x_low), f(x_high)
    if y_high == y_low:
        return None
    return x_low + (target - y_low) * (x_high - x_low) / (y_high - y_low)


def _bisection_root(f, lo: float, hi: float, iterations: int = 60):
    """Root-find for f(x) == 0 where f is monotonic but not necessarily
    linear (used for the churn break-even, since finite-horizon LTV is a
    geometric series in churn, not a straight line). Returns None if f
    doesn't change sign across [lo, hi] — no root in range."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None
    for _ in range(iterations):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if f_mid == 0:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def _wholesale_net_cash_at(acv, gross_margin, monthly_churn, cac, new_accounts_per_month, month):
    df = wholesale_cohort_projection(acv, gross_margin, monthly_churn, cac, new_accounts_per_month, months=month)
    row = df[(df.scenario == "Base") & (df.month == month)]
    return float(row["cum_net_cash_position"].iloc[0])


def _retail_net_cash_at(build_out_cost, monthly_rent, avg_ticket, daily_transactions, cogs_pct, labor_pct, cafes_per_year, month):
    df = retail_projection(build_out_cost, monthly_rent, avg_ticket, daily_transactions, cogs_pct, labor_pct, cafes_per_year, months=month)
    row = df[df.month == month]
    return float(row["cum_cash_position"].iloc[0])


def compute_breakevens(
    acv: float,
    gross_margin_ws: float,
    monthly_churn: float,
    cac: float,
    new_accounts_per_month: float,
    build_out_cost: float,
    monthly_rent: float,
    avg_ticket: float,
    daily_transactions: float,
    cogs_pct: float,
    labor_pct: float,
    cafes_per_year: float,
    horizon_months: int = COMPARISON_HORIZON_MONTHS,
) -> dict:
    """For each key driver, find the value at which the strategic
    recommendation flips, holding every other input at its current slider
    value. Each entry reports current value, break-even value, direction
    of the change needed, and whether that break-even falls within a
    realistic range (flagged rather than shown as a real number if not)."""

    wholesale_net = _wholesale_net_cash_at(acv, gross_margin_ws, monthly_churn, cac, new_accounts_per_month, horizon_months)
    retail_net = _retail_net_cash_at(build_out_cost, monthly_rent, avg_ticket, daily_transactions, cogs_pct, labor_pct, cafes_per_year, horizon_months)

    results = {}

    # 1. Wholesale churn — solved numerically against the same finite-horizon
    #    LTV definition the headline metric uses (LTV is a geometric series
    #    in churn, not linear, so this isn't a closed-form solve).
    churn_breakeven = _bisection_root(
        lambda c: ltv_finite_horizon(acv, gross_margin_ws, c, LTV_HORIZON_MONTHS) - cac,
        1e-6, 0.99,
    ) if cac > 0 else None
    results["churn"] = _threshold_entry(
        label="Wholesale monthly churn",
        current=monthly_churn,
        breakeven=churn_breakeven,
        valid_range=(0.0, 1.0),
        what_it_means=(
            f"Above this, wholesale accounts stop paying for themselves within the "
            f"{LTV_HORIZON_MONTHS}-month window (LTV:CAC < 1)."
        ),
    )

    # 2. Wholesale CAC vs. retail — exactly linear in CAC (new-signup
    #    schedule doesn't depend on CAC), so the two-point solve is exact.
    cac_hi = max(cac * 4, cac + 2000)
    cac_breakeven = _linear_crossover(
        lambda c: _wholesale_net_cash_at(acv, gross_margin_ws, monthly_churn, c, new_accounts_per_month, horizon_months) - retail_net,
        0, cac_hi,
    )
    results["cac"] = _threshold_entry(
        label="Wholesale CAC",
        current=cac,
        breakeven=cac_breakeven,
        valid_range=(0.0, cac * 10),
        what_it_means="Beyond this, retail overtakes wholesale as the better use of capital.",
    )

    # 3. Cafe daily transactions vs. wholesale — linear in transactions
    #    (revenue and gross profit both scale linearly with it).
    tx_hi = max(daily_transactions * 4, daily_transactions + 200)
    tx_breakeven = _linear_crossover(
        lambda tx: _retail_net_cash_at(build_out_cost, monthly_rent, avg_ticket, tx, cogs_pct, labor_pct, cafes_per_year, horizon_months) - wholesale_net,
        0, tx_hi,
    )
    revenue_breakeven = avg_ticket * tx_breakeven * 30 if tx_breakeven is not None else None
    results["cafe_revenue"] = _threshold_entry(
        label="Cafe monthly revenue",
        current=avg_ticket * daily_transactions * 30,
        breakeven=revenue_breakeven,
        valid_range=(0.0, avg_ticket * daily_transactions * 30 * 10),
        what_it_means="At this level, retail catches up to wholesale's capital efficiency.",
    )

    # 4. Cafe build-out cost vs. wholesale — linear in build-out cost
    #    (capex is the only term build-out cost enters).
    build_hi = max(build_out_cost * 4, build_out_cost + 200_000)
    build_breakeven = _linear_crossover(
        lambda b: _retail_net_cash_at(b, monthly_rent, avg_ticket, daily_transactions, cogs_pct, labor_pct, cafes_per_year, horizon_months) - wholesale_net,
        0, build_hi,
    )
    results["build_out"] = _threshold_entry(
        label="Cafe build-out cost",
        current=build_out_cost,
        breakeven=build_breakeven,
        valid_range=(0.0, build_out_cost * 10),
        what_it_means="Below this, a cheaper build-out makes retail competitive with wholesale.",
    )

    return results


def _threshold_entry(label, current, breakeven, valid_range, what_it_means) -> dict:
    lo, hi = valid_range
    in_range = breakeven is not None and lo < breakeven <= hi
    direction = None
    if in_range:
        direction = "up" if breakeven > current else "down"
    return {
        "label": label,
        "current": current,
        "breakeven": breakeven if in_range else None,
        "in_range": in_range,
        "direction": direction,
        "what_it_means": what_it_means,
    }


def classify_breakeven_robustness(entry: dict) -> str:
    """Plain-English read on one break-even row: is this a fragile part of
    the recommendation, or a driver that would have to move an unrealistic
    amount to matter?"""
    if not entry["in_range"]:
        return "Robust — no realistic value of this driver alone flips the recommendation."
    current, breakeven = entry["current"], entry["breakeven"]
    rel_change = abs(breakeven - current) / abs(current) if current else float("inf")
    if rel_change >= 1.0:
        return "Would need to roughly double (or more) — a large swing, but not impossible."
    elif rel_change >= 0.25:
        return "A meaningful but plausible swing — worth keeping an eye on."
    else:
        return "Close to the current assumption — this is a fragile point in the recommendation."


def breakeven_robustness_summary(breakevens: dict) -> str:
    """One dynamically-generated sentence summarizing how robust the overall
    recommendation is across every driver tested, and naming the most
    fragile one if there is one."""
    entries = list(breakevens.values())
    fragile = [e for e in entries if e["in_range"]]
    if not fragile:
        return (
            "The recommendation is robust to every driver tested individually — no single "
            "input's realistic range flips the call on its own."
        )

    def _rel_dist(e):
        return abs(e["breakeven"] - e["current"]) / abs(e["current"]) if e["current"] else float("inf")

    most_fragile = min(fragile, key=_rel_dist)
    robust_count = len(entries) - len(fragile)
    return (
        f"{robust_count} of {len(entries)} drivers tested would need an unrealistic swing to "
        f"flip the call on their own. The most sensitive is {most_fragile['label'].lower()} — "
        f"worth validating first if you're stress-testing this recommendation."
    )


# ---------------------------------------------------------------------------
# Market-sizing sanity check
# ---------------------------------------------------------------------------


def interpret_sam_penetration(penetration_of_sam: float) -> str:
    """Soft-worded read on how much of the qualified market the modeled
    new-account pace implies capturing. These bands are rough planning
    heuristics for sanity-checking an assumption against itself, not a
    sourced industry benchmark."""
    if penetration_of_sam < 0.10:
        return (
            "Conservative relative to the qualified market — plausible at the modeled "
            "acquisition pace."
        )
    elif penetration_of_sam < 0.20:
        return (
            "Plausible but meaningful — achieving this would require sustained execution at "
            "the modeled new-account pace."
        )
    elif penetration_of_sam < 0.30:
        return (
            "Ambitious — capturing this much of the qualified market in the window modeled "
            "would likely need more acquisition capacity than the current new-accounts "
            "assumption implies."
        )
    else:
        return (
            "Aggressive — this level of penetration this quickly is uncommon and worth "
            "stress-testing directly against the new-accounts-per-month assumption."
        )


# ---------------------------------------------------------------------------
# Assumption presets — Conservative / Base case / Aggressive starting points
# for the wholesale sliders, plus one fixed set of retail/market defaults.
# None of these are sourced market data; they're illustrative planning
# assumptions, spread out so the model isn't quietly defaulted to whichever
# numbers make wholesale look best.
# ---------------------------------------------------------------------------

PRESETS = {
    "Conservative": {
        "acv": 6000, "cac": 2200, "gross_margin_ws": 52,
        "monthly_churn": 4.0, "new_accounts": 3, "sales_cycle": 3.5,
    },
    "Base case": {
        "acv": 9000, "cac": 1500, "gross_margin_ws": 58,
        "monthly_churn": 2.5, "new_accounts": 5, "sales_cycle": 2.5,
    },
    "Aggressive": {
        "acv": 13000, "cac": 1000, "gross_margin_ws": 63,
        "monthly_churn": 1.2, "new_accounts": 9, "sales_cycle": 1.5,
    },
}

RETAIL_MARKET_DEFAULTS = {
    "build_out": 220_000,
    "rent": 11_000,
    "avg_ticket": 6.25,
    "daily_tx": 220,
    "cogs_pct": 28,
    "labor_pct": 32,
    "cafes_per_year": 1.0,
    "region_select": "New York City",
    "addressable_accounts": REGION_DEFAULTS["New York City"]["addressable_accounts"],
    "qualified_pct": int(REGION_DEFAULTS["New York City"]["qualified_pct"] * 100),
}