"""
tests/test_model.py — pytest suite for model.py's core unit-economics logic.

Run with:  pytest
(install test-only deps first: pip install -r requirements-dev.txt)

Note: these tests target the actual function signatures in model.py
(cac_payback_months, ltv_to_cac, wholesale_cohort_projection, etc.) —
not a hypothetical calculate_financial_projections() wrapper.
"""

import math

import pytest

from model import (
    cac_payback_months,
    cohort_retention_curve,
    generate_takeaway,
    ltv,
    ltv_to_cac,
    market_sizing,
    retail_projection,
    retail_unit_economics,
    wholesale_cohort_projection,
)


# ---------------------------------------------------------------------------
# CAC payback
# ---------------------------------------------------------------------------


def test_payback_decreases_as_acv_increases():
    """Higher ACV relative to a fixed CAC should pay back faster, not slower."""
    high_acv_payback = cac_payback_months(cac=3000, acv=24000, gross_margin=0.58)
    low_acv_payback = cac_payback_months(cac=3000, acv=6000, gross_margin=0.58)
    assert high_acv_payback < low_acv_payback


def test_payback_is_infinite_at_zero_margin():
    """Zero gross margin means CAC is never recouped — should be inf, not a
    divide-by-zero crash or a silently wrong finite number."""
    assert cac_payback_months(cac=1200, acv=8000, gross_margin=0.0) == float("inf")


def test_payback_scales_linearly_with_cac():
    base = cac_payback_months(cac=1000, acv=8000, gross_margin=0.5)
    doubled = cac_payback_months(cac=2000, acv=8000, gross_margin=0.5)
    assert doubled == pytest.approx(base * 2)


# ---------------------------------------------------------------------------
# LTV / LTV:CAC
# ---------------------------------------------------------------------------


def test_ltv_is_infinite_at_zero_churn():
    """With 0% churn accounts never leave, so LTV (undiscounted) is unbounded —
    matches the model's approach of returning inf rather than a wrong number
    from dividing by a zero denominator."""
    assert ltv(acv=12000, gross_margin=0.6, monthly_churn=0.0) == float("inf")


def test_ltv_to_cac_matches_manual_calculation():
    acv, margin, churn, cac = 8000, 0.58, 0.025, 1200
    expected_ltv = (acv / 12 * margin) / churn
    expected_ratio = expected_ltv / cac
    assert ltv_to_cac(acv, margin, churn, cac) == pytest.approx(expected_ratio)


def test_ltv_to_cac_improves_as_churn_drops():
    high_churn_ratio = ltv_to_cac(acv=8000, gross_margin=0.58, monthly_churn=0.06, cac=1200)
    low_churn_ratio = ltv_to_cac(acv=8000, gross_margin=0.58, monthly_churn=0.01, cac=1200)
    assert low_churn_ratio > high_churn_ratio


# ---------------------------------------------------------------------------
# Wholesale cohort / scenario projection
# ---------------------------------------------------------------------------


def test_wholesale_projection_zero_churn_never_loses_accounts():
    """At 0% churn (Base scenario churn_mult applies to 0, so still 0), active
    accounts should be non-decreasing month over month since nobody leaves."""
    df = wholesale_cohort_projection(
        acv=12000, gross_margin=0.5, monthly_churn=0.0, new_accounts_per_month=3, months=24
    )
    base = df[df.scenario == "Base"].sort_values("month")
    diffs = base["active_accounts"].diff().dropna()
    assert (diffs >= -1e-9).all()  # never decreases


def test_wholesale_projection_returns_all_three_scenarios():
    df = wholesale_cohort_projection(
        acv=8000, gross_margin=0.58, monthly_churn=0.025, new_accounts_per_month=4, months=36
    )
    assert set(df["scenario"].unique()) == {"Conservative", "Base", "Aggressive"}
    assert df.shape[0] == 3 * 36


def test_aggressive_scenario_outgrows_conservative():
    df = wholesale_cohort_projection(
        acv=8000, gross_margin=0.58, monthly_churn=0.025, new_accounts_per_month=4, months=36
    )
    final_conservative = df[(df.scenario == "Conservative") & (df.month == 36)]["mrr"].iloc[0]
    final_aggressive = df[(df.scenario == "Aggressive") & (df.month == 36)]["mrr"].iloc[0]
    assert final_aggressive > final_conservative


def test_cohort_retention_curve_starts_at_100_and_decays():
    df = cohort_retention_curve(monthly_churn=0.05, months=24)
    assert df.iloc[0]["pct_retained"] == pytest.approx(100.0)
    assert df.iloc[-1]["pct_retained"] < df.iloc[0]["pct_retained"]
    # monotonically non-increasing
    assert (df["pct_retained"].diff().dropna() <= 1e-9).all()


def test_cohort_retention_curve_flat_at_zero_churn():
    df = cohort_retention_curve(monthly_churn=0.0, months=12)
    assert df["pct_retained"].apply(lambda v: v == pytest.approx(100.0)).all()


# ---------------------------------------------------------------------------
# Retail economics
# ---------------------------------------------------------------------------


def test_retail_payback_is_infinite_when_unprofitable():
    """High rent relative to revenue should make monthly gross profit
    negative, so payback should be inf, not a negative or nonsensical number."""
    result = retail_unit_economics(
        build_out_cost=220_000,
        monthly_rent=50_000,  # deliberately unaffordable
        avg_ticket=6.25,
        daily_transactions=220,
        cogs_pct=0.28,
        labor_pct=0.32,
    )
    assert result["payback_months"] == float("inf")
    assert result["monthly_gross_profit"] < 0


def test_retail_payback_is_positive_and_finite_under_normal_assumptions():
    result = retail_unit_economics(
        build_out_cost=220_000,
        monthly_rent=11_000,
        avg_ticket=6.25,
        daily_transactions=220,
        cogs_pct=0.28,
        labor_pct=0.32,
    )
    assert result["monthly_gross_profit"] > 0
    assert 0 < result["payback_months"] < float("inf")


def test_retail_projection_tracks_store_count():
    df = retail_projection(
        build_out_cost=220_000,
        monthly_rent=11_000,
        avg_ticket=6.25,
        daily_transactions=220,
        cogs_pct=0.28,
        labor_pct=0.32,
        cafes_per_year=2.0,
        months=12,
    )
    # after 12 months at 2/year, ~2 stores should be open
    assert df.iloc[-1]["stores_open"] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# Market sizing
# ---------------------------------------------------------------------------


def test_som_never_exceeds_sam():
    """Even with an unrealistically high new-account velocity, modeled
    capture (SOM) should be capped at the serviceable addressable market —
    you can't sign more accounts than exist in the addressable pool."""
    result = market_sizing(
        addressable_accounts=1000,
        qualified_pct=0.3,
        acv=8000,
        new_accounts_per_month=100,  # unrealistically aggressive on purpose
        months=36,
    )
    assert result["som_accounts"] <= result["sam_accounts"] + 1e-9
    assert result["penetration_of_sam"] <= 1.0 + 1e-9


def test_sam_is_subset_of_tam():
    result = market_sizing(
        addressable_accounts=5200, qualified_pct=0.35, acv=8000, new_accounts_per_month=4
    )
    assert result["sam_accounts"] <= result["tam_accounts"]
    assert result["sam_dollars"] <= result["tam_dollars"]


# ---------------------------------------------------------------------------
# Executive takeaway text generation
# ---------------------------------------------------------------------------


def test_takeaway_returns_list_of_strings():
    lines = generate_takeaway(
        cac_payback=3.1,
        ltv_cac=12.9,
        sales_cycle_months=2.0,
        retail_payback=40.0,
        wholesale_24mo_profit=386_111,
        retail_24mo_cash=-302_500,
        som_penetration=0.08,
    )
    assert isinstance(lines, list)
    assert len(lines) >= 4
    assert all(isinstance(line, str) for line in lines)


def test_takeaway_flags_bad_unit_economics_when_ltv_cac_below_one():
    lines = generate_takeaway(
        cac_payback=30.0,
        ltv_cac=0.4,
        sales_cycle_months=2.0,
        retail_payback=float("inf"),
        wholesale_24mo_profit=-10_000,
        retail_24mo_cash=-50_000,
        som_penetration=0.02,
    )
    combined = " ".join(lines).lower()
    assert "don't work yet" in combined or "cost more to acquire" in combined
