"""
tests/test_model.py — pytest suite for model.py's unit-economics, strategic
summary, and break-even logic.

Run with:  pytest
(install test-only deps first: pip install -r requirements-dev.txt)
"""

import pytest

from model import (
    LTV_HORIZON_MONTHS,
    PRESETS,
    RETAIL_MARKET_DEFAULTS,
    breakeven_robustness_summary,
    cac_payback_months,
    classify_breakeven_robustness,
    compute_breakevens,
    generate_strategic_summary,
    interpret_sam_penetration,
    ltv,
    ltv_cac_sensitivity_to_churn,
    ltv_finite_horizon,
    ltv_to_cac,
    ltv_to_cac_finite,
    market_sizing,
    retail_projection,
    retail_unit_economics,
    wholesale_cohort_projection,
)


# ---------------------------------------------------------------------------
# CAC payback
# ---------------------------------------------------------------------------


def test_payback_decreases_as_acv_increases():
    high_acv_payback = cac_payback_months(cac=3000, acv=24000, gross_margin=0.58)
    low_acv_payback = cac_payback_months(cac=3000, acv=6000, gross_margin=0.58)
    assert high_acv_payback < low_acv_payback


def test_payback_is_infinite_at_zero_margin():
    assert cac_payback_months(cac=1200, acv=8000, gross_margin=0.0) == float("inf")


def test_payback_scales_linearly_with_cac():
    base = cac_payback_months(cac=1000, acv=8000, gross_margin=0.5)
    doubled = cac_payback_months(cac=2000, acv=8000, gross_margin=0.5)
    assert doubled == pytest.approx(base * 2)


# ---------------------------------------------------------------------------
# LTV / LTV:CAC
# ---------------------------------------------------------------------------


def test_ltv_is_infinite_at_zero_churn():
    assert ltv(acv=12000, gross_margin=0.6, monthly_churn=0.0) == float("inf")


def test_ltv_to_cac_matches_manual_calculation():
    acv, margin, churn, cac = 8000, 0.58, 0.025, 1200
    expected_ratio = ((acv / 12 * margin) / churn) / cac
    assert ltv_to_cac(acv, margin, churn, cac) == pytest.approx(expected_ratio)


def test_ltv_to_cac_improves_as_churn_drops():
    high_churn_ratio = ltv_to_cac(acv=8000, gross_margin=0.58, monthly_churn=0.06, cac=1200)
    low_churn_ratio = ltv_to_cac(acv=8000, gross_margin=0.58, monthly_churn=0.01, cac=1200)
    assert low_churn_ratio > high_churn_ratio


def test_ltv_cac_sensitivity_is_monotonic_decreasing_in_churn():
    """Higher churn should never produce a higher LTV:CAC, holding everything
    else fixed — this is the curve the sidebar chart plots directly."""
    df = ltv_cac_sensitivity_to_churn(acv=8000, gross_margin=0.58, cac=1200)
    assert (df["ltv_cac"].diff().dropna() <= 1e-9).all()


# ---------------------------------------------------------------------------
# Finite-horizon LTV — the headline metric, replacing the perpetuity-style
# steady-state number that looked near-infinite at low churn.
# ---------------------------------------------------------------------------


def test_ltv_finite_horizon_is_less_than_or_equal_to_steady_state():
    """A bounded window can never return more than the indefinite-retention
    perpetuity value — this is the whole point of the fix."""
    steady = ltv(acv=9000, gross_margin=0.58, monthly_churn=0.025)
    finite = ltv_finite_horizon(acv=9000, gross_margin=0.58, monthly_churn=0.025, months=36)
    assert finite < steady


def test_ltv_finite_horizon_approaches_steady_state_over_a_long_window():
    steady = ltv(acv=9000, gross_margin=0.58, monthly_churn=0.025)
    very_long_finite = ltv_finite_horizon(acv=9000, gross_margin=0.58, monthly_churn=0.025, months=100_000)
    assert very_long_finite == pytest.approx(steady, rel=1e-3)


def test_ltv_finite_horizon_at_zero_churn_is_flat_monthly_profit_times_months():
    monthly_gp = (9000 / 12) * 0.58
    finite = ltv_finite_horizon(acv=9000, gross_margin=0.58, monthly_churn=0.0, months=36)
    assert finite == pytest.approx(monthly_gp * 36)


def test_ltv_to_cac_finite_matches_manual_calculation():
    acv, margin, churn, cac, months = 9000, 0.58, 0.025, 1500, 36
    monthly_gp = (acv / 12) * margin
    expected_ltv = monthly_gp * (1 - (1 - churn) ** months) / churn
    assert ltv_to_cac_finite(acv, margin, churn, cac, months) == pytest.approx(expected_ltv / cac)


def test_ltv_to_cac_finite_is_infinite_at_zero_cac():
    assert ltv_to_cac_finite(acv=9000, gross_margin=0.58, monthly_churn=0.025, cac=0) == float("inf")


# ---------------------------------------------------------------------------
# Wholesale cohort / scenario projection (now net of CAC spend)
# ---------------------------------------------------------------------------


def test_wholesale_net_cash_is_below_gross_profit_when_cac_is_positive():
    """The whole point of tracking net cash separately from gross profit is
    that CAC spend should reduce it — this is the correctness fix from the
    original model, so it's worth asserting directly."""
    df = wholesale_cohort_projection(
        acv=8000, gross_margin=0.58, monthly_churn=0.025, cac=1200,
        new_accounts_per_month=4, months=24,
    )
    row = df[(df.scenario == "Base") & (df.month == 24)].iloc[0]
    assert row["cum_net_cash_position"] < row["cum_gross_profit"]


def test_wholesale_net_cash_equals_gross_profit_at_zero_cac():
    df = wholesale_cohort_projection(
        acv=8000, gross_margin=0.58, monthly_churn=0.025, cac=0,
        new_accounts_per_month=4, months=24,
    )
    row = df[(df.scenario == "Base") & (df.month == 24)].iloc[0]
    assert row["cum_net_cash_position"] == pytest.approx(row["cum_gross_profit"])


def test_wholesale_projection_returns_all_three_scenarios():
    df = wholesale_cohort_projection(
        acv=8000, gross_margin=0.58, monthly_churn=0.025, cac=1200,
        new_accounts_per_month=4, months=36,
    )
    assert set(df["scenario"].unique()) == {"Conservative", "Base", "Aggressive"}
    assert df.shape[0] == 3 * 36


def test_aggressive_scenario_outgrows_conservative():
    df = wholesale_cohort_projection(
        acv=8000, gross_margin=0.58, monthly_churn=0.025, cac=1200,
        new_accounts_per_month=4, months=36,
    )
    final_conservative = df[(df.scenario == "Conservative") & (df.month == 36)]["mrr"].iloc[0]
    final_aggressive = df[(df.scenario == "Aggressive") & (df.month == 36)]["mrr"].iloc[0]
    assert final_aggressive > final_conservative


# ---------------------------------------------------------------------------
# Retail economics
# ---------------------------------------------------------------------------


def test_retail_breakdown_sums_to_gross_profit():
    result = retail_unit_economics(
        build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25,
        daily_transactions=220, cogs_pct=0.28, labor_pct=0.32,
    )
    recomputed = (
        result["monthly_revenue"] - result["cogs_amount"] - result["labor_amount"] - result["rent"]
    )
    assert recomputed == pytest.approx(result["monthly_gross_profit"])


def test_retail_payback_is_infinite_when_unprofitable():
    result = retail_unit_economics(
        build_out_cost=220_000, monthly_rent=50_000, avg_ticket=6.25,
        daily_transactions=220, cogs_pct=0.28, labor_pct=0.32,
    )
    assert result["payback_months"] == float("inf")
    assert result["monthly_gross_profit"] < 0


def test_retail_payback_is_positive_and_finite_under_normal_assumptions():
    result = retail_unit_economics(
        build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25,
        daily_transactions=220, cogs_pct=0.28, labor_pct=0.32,
    )
    assert result["monthly_gross_profit"] > 0
    assert 0 < result["payback_months"] < float("inf")


def test_retail_projection_tracks_store_count():
    df = retail_projection(
        build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25,
        daily_transactions=220, cogs_pct=0.28, labor_pct=0.32,
        cafes_per_year=2.0, months=12,
    )
    assert df.iloc[-1]["stores_open"] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# Market sizing
# ---------------------------------------------------------------------------


def test_som_never_exceeds_sam():
    result = market_sizing(
        addressable_accounts=1000, qualified_pct=0.3, acv=8000,
        new_accounts_per_month=100, months=36,
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
# Strategic summary
# ---------------------------------------------------------------------------


def test_strategic_summary_recommends_wholesale_when_it_leads():
    summary = generate_strategic_summary(
        ltv_cac=12.9, cac_payback=3.1, retail_payback_months=40.0,
        wholesale_net_24=270_000, retail_net_24=-300_000,
    )
    assert summary["recommendation"]["choice"] == "Wholesale"


def test_strategic_summary_recommends_retail_when_it_leads():
    summary = generate_strategic_summary(
        ltv_cac=12.9, cac_payback=3.1, retail_payback_months=2.4,
        wholesale_net_24=270_000, retail_net_24=2_500_000,
    )
    assert summary["recommendation"]["choice"] == "Retail"


def test_strategic_summary_recommends_neither_when_both_fail():
    summary = generate_strategic_summary(
        ltv_cac=0.6, cac_payback=30.0, retail_payback_months=float("inf"),
        wholesale_net_24=-50_000, retail_net_24=-400_000,
    )
    assert summary["recommendation"]["choice"] == "Neither"


def test_strategic_summary_includes_total_cash_to_recovery_when_sales_cycle_given():
    """sales_cycle_months is a real sidebar input (not a decoration) — it
    should show up in the wholesale interpretation as the total time from
    outreach to cash recovery, not just get collected and dropped."""
    summary = generate_strategic_summary(
        ltv_cac=12.9, cac_payback=3.1, retail_payback_months=40.0,
        wholesale_net_24=270_000, retail_net_24=-300_000, sales_cycle_months=2.0,
    )
    assert "5.1" in summary["wholesale"]["interpretation"]


def test_strategic_summary_has_no_bare_dollar_pairs_that_break_markdown():
    """Streamlit renders a pair of literal '$' in one markdown string as
    inline LaTeX. generate_strategic_summary's job is just to produce the
    text — app.py is responsible for escaping before display — but this
    guards against ever depending on unescaped output looking fine by luck."""
    summary = generate_strategic_summary(
        ltv_cac=12.9, cac_payback=3.1, retail_payback_months=40.0,
        wholesale_net_24=270_000, retail_net_24=-300_000,
    )
    reason = summary["recommendation"]["reason"]
    assert reason.count("$") % 2 == 0  # even count is escapable in pairs by app.py


# ---------------------------------------------------------------------------
# Break-even thresholds
# ---------------------------------------------------------------------------


COMMON_INPUTS = dict(
    acv=8000, gross_margin_ws=0.58, monthly_churn=0.025, cac=1200, new_accounts_per_month=4,
    build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25, daily_transactions=220,
    cogs_pct=0.28, labor_pct=0.32, cafes_per_year=1.0,
)


def test_breakevens_returns_all_four_drivers():
    result = compute_breakevens(**COMMON_INPUTS)
    assert set(result.keys()) == {"churn", "cac", "cafe_revenue", "build_out"}


def test_churn_breakeven_matches_finite_horizon_ltv_cac_one():
    """The churn break-even should be the churn at which the *finite-horizon*
    LTV:CAC (the headline metric) equals 1 — not the steady-state formula,
    since the table right above it is explaining the headline number."""
    result = compute_breakevens(**COMMON_INPUTS)
    acv, margin, cac = COMMON_INPUTS["acv"], COMMON_INPUTS["gross_margin_ws"], COMMON_INPUTS["cac"]
    breakeven = result["churn"]["breakeven"]
    assert breakeven is not None
    ratio_at_breakeven = ltv_to_cac_finite(acv, margin, breakeven, cac, months=LTV_HORIZON_MONTHS)
    assert ratio_at_breakeven == pytest.approx(1.0, rel=1e-4)


def test_cac_breakeven_actually_flips_the_comparison():
    """The CAC break-even value should be the point where wholesale's net
    cash at month 24 equals retail's — verify by plugging it back in."""
    from model import _retail_net_cash_at, _wholesale_net_cash_at

    result = compute_breakevens(**COMMON_INPUTS)
    cac_star = result["cac"]["breakeven"]
    assert cac_star is not None

    wholesale_at_star = _wholesale_net_cash_at(
        COMMON_INPUTS["acv"], COMMON_INPUTS["gross_margin_ws"], COMMON_INPUTS["monthly_churn"],
        cac_star, COMMON_INPUTS["new_accounts_per_month"], 24,
    )
    retail_net = _retail_net_cash_at(
        COMMON_INPUTS["build_out_cost"], COMMON_INPUTS["monthly_rent"], COMMON_INPUTS["avg_ticket"],
        COMMON_INPUTS["daily_transactions"], COMMON_INPUTS["cogs_pct"], COMMON_INPUTS["labor_pct"],
        COMMON_INPUTS["cafes_per_year"], 24,
    )
    assert wholesale_at_star == pytest.approx(retail_net, rel=1e-6)


def test_build_out_breakeven_flagged_out_of_range_when_unreachable():
    """At the default inputs, even a free build-out doesn't make retail
    catch up to wholesale — this should be flagged, not shown as a
    fabricated (e.g. negative) number."""
    result = compute_breakevens(**COMMON_INPUTS)
    entry = result["build_out"]
    if not entry["in_range"]:
        assert entry["breakeven"] is None
    else:
        assert entry["breakeven"] >= 0


def test_breakeven_direction_is_consistent_with_current_vs_breakeven():
    result = compute_breakevens(**COMMON_INPUTS)
    for entry in result.values():
        if entry["in_range"]:
            if entry["breakeven"] > entry["current"]:
                assert entry["direction"] == "up"
            else:
                assert entry["direction"] == "down"


# ---------------------------------------------------------------------------
# Break-even robustness interpretation
# ---------------------------------------------------------------------------


def test_classify_breakeven_robustness_out_of_range_is_robust():
    result = compute_breakevens(**COMMON_INPUTS)
    out_of_range = [e for e in result.values() if not e["in_range"]]
    assert out_of_range, "fixture should have at least one out-of-range driver to test against"
    assert classify_breakeven_robustness(out_of_range[0]).startswith("Robust")


def test_classify_breakeven_robustness_close_breakeven_is_flagged_fragile():
    entry = {"label": "Test driver", "current": 100.0, "breakeven": 105.0, "in_range": True}
    assert "fragile" in classify_breakeven_robustness(entry).lower()


def test_classify_breakeven_robustness_far_breakeven_is_not_flagged_fragile():
    entry = {"label": "Test driver", "current": 100.0, "breakeven": 500.0, "in_range": True}
    assert "fragile" not in classify_breakeven_robustness(entry).lower()


def test_breakeven_robustness_summary_all_robust_says_so():
    fake = {