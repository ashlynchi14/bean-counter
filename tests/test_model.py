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
    _build_lagged_schedule,
    _retail_net_cash_at,
    _wholesale_net_cash_at,
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
    high_acv_payback = cac_payback_months(cac=3000, acv=24000, contribution_margin=0.58)
    low_acv_payback = cac_payback_months(cac=3000, acv=6000, contribution_margin=0.58)
    assert high_acv_payback < low_acv_payback


def test_payback_is_infinite_at_zero_margin():
    assert cac_payback_months(cac=1200, acv=8000, contribution_margin=0.0) == float("inf")


def test_payback_scales_linearly_with_cac():
    base = cac_payback_months(cac=1000, acv=8000, contribution_margin=0.5)
    doubled = cac_payback_months(cac=2000, acv=8000, contribution_margin=0.5)
    assert doubled == pytest.approx(base * 2)


# ---------------------------------------------------------------------------
# LTV / LTV:CAC
# ---------------------------------------------------------------------------


def test_ltv_is_infinite_at_zero_churn():
    assert ltv(acv=12000, contribution_margin=0.6, monthly_churn=0.0) == float("inf")


def test_ltv_to_cac_matches_manual_calculation():
    acv, margin, churn, cac = 8000, 0.58, 0.025, 1200
    expected_ratio = ((acv / 12 * margin) / churn) / cac
    assert ltv_to_cac(acv, margin, churn, cac) == pytest.approx(expected_ratio)


def test_ltv_to_cac_improves_as_churn_drops():
    high_churn_ratio = ltv_to_cac(acv=8000, contribution_margin=0.58, monthly_churn=0.06, cac=1200)
    low_churn_ratio = ltv_to_cac(acv=8000, contribution_margin=0.58, monthly_churn=0.01, cac=1200)
    assert low_churn_ratio > high_churn_ratio


def test_ltv_cac_sensitivity_is_monotonic_decreasing_in_churn():
    """Higher churn should never produce a higher LTV:CAC, holding everything
    else fixed — this is the curve the sidebar chart plots directly."""
    df = ltv_cac_sensitivity_to_churn(acv=8000, contribution_margin=0.58, cac=1200)
    assert (df["ltv_cac"].diff().dropna() <= 1e-9).all()


# ---------------------------------------------------------------------------
# Finite-horizon LTV — the headline metric, replacing the perpetuity-style
# steady-state number that looked near-infinite at low churn.
# ---------------------------------------------------------------------------


def test_ltv_finite_horizon_is_less_than_or_equal_to_steady_state():
    """A bounded window can never return more than the indefinite-retention
    perpetuity value — this is the whole point of the fix."""
    steady = ltv(acv=9000, contribution_margin=0.58, monthly_churn=0.025)
    finite = ltv_finite_horizon(acv=9000, contribution_margin=0.58, monthly_churn=0.025, months=36)
    assert finite < steady


def test_ltv_finite_horizon_approaches_steady_state_over_a_long_window():
    steady = ltv(acv=9000, contribution_margin=0.58, monthly_churn=0.025)
    very_long_finite = ltv_finite_horizon(acv=9000, contribution_margin=0.58, monthly_churn=0.025, months=100_000)
    assert very_long_finite == pytest.approx(steady, rel=1e-3)


def test_ltv_finite_horizon_at_zero_churn_is_flat_monthly_profit_times_months():
    monthly_gp = (9000 / 12) * 0.58
    finite = ltv_finite_horizon(acv=9000, contribution_margin=0.58, monthly_churn=0.0, months=36)
    assert finite == pytest.approx(monthly_gp * 36)


def test_ltv_to_cac_finite_matches_manual_calculation():
    acv, margin, churn, cac, months = 9000, 0.58, 0.025, 1500, 36
    monthly_gp = (acv / 12) * margin
    expected_ltv = monthly_gp * (1 - (1 - churn) ** months) / churn
    assert ltv_to_cac_finite(acv, margin, churn, cac, months) == pytest.approx(expected_ltv / cac)


def test_ltv_to_cac_finite_is_infinite_at_zero_cac():
    assert ltv_to_cac_finite(acv=9000, contribution_margin=0.58, monthly_churn=0.025, cac=0) == float("inf")


# ---------------------------------------------------------------------------
# Wholesale cohort / scenario projection (now net of CAC spend)
# ---------------------------------------------------------------------------


def test_wholesale_net_cash_is_below_gross_profit_when_cac_is_positive():
    """The whole point of tracking net cash separately from gross profit is
    that CAC spend should reduce it — this is the correctness fix from the
    original model, so it's worth asserting directly."""
    df = wholesale_cohort_projection(
        acv=8000, contribution_margin=0.58, monthly_churn=0.025, cac=1200,
        new_accounts_per_month=4, months=24,
    )
    row = df[(df.scenario == "Base") & (df.month == 24)].iloc[0]
    assert row["cum_net_cash_position"] < row["cum_contribution_profit"]


def test_wholesale_net_cash_equals_gross_profit_at_zero_cac():
    df = wholesale_cohort_projection(
        acv=8000, contribution_margin=0.58, monthly_churn=0.025, cac=0,
        new_accounts_per_month=4, months=24,
    )
    row = df[(df.scenario == "Base") & (df.month == 24)].iloc[0]
    assert row["cum_net_cash_position"] == pytest.approx(row["cum_contribution_profit"])


def test_wholesale_projection_returns_all_three_scenarios():
    df = wholesale_cohort_projection(
        acv=8000, contribution_margin=0.58, monthly_churn=0.025, cac=1200,
        new_accounts_per_month=4, months=36,
    )
    assert set(df["scenario"].unique()) == {"Conservative", "Base", "Aggressive"}
    assert df.shape[0] == 3 * 36


def test_aggressive_scenario_outgrows_conservative():
    df = wholesale_cohort_projection(
        acv=8000, contribution_margin=0.58, monthly_churn=0.025, cac=1200,
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
    acv=8000, contribution_margin_ws=0.58, monthly_churn=0.025, cac=1200, new_accounts_per_month=4,
    build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25, daily_transactions=220,
    cogs_pct=0.28, labor_pct=0.32, cafes_per_year=1.0,
)


def test_breakevens_returns_all_four_drivers():
    result = compute_breakevens(**COMMON_INPUTS)
    assert set(result.keys()) == {"churn", "cac", "cafe_revenue", "build_out"}


def test_churn_breakeven_uses_the_same_objective_as_the_recommendation():
    """Audit finding (Critical, fixed): the churn break-even must answer
    the same question as the other three rows — the churn at which
    wholesale's net cash at the comparison horizon equals retail's — not
    the unrelated single-account LTV:CAC=1 threshold, which can differ
    substantially from where the actual recommendation flips."""
    from model import _retail_net_cash_at, _wholesale_net_cash_at

    # Retail assumptions tuned so a churn break-even actually falls in range,
    # to exercise the round-trip check meaningfully.
    retail_kwargs = dict(
        build_out_cost=150_000, monthly_rent=8_000, avg_ticket=7.0,
        daily_transactions=250, cogs_pct=0.28, labor_pct=0.30, cafes_per_year=1.0,
    )
    inputs = {**COMMON_INPUTS, **retail_kwargs}
    result = compute_breakevens(**inputs)
    churn_star = result["churn"]["breakeven"]
    assert churn_star is not None

    retail_net = _retail_net_cash_at(
        inputs["build_out_cost"], inputs["monthly_rent"], inputs["avg_ticket"],
        inputs["daily_transactions"], inputs["cogs_pct"], inputs["labor_pct"],
        inputs["cafes_per_year"], 24,
    )
    wholesale_at_star = _wholesale_net_cash_at(
        inputs["acv"], inputs["contribution_margin_ws"], churn_star, inputs["cac"],
        inputs["new_accounts_per_month"], 24,
    )
    assert wholesale_at_star == pytest.approx(retail_net, rel=1e-6)


def test_churn_breakeven_is_out_of_range_when_no_churn_value_flips_the_call():
    """At the default fixture, retail is deep in its build-out payback
    period at month 24 — no realistic churn value makes wholesale fall
    behind it, so the table should say so rather than showing a
    misleading precise-looking number (this was the actual bug: the old
    LTV:CAC=1 formula returned a specific churn value here even though
    churn alone can never flip this particular recommendation)."""
    result = compute_breakevens(**COMMON_INPUTS)
    assert result["churn"]["in_range"] is False
    assert result["churn"]["breakeven"] is None


def test_cac_breakeven_actually_flips_the_comparison():
    """The CAC break-even value should be the point where wholesale's net
    cash at month 24 equals retail's — verify by plugging it back in."""
    from model import _retail_net_cash_at, _wholesale_net_cash_at

    result = compute_breakevens(**COMMON_INPUTS)
    cac_star = result["cac"]["breakeven"]
    assert cac_star is not None

    wholesale_at_star = _wholesale_net_cash_at(
        COMMON_INPUTS["acv"], COMMON_INPUTS["contribution_margin_ws"], COMMON_INPUTS["monthly_churn"],
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
        "a": {"label": "A", "current": 1.0, "breakeven": None, "in_range": False},
        "b": {"label": "B", "current": 1.0, "breakeven": None, "in_range": False},
    }
    summary = breakeven_robustness_summary(fake)
    assert "robust to every driver" in summary


def test_breakeven_robustness_summary_names_the_most_fragile_driver():
    fake = {
        "a": {"label": "Driver A", "current": 100.0, "breakeven": 101.0, "in_range": True},
        "b": {"label": "Driver B", "current": 100.0, "breakeven": 400.0, "in_range": True},
    }
    summary = breakeven_robustness_summary(fake)
    assert "driver a" in summary.lower()


# ---------------------------------------------------------------------------
# SAM penetration sanity check
# ---------------------------------------------------------------------------


def test_interpret_sam_penetration_bands_are_ordered_and_non_empty():
    bands = [0.05, 0.15, 0.25, 0.40]
    texts = [interpret_sam_penetration(b) for b in bands]
    assert all(texts)
    assert len(set(texts)) == 4  # each band gets distinct wording


def test_interpret_sam_penetration_low_is_conservative():
    assert "conservative" in interpret_sam_penetration(0.05).lower()


def test_interpret_sam_penetration_high_is_aggressive():
    assert "aggressive" in interpret_sam_penetration(0.35).lower()


# ---------------------------------------------------------------------------
# Assumption presets
# ---------------------------------------------------------------------------


def test_presets_include_conservative_base_and_aggressive():
    assert set(PRESETS.keys()) == {"Conservative", "Base case", "Aggressive"}


def test_presets_share_the_same_keys():
    key_sets = [set(p.keys()) for p in PRESETS.values()]
    assert key_sets[0] == key_sets[1] == key_sets[2]


def test_presets_are_ordered_conservative_to_aggressive():
    """Aggressive should look better than Base, which should look better
    than Conservative, on every wholesale driver — otherwise the labels
    would be misleading."""
    c, b, a = PRESETS["Conservative"], PRESETS["Base case"], PRESETS["Aggressive"]
    assert c["acv"] < b["acv"] < a["acv"]
    assert c["cac"] > b["cac"] > a["cac"]
    assert c["monthly_churn"] > b["monthly_churn"] > a["monthly_churn"]
    assert c["new_accounts"] < b["new_accounts"] < a["new_accounts"]
    assert c["contribution_margin_ws"] < b["contribution_margin_ws"] < a["contribution_margin_ws"]


def test_base_case_preset_is_defensible_not_engineered_to_dominate():
    """The Base case shouldn't produce an implausibly large LTV:CAC — this
    guards against shipping defaults that look rigged to make wholesale
    win, which is exactly what this preset system was built to avoid."""
    p = PRESETS["Base case"]
    ratio = ltv_to_cac_finite(
        p["acv"], p["contribution_margin_ws"] / 100, p["monthly_churn"] / 100, p["cac"], months=LTV_HORIZON_MONTHS
    )
    assert 1 < ratio < 15


def test_retail_market_defaults_produce_profitable_retail_unit_economics():
    """Base Case retail defaults should be a fair fight against wholesale —
    not secretly rigged to lose."""
    d = RETAIL_MARKET_DEFAULTS
    result = retail_unit_economics(
        d["build_out"], d["rent"], d["avg_ticket"], d["daily_tx"],
        d["cogs_pct"] / 100, d["labor_pct"] / 100,
    )
    assert result["monthly_gross_profit"] > 0
    assert result["payback_months"] < 60


def test_retail_market_defaults_include_a_reasonable_opening_lag():
    assert 0 < RETAIL_MARKET_DEFAULTS["opening_lag"] <= 12


# ---------------------------------------------------------------------------
# Timing lags — second audit round: sales-cycle and cafe-opening lags must
# actually move cash, not just narrative text.
# ---------------------------------------------------------------------------


def test_lagged_schedule_with_zero_lag_matches_immediate_activation():
    """At lag=0, every event should complete in the same month it starts —
    this is the backward-compatibility case every un-lagged test above
    relies on."""
    schedule = _build_lagged_schedule(monthly_rate=10.0, lag_months=0.0, months=6)
    for m in range(1, 7):
        assert schedule[m] == pytest.approx(10.0)


def test_lagged_schedule_integer_lag_shifts_completions_by_that_many_months():
    schedule = _build_lagged_schedule(monthly_rate=5.0, lag_months=3.0, months=4)
    # nothing completes before month 4 (1 + 3)
    assert schedule[1] == 0 and schedule[2] == 0 and schedule[3] == 0
    assert schedule[4] == pytest.approx(5.0)
    assert schedule[7] == pytest.approx(5.0)  # month 4's start (4+3)


def test_lagged_schedule_fractional_lag_splits_between_two_months():
    """A 2.5-month lag should split each month's cohort roughly in half
    between the 2- and 3-month completion points, matching the app's own
    description ('begins contributing revenue approximately 2-3 months
    later')."""
    schedule = _build_lagged_schedule(monthly_rate=10.0, lag_months=2.5, months=1)
    assert schedule[3] == pytest.approx(5.0)   # floor: 1+2
    assert schedule[4] == pytest.approx(5.0)   # ceil: 1+3
    assert schedule[3] + schedule[4] == pytest.approx(10.0)  # nothing lost


def test_lagged_schedule_conserves_total_volume():
    """Every unit that starts must complete somewhere in the schedule —
    lag should delay, never destroy, volume."""
    schedule = _build_lagged_schedule(monthly_rate=7.0, lag_months=4.25, months=10)
    assert sum(schedule) == pytest.approx(7.0 * 10)


def test_sales_cycle_lag_delays_but_converges_toward_the_same_steady_state():
    """Churn pulls the active-account population toward a steady state
    regardless of when cohorts started, so a fixed sales-cycle lag should
    matter less and less the further out you look — the lag delays cash,
    it doesn't destroy accounts. Check the relative gap actually shrinks
    between month 30 and month 90."""
    def active_at(month, sales_cycle_months):
        df = wholesale_cohort_projection(
            acv=8000, contribution_margin=0.5, monthly_churn=0.03, cac=1200,
            new_accounts_per_month=4, months=month, sales_cycle_months=sales_cycle_months,
        )
        return df[(df.scenario == "Base") & (df.month == month)]["active_accounts"].iloc[0]

    gap_30 = active_at(30, 0.0) - active_at(30, 3.0)
    gap_90 = active_at(90, 0.0) - active_at(90, 3.0)
    assert 0 < gap_90 < gap_30  # still behind, but catching up


def test_sales_cycle_lag_strictly_worsens_cumulative_net_cash_at_a_fixed_month():
    """A longer sales cycle means CAC is still spent on schedule but revenue
    starts later — cumulative net cash at any fixed early/mid month should
    never improve as the lag grows."""
    kwargs = dict(acv=8000, contribution_margin=0.5, monthly_churn=0.03, cac=1200, new_accounts_per_month=4)
    net_0 = _wholesale_net_cash_at(**kwargs, month=18, sales_cycle_months=0.0)
    net_2 = _wholesale_net_cash_at(**kwargs, month=18, sales_cycle_months=2.0)
    net_5 = _wholesale_net_cash_at(**kwargs, month=18, sales_cycle_months=5.0)
    assert net_0 > net_2 > net_5


def test_cac_is_charged_on_schedule_regardless_of_sales_cycle_length():
    """CAC spend must be anchored to when outreach begins, not shifted by
    the lag — verified by checking month-1 cash position is CAC spend
    minus zero revenue (nothing has activated yet) at any sales-cycle
    length >= 1 month."""
    for sales_cycle in [1.0, 2.5, 6.0]:
        df = wholesale_cohort_projection(
            acv=8000, contribution_margin=0.5, monthly_churn=0.03, cac=1200,
            new_accounts_per_month=4, months=1, sales_cycle_months=sales_cycle,
        )
        row = df[(df.scenario == "Base") & (df.month == 1)].iloc[0]
        assert row["mrr"] == pytest.approx(0.0)  # nothing active yet
        assert row["cum_net_cash_position"] == pytest.approx(-4 * 1200)  # CAC spent regardless


def test_very_long_sales_cycle_can_flip_the_recommendation_to_retail():
    """Extreme-value check requested in the audit: pushing the sales cycle
    high enough, with everything else fixed, should be capable of making
    wholesale look worse than retail — proving the lag has real teeth
    rather than being cosmetic."""
    common = dict(
        acv=8000, contribution_margin_ws=0.5, monthly_churn=0.03, cac=1200, new_accounts_per_month=4,
        build_out_cost=150_000, monthly_rent=8_000, avg_ticket=7.0, daily_transactions=250,
        cogs_pct=0.28, labor_pct=0.30, cafes_per_year=1.0,
    )
    short_cycle = generate_strategic_summary(
        ltv_cac=ltv_to_cac_finite(common["acv"], common["contribution_margin_ws"], common["monthly_churn"], common["cac"]),
        cac_payback=cac_payback_months(common["cac"], common["acv"], common["contribution_margin_ws"]),
        retail_payback_months=retail_unit_economics(150_000, 8_000, 7.0, 250, 0.28, 0.30)["payback_months"],
        wholesale_net_24=_wholesale_net_cash_at(
            common["acv"], common["contribution_margin_ws"], common["monthly_churn"], common["cac"],
            common["new_accounts_per_month"], 24, sales_cycle_months=0.5,
        ),
        retail_net_24=_retail_net_cash_at(
            common["build_out_cost"], common["monthly_rent"], common["avg_ticket"], common["daily_transactions"],
            common["cogs_pct"], common["labor_pct"], common["cafes_per_year"], 24,
        ),
    )
    long_cycle = generate_strategic_summary(
        ltv_cac=ltv_to_cac_finite(common["acv"], common["contribution_margin_ws"], common["monthly_churn"], common["cac"]),
        cac_payback=cac_payback_months(common["cac"], common["acv"], common["contribution_margin_ws"]),
        retail_payback_months=retail_unit_economics(150_000, 8_000, 7.0, 250, 0.28, 0.30)["payback_months"],
        wholesale_net_24=_wholesale_net_cash_at(
            common["acv"], common["contribution_margin_ws"], common["monthly_churn"], common["cac"],
            common["new_accounts_per_month"], 24, sales_cycle_months=18.0,
        ),
        retail_net_24=_retail_net_cash_at(
            common["build_out_cost"], common["monthly_rent"], common["avg_ticket"], common["daily_transactions"],
            common["cogs_pct"], common["labor_pct"], common["cafes_per_year"], 24,
        ),
    )
    assert short_cycle["recommendation"]["choice"] == "Wholesale"
    assert long_cycle["recommendation"]["choice"] == "Retail"


def test_opening_lag_shifts_store_count_by_exactly_the_lagged_commitments():
    """Cafes are committed at a constant rate forever, so a fixed opening
    lag permanently shifts stores_open by exactly cafes_per_year *
    (lag / 12) stores — not an approximation, an exact consequence of a
    constant-rate schedule. This is the precise version of "delays but
    doesn't destroy" for a channel that never reaches a self-correcting
    steady state the way churn-driven wholesale does."""
    no_lag = retail_projection(
        build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25, daily_transactions=220,
        cogs_pct=0.28, labor_pct=0.32, cafes_per_year=2.0, months=36, opening_lag_months=0.0,
    )
    with_lag = retail_projection(
        build_out_cost=220_000, monthly_rent=11_000, avg_ticket=6.25, daily_transactions=220,
        cogs_pct=0.28, labor_pct=0.32, cafes_per_year=2.0, months=36, opening_lag_months=6.0,
    )
    expected_shortfall = 2.0 * (6.0 / 12)
    assert no_lag.iloc[-1]["stores_open"] - with_lag.iloc[-1]["stores_open"] == pytest.approx(expected_shortfall)


def test_opening_lag_strictly_worsens_cumulative_retail_cash_at_a_fixed_month():
    kwargs = dict(build_out_cost=150_000, monthly_rent=8_000, avg_ticket=7.0, daily_transactions=250, cogs_pct=0.28, labor_pct=0.30, cafes_per_year=1.0)
    net_0 = _retail_net_cash_at(**kwargs, month=18, opening_lag_months=0.0)
    net_3 = _retail_net_cash_at(**kwargs, month=18, opening_lag_months=3.0)
    net_8 = _retail_net_cash_at(**kwargs, month=18, opening_lag_months=8.0)
    assert net_0 > net_3 > net_8


def test_build_out_capex_is_charged_on_schedule_regardless_of_opening_lag():
    """Capex commitment timing must not shift with the opening lag — only
    revenue should be delayed. With any lag >= 1 month, month 1 has zero
    stores open (no revenue yet), so cum_cash_position in month 1 should
    equal exactly one month's prorated capex, no more and no less."""
    for lag in [1.0, 4.0, 9.0]:
        df = retail_projection(
            build_out_cost=120_000, monthly_rent=5_000, avg_ticket=7.0, daily_transactions=200,
            cogs_pct=0.28, labor_pct=0.30, cafes_per_year=1.0, months=1, opening_lag_months=lag,
        )
        row = df.iloc[0]
        assert row["stores_open"] == pytest.approx(0.0)
        assert row["cum_cash_position"] == pytest.approx(-120_000 / 12)


def test_very_long_opening_lag_can_flip_the_recommendation_to_wholesale():
    """Mirror of the sales-cycle extreme-value check: pushing the cafe
    opening lag high enough should be able to make retail look worse than
    wholesale, proving the retail lag also has real teeth."""
    wholesale_kwargs = dict(acv=8000, contribution_margin_ws=0.5, monthly_churn=0.03, cac=1200, new_accounts_per_month=4)
    retail_kwargs = dict(build_out_cost=90_000, monthly_rent=4_000, avg_ticket=9.0, daily_transactions=300, cogs_pct=0.22, labor_pct=0.22, cafes_per_year=2.0)

    def summary_at(opening_lag):
        return generate_strategic_summary(
            ltv_cac=ltv_to_cac_finite(wholesale_kwargs["acv"], wholesale_kwargs["contribution_margin_ws"], wholesale_kwargs["monthly_churn"], wholesale_kwargs["cac"]),
            cac_payback=cac_payback_months(wholesale_kwargs["cac"], wholesale_kwargs["acv"], wholesale_kwargs["contribution_margin_ws"]),
            retail_payback_months=retail_unit_economics(
                retail_kwargs["build_out_cost"], retail_kwargs["monthly_rent"], retail_kwargs["avg_ticket"],
                retail_kwargs["daily_transactions"], retail_kwargs["cogs_pct"], retail_kwargs["labor_pct"],
            )["payback_months"],
            wholesale_net_24=_wholesale_net_cash_at(
                wholesale_kwargs["acv"], wholesale_kwargs["contribution_margin_ws"], wholesale_kwargs["monthly_churn"],
                wholesale_kwargs["cac"], wholesale_kwargs["new_accounts_per_month"], 24,
            ),
            retail_net_24=_retail_net_cash_at(
                retail_kwargs["build_out_cost"], retail_kwargs["monthly_rent"], retail_kwargs["avg_ticket"],
                retail_kwargs["daily_transactions"], retail_kwargs["cogs_pct"], retail_kwargs["labor_pct"],
                retail_kwargs["cafes_per_year"], 24, opening_lag_months=opening_lag,
            ),
        )

    short_lag = summary_at(0.5)
    long_lag = summary_at(20.0)
    assert short_lag["recommendation"]["choice"] == "Retail"
    assert long_lag["recommendation"]["choice"] == "Wholesale"


def test_very_low_contribution_margin_makes_wholesale_unviable():
    """Extreme-value check: at the slider-range extremes (lowest contribution
    margin, highest churn, highest CAC), wholesale should be flagged
    unviable (LTV:CAC < 1), not silently treated as fine."""
    ratio = ltv_to_cac_finite(acv=8000, contribution_margin=0.20, monthly_churn=0.08, cac=5000)
    assert ratio < 1
    summary = generate_strategic_summary(
        ltv_cac=ratio, cac_payback=cac_payback_months(5000, 8000, 0.20),
        retail_payback_months=float("inf"), wholesale_net_24=-50_000, retail_net_24=-400_000,
    )
    assert summary["recommendation"]["choice"] == "Neither"


def test_breakevens_thread_sales_cycle_and_opening_lag_into_the_same_objective():
    """Audit re-check: the break-even rows must still solve against the
    exact wholesale-vs-retail net cash comparison once timing lags are
    included, not against a lag-naive version of that comparison."""
    inputs = dict(
        acv=8000, contribution_margin_ws=0.5, monthly_churn=0.03, cac=1200, new_accounts_per_month=4,
        build_out_cost=150_000, monthly_rent=8_000, avg_ticket=7.0, daily_transactions=250,
        cogs_pct=0.28, labor_pct=0.30, cafes_per_year=1.0,
        sales_cycle_months=2.5, opening_lag_months=4.0,
    )
    result = compute_breakevens(**inputs)
    cac_star = result["cac"]["breakeven"]
    assert cac_star is not None
    retail_net = _retail_net_cash_at(
        inputs["build_out_cost"], inputs["monthly_rent"], inputs["avg_ticket"], inputs["daily_transactions"],
        inputs["cogs_pct"], inputs["labor_pct"], inputs["cafes_per_year"], 24,
        opening_lag_months=inputs["opening_lag_months"],
    )
    wholesale_at_star = _wholesale_net_cash_at(
        inputs["acv"], inputs["contribution_margin_ws"], inputs["monthly_churn"], cac_star,
        inputs["new_accounts_per_month"], 24, sales_cycle_months=inputs["sales_cycle_months"],
    )
    assert wholesale_at_star == pytest.approx(retail_net, rel=1e-6)