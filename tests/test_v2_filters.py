"""Tests for the Mock Two Phase 2 filter helpers.

Each helper is small enough to unit-test in isolation. Integration with
ev_backtest_simulate_v2 is covered by the walk-forward validation script
(scripts/wf_validate_v2_features.py) — these tests just lock in the maths.
"""
from __future__ import annotations

from datetime import datetime

import pytest

import portfolio as pf


# ─── compute_team_roi_from_bets ──────────────────────────────────────────────

def _bet(home, away, market, stake, profit, status="won"):
    return {"home": home, "away": away, "market": market,
            "stake": stake, "profit": profit, "status": status}


class TestComputeTeamRoi:

    def test_empty_input_returns_empty(self):
        assert pf.compute_team_roi_from_bets([]) == {}

    def test_pending_bets_excluded(self):
        bets = [_bet("A", "B", "D", 100, 50, status="pending")]
        assert pf.compute_team_roi_from_bets(bets, min_n=1) == {}

    def test_min_n_threshold_filters_low_sample_teams(self):
        bets = [
            _bet("A", "B", "D", 100,  50),    # A: 1
            _bet("C", "D", "D", 100, -100, "lost"),  # C, D: 1 each
        ]
        # min_n=2 → all teams below threshold, returns empty
        assert pf.compute_team_roi_from_bets(bets, min_n=2) == {}
        # min_n=1 → all teams pass
        out = pf.compute_team_roi_from_bets(bets, min_n=1)
        assert set(out.keys()) == {"A", "B", "C", "D"}

    def test_roi_pct_computation(self):
        # Two bets on team X — one wins £50 on £100 stake, one loses £50 on £50 stake
        bets = [
            _bet("X", "Y", "D", 100,  50),         # win
            _bet("X", "Z", "D",  50, -50, "lost"), # loss
        ]
        out = pf.compute_team_roi_from_bets(bets, min_n=1)
        # Team X: stake 150, profit 0 → ROI 0
        assert out["X"]["roi"] == 0.0
        assert out["X"]["n"] == 2

    def test_market_filter(self):
        bets = [
            _bet("A", "B", "D",       100,  50),
            _bet("A", "B", "under25", 100, -100, "lost"),
        ]
        out_d = pf.compute_team_roi_from_bets(bets, market_filter={"D"}, min_n=1)
        assert out_d["A"]["n"] == 1 and out_d["A"]["profit"] == 50.0


# ─── should_skip_team_pair ───────────────────────────────────────────────────

class TestShouldSkipTeamPair:

    def test_returns_false_when_table_empty_or_none(self):
        assert pf.should_skip_team_pair(None, "A", "B", -25.0) is False
        assert pf.should_skip_team_pair({}, "A", "B", -25.0) is False

    def test_skips_when_team_below_threshold(self):
        table = {"Everton": {"n": 6, "roi": -69.8, "profit": -100, "stake": 100}}
        assert pf.should_skip_team_pair(table, "Everton", "Brighton", -25.0) is True
        assert pf.should_skip_team_pair(table, "Brighton", "Everton", -25.0) is True

    def test_keeps_when_team_above_threshold(self):
        table = {"Arsenal": {"n": 14, "roi": 112.4, "profit": 1, "stake": 1}}
        assert pf.should_skip_team_pair(table, "Arsenal", "Wolves", -25.0) is False

    def test_unseen_teams_pass_through(self):
        # Insufficient sample teams aren't in the table → never blacklisted
        table = {"Everton": {"n": 6, "roi": -69.8, "profit": -1, "stake": 1}}
        assert pf.should_skip_team_pair(table, "NewClub", "OtherClub", -25.0) is False


# ─── kelly_drawdown_throttle ─────────────────────────────────────────────────

class TestKellyDrawdownThrottle:

    def test_no_drawdown_returns_one(self):
        assert pf.kelly_drawdown_throttle(1000, 1000) == 1.0

    def test_above_peak_returns_one(self):
        # If bankroll > peak (peak is stale), no throttle
        assert pf.kelly_drawdown_throttle(1100, 1000) == 1.0

    def test_at_throttle_threshold_returns_min_factor(self):
        # 20% drawdown with default throttle_at_pct=0.20 → min_factor (0.25)
        result = pf.kelly_drawdown_throttle(800, 1000, 0.20, 0.25)
        assert result == pytest.approx(0.25)

    def test_beyond_threshold_clamped_to_min_factor(self):
        # 50% drawdown — well past the 20% point — still clamped to 0.25
        assert pf.kelly_drawdown_throttle(500, 1000, 0.20, 0.25) == 0.25

    def test_linear_interpolation_between(self):
        # 10% drawdown halfway to the 20% throttle → midpoint between 1.0 and 0.25
        # → 1.0 - (1.0-0.25) * (0.10/0.20) = 1.0 - 0.375 = 0.625
        assert pf.kelly_drawdown_throttle(900, 1000, 0.20, 0.25) == pytest.approx(0.625)

    def test_zero_or_negative_bankroll_returns_one(self):
        # Edge cases — guard against div-by-zero
        assert pf.kelly_drawdown_throttle(0, 0) == 1.0
        assert pf.kelly_drawdown_throttle(-100, 1000) == 1.0


# ─── _should_skip_calendar ───────────────────────────────────────────────────

class TestShouldSkipCalendar:

    def test_no_filters_never_skips(self):
        assert pf._should_skip_calendar("2026-05-10", None, None) is False
        assert pf._should_skip_calendar("2026-05-10", set(), set()) is False

    def test_skips_banned_dow(self):
        # 2026-05-11 is a Monday
        assert pf._should_skip_calendar("2026-05-11", {"Mon"}, None) is True
        # 2026-05-12 is a Tuesday → not banned
        assert pf._should_skip_calendar("2026-05-12", {"Mon"}, None) is False

    def test_skips_banned_month(self):
        assert pf._should_skip_calendar("2026-10-04", None, {"Oct"}) is True
        assert pf._should_skip_calendar("2026-09-30", None, {"Oct"}) is False

    def test_combined_filters(self):
        # Friday in October → caught by either filter
        assert pf._should_skip_calendar("2026-10-02", {"Fri"}, {"Oct"}) is True

    def test_invalid_date_does_not_crash(self):
        # Non-parseable strings → False (don't skip on garbage)
        assert pf._should_skip_calendar("not-a-date", {"Mon"}, None) is False


# ─── _is_late_season ─────────────────────────────────────────────────────────

class TestIsLateSeason:
    """The Mar-Apr empirical filter — May is NOT included as of 2026-05-10."""

    def test_march_is_late_season(self):
        assert pf._is_late_season("2026-03-15") is True

    def test_april_is_late_season(self):
        assert pf._is_late_season("2026-04-22") is True

    def test_may_is_NOT_late_season(self):
        # Filter was overly broad pre-2026-05-10 (bundled May with Mar-Apr).
        # Empirical 0/7 sample was Mar-Apr only; May had no data behind it.
        assert pf._is_late_season("2026-05-10") is False
        assert pf._is_late_season("2026-05-31") is False

    def test_pre_run_in_months_safe(self):
        for date in ("2026-01-15", "2026-02-28", "2026-08-10",
                     "2026-09-30", "2026-12-25"):
            assert pf._is_late_season(date) is False, f"{date} flagged as late"


# ─── should_skip_elo_profile ─────────────────────────────────────────────────

class TestShouldSkipEloProfile:
    """ELO-profile filter for Mock Two — proactive (no reactive lag like
    team_roi_filter). Each filter independent; missing data never skips."""

    def test_no_filters_never_skips(self):
        assert pf.should_skip_elo_profile(1500, 1700) is False

    def test_missing_elo_never_skips(self):
        assert pf.should_skip_elo_profile(None, 1700,
                                          min_team_elo=1600) is False
        assert pf.should_skip_elo_profile(1500, None,
                                          min_team_elo=1600) is False

    def test_min_team_elo_skips_weak_team(self):
        # Either side below floor → skip
        assert pf.should_skip_elo_profile(1450, 1700,
                                          min_team_elo=1500) is True
        assert pf.should_skip_elo_profile(1700, 1450,
                                          min_team_elo=1500) is True
        # Both above → keep
        assert pf.should_skip_elo_profile(1600, 1700,
                                          min_team_elo=1500) is False

    def test_max_team_elo_skips_top_teams(self):
        assert pf.should_skip_elo_profile(1850, 1500,
                                          max_team_elo=1800) is True
        assert pf.should_skip_elo_profile(1500, 1850,
                                          max_team_elo=1800) is True
        assert pf.should_skip_elo_profile(1700, 1750,
                                          max_team_elo=1800) is False

    def test_elo_gap_min_skips_close_matches(self):
        # Gap = 50, floor = 100 → skip
        assert pf.should_skip_elo_profile(1700, 1750,
                                          elo_gap_min=100) is True
        # Gap = 200 → keep
        assert pf.should_skip_elo_profile(1700, 1900,
                                          elo_gap_min=100) is False

    def test_elo_gap_max_skips_lopsided_matches(self):
        # Gap = 300, ceiling = 200 → skip
        assert pf.should_skip_elo_profile(1500, 1800,
                                          elo_gap_max=200) is True
        # Gap = 100 → keep
        assert pf.should_skip_elo_profile(1700, 1800,
                                          elo_gap_max=200) is False

    def test_combined_band_filter(self):
        # Sweet spot: gap between [80, 200]
        assert pf.should_skip_elo_profile(1700, 1850,
                                          elo_gap_min=80,
                                          elo_gap_max=200) is False
        # Too close — gap = 30
        assert pf.should_skip_elo_profile(1700, 1730,
                                          elo_gap_min=80,
                                          elo_gap_max=200) is True
        # Too lopsided — gap = 300
        assert pf.should_skip_elo_profile(1500, 1800,
                                          elo_gap_min=80,
                                          elo_gap_max=200) is True

    def test_nan_elo_does_not_skip(self):
        import math
        assert pf.should_skip_elo_profile(math.nan, 1700,
                                          min_team_elo=1600) is False


# ─── should_skip_xg_overperform ────────────────────────────────────────────

class TestShouldSkipXgOverperform:
    """xG-regression filter — skip when a team's actual goals diverge from
    their xG by more than threshold percent (luck driven, due to revert)."""

    def test_none_threshold_never_skips(self):
        assert pf.should_skip_xg_overperform(2.0, 1.5, 1.0, 1.0, None) is False

    def test_zero_threshold_skips_any_divergence(self):
        assert pf.should_skip_xg_overperform(2.0, 1.5, 1.0, 1.0, 0.0) is True

    def test_within_threshold_keeps(self):
        # home: |2.0-1.8|/1.8 = 11% < 25% threshold → keep
        assert pf.should_skip_xg_overperform(2.0, 1.8, 1.0, 1.0, 0.25) is False

    def test_overperforming_team_skipped(self):
        # home: |2.0-1.0|/1.0 = 100% > 25% → skip
        assert pf.should_skip_xg_overperform(2.0, 1.0, 1.0, 1.0, 0.25) is True

    def test_underperforming_also_skipped(self):
        # away: |0.5-1.5|/1.5 = 67% > 25% → skip
        assert pf.should_skip_xg_overperform(1.5, 1.5, 0.5, 1.5, 0.25) is True

    def test_missing_data_does_not_skip(self):
        # None or NaN xG → can't compute → don't skip
        assert pf.should_skip_xg_overperform(2.0, None, 1.0, 1.0, 0.25) is False
        import math
        assert pf.should_skip_xg_overperform(2.0, math.nan, 1.0, 1.0, 0.25) is False

    def test_near_zero_xg_does_not_skip(self):
        # xG = 0.02 too small to compute a sensible ratio → don't skip
        assert pf.should_skip_xg_overperform(2.0, 0.02, 1.0, 1.0, 0.25) is False
