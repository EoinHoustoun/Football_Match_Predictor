"""Tests for the 2026-06-09 off-season hardening session:

- dynamic season config (auto-rollover each July, no hardcoded season strings)
- fuzzy team-name resolution (promoted teams resolve without map edits)
- ODDS_API_KEY environment-variable precedence
- explicit NaN guards in both LIVE auto-bet paths (parity with the backtest
  simulators, which already had them)
"""
from datetime import date

import pytest


# ── Dynamic season configuration ─────────────────────────────────────────────

def test_season_start_year_rolls_over_in_july():
    from data import _season_start_year
    assert _season_start_year(date(2026, 6, 9)) == 2025   # off-season → still 2025-26
    assert _season_start_year(date(2026, 7, 1)) == 2026   # July → new season
    assert _season_start_year(date(2027, 1, 15)) == 2026  # mid-season → 2026-27


def test_build_seasons_extends_automatically():
    from data import _build_seasons
    s_now = _build_seasons(date(2026, 6, 9))
    assert list(s_now)[0] == "2021-22"
    assert list(s_now)[-1] == "2025-26"

    s_aug = _build_seasons(date(2026, 8, 5))
    assert list(s_aug)[-1] == "2026-27"
    assert s_aug["2026-27"].endswith("/2627/E0.csv")
    assert s_aug["2021-22"].endswith("/2122/E0.csv")


# ── Fuzzy team-name resolution ───────────────────────────────────────────────

def test_resolve_team_name_falls_back_to_fuzzy():
    import data
    saved = set(data.KNOWN_FD_TEAMS)
    try:
        data.KNOWN_FD_TEAMS.update({"Leeds", "Burnley", "Nott'm Forest"})
        # substring containment (typical promoted-team shapes)
        assert data._resolve_team_name("Leeds United") == "Leeds"
        assert data._resolve_team_name("Burnley FC") == "Burnley"
        # hardcoded map still wins when it has the answer
        assert data._resolve_team_name("Manchester City") == "Man City"
        # genuinely unknown names pass through unchanged
        assert data._resolve_team_name("Real Madrid") == "Real Madrid"
    finally:
        data.KNOWN_FD_TEAMS.clear()
        data.KNOWN_FD_TEAMS.update(saved)


# ── ODDS_API_KEY env-var precedence ──────────────────────────────────────────

def test_odds_api_key_env_var_takes_precedence(monkeypatch):
    from portfolio import resolve_odds_api_key
    monkeypatch.setenv("ODDS_API_KEY", "env-key")
    assert resolve_odds_api_key("settings-key") == "env-key"
    monkeypatch.delenv("ODDS_API_KEY")
    assert resolve_odds_api_key("settings-key") == "settings-key"
    assert resolve_odds_api_key("") == ""


# ── NaN guards in the live auto-bet paths ────────────────────────────────────

def _port() -> dict:
    return {
        "initial_bankroll": 10_000.0, "bankroll": 10_000.0,
        "bets": [],
        "settings": {
            "kelly_fraction": 0.5, "max_stake_pct": 0.10,
            "auto_markets": ["D"], "min_prob": 0.20,
            "use_calibrated_probs": False,
        },
    }


def _nan_prob_candidate() -> list[dict]:
    # NaN model prob + valid detect_odds → EV = NaN. Without an explicit
    # guard, NaN < threshold is False and the EV/prob gates both pass.
    return [{
        "home": "A", "away": "B", "date": "2026-05-01",
        "market": "D", "selection": "Draw",
        "model_prob": float("nan"), "odds": 3.5, "ev": 0.2,
        "detect_odds": 3.6,
    }]


def test_live_auto_bet_rejects_nan_prob():
    from portfolio import auto_place_value_bets
    port = _port()
    placed = auto_place_value_bets(port, _nan_prob_candidate(), threshold=0.05)
    assert placed == []
    assert port["bets"] == []


def test_live_auto_bet_v2_rejects_nan_prob():
    from portfolio import auto_place_value_bets_v2
    port = _port()
    placed = auto_place_value_bets_v2(port, _nan_prob_candidate(), threshold=0.05)
    assert placed == []
    assert port["bets"] == []


# ── Calibrated draw-probability cap (small-sample isotonic guard) ────────────

class _FakeIso:
    """Stands in for a fitted IsotonicRegression with a fixed output —
    reproduces the observed 91-match plateau that mapped raw ≥0.40 → 0.667."""
    def __init__(self, value: float):
        self.value = value

    def predict(self, xs):
        return [self.value for _ in xs]


def test_calibrated_draw_prob_is_capped():
    import portfolio as pf
    cal = {"D": _FakeIso(0.667), "H": _FakeIso(0.667)}
    assert pf.calibrate_prob(0.40, "D", cal) == pytest.approx(0.45)
    # Home market is NOT capped — true H probs above 0.45 are legitimate
    assert pf.calibrate_prob(0.40, "H", cal) == pytest.approx(0.667)
    # Sane calibrator outputs pass through untouched
    cal_ok = {"D": _FakeIso(0.31)}
    assert pf.calibrate_prob(0.28, "D", cal_ok) == pytest.approx(0.31)


def test_draw_prob_cap_can_be_disabled_for_research():
    import portfolio as pf
    cal = {"D": _FakeIso(0.667)}
    saved = pf.DRAW_PROB_CAP
    try:
        pf.DRAW_PROB_CAP = None
        assert pf.calibrate_prob(0.40, "D", cal) == pytest.approx(0.667)
    finally:
        pf.DRAW_PROB_CAP = saved


def test_live_auto_bet_still_places_valid_candidate():
    """Guard must not over-reject: a clean +EV candidate still goes through."""
    from portfolio import auto_place_value_bets
    port = _port()
    cand = [{
        "home": "A", "away": "B", "date": "2026-05-01",
        "market": "D", "selection": "Draw",
        "model_prob": 0.35, "odds": 3.4, "ev": 0.35 * 3.4 - 1,
    }]
    placed = auto_place_value_bets(port, cand, threshold=0.05)
    assert len(placed) == 1
