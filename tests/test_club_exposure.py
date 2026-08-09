"""Cap how much of a season can ride on one club.

The 2026-27 re-validation found the deployed draws-only config is far more
concentrated than its headline suggests. In 2025-26 it placed 14 bets, 11 of
them on Sunderland, and Sunderland returned +£66,577 against a season total of
+£51,222 — so every other club combined lost £15k. The other seasons are milder
but the same shape: 43-50% of a season's bets on a single club.

That is not an edge in Premier League draws, it is one club having a strange
season. The no-history gate does not help: it only covers a promoted side's
first six matches, after which Sunderland became "rated" and the run continued.

This cap is a risk control, not a profit filter. The value is chosen a priori
(no single club should carry more than about a third of a season's bets), NOT
by picking whichever number backtests best. Choosing it on profit would be
selecting on the same four seasons the problem was found in.
"""
from __future__ import annotations

import pandas as pd
import pytest

import portfolio as pf


# ── The predicate ─────────────────────────────────────────────────────────────

def test_no_cap_configured_allows_everything():
    counts = {"Sunderland": 99}
    assert pf.should_skip_club_exposure("Sunderland", "Arsenal", counts, None) is False


def test_club_at_the_cap_is_skipped():
    counts = {"Sunderland": 5}
    assert pf.should_skip_club_exposure("Sunderland", "Arsenal", counts, 5) is True


def test_club_below_the_cap_passes():
    counts = {"Sunderland": 4}
    assert pf.should_skip_club_exposure("Sunderland", "Arsenal", counts, 5) is False


def test_either_side_hitting_the_cap_blocks_the_fixture():
    """A fixture exposes both clubs, so either one at the cap is enough."""
    counts = {"Arsenal": 5}
    assert pf.should_skip_club_exposure("Sunderland", "Arsenal", counts, 5) is True


def test_a_club_with_no_bets_yet_passes():
    assert pf.should_skip_club_exposure("Hull", "Arsenal", {}, 5) is False


# ── Counting what is already at risk ──────────────────────────────────────────

def test_club_bet_counts_counts_both_teams_in_a_bet():
    bets = [
        {"home": "Sunderland", "away": "Arsenal", "status": "won"},
        {"home": "Fulham", "away": "Sunderland", "status": "pending"},
    ]
    counts = pf.club_bet_counts(bets)
    assert counts["Sunderland"] == 2
    assert counts["Arsenal"] == 1
    assert counts["Fulham"] == 1


def test_club_bet_counts_ignores_voided_bets():
    """A void never carried risk, so it should not use up the allowance."""
    bets = [{"home": "Sunderland", "away": "Arsenal", "status": "void"}]
    assert pf.club_bet_counts(bets) == {}


def test_club_bet_counts_can_be_limited_to_one_season():
    bets = [
        {"home": "Sunderland", "away": "Arsenal", "status": "won",
         "date": "2025-09-13"},
        {"home": "Sunderland", "away": "Chelsea", "status": "won",
         "date": "2026-08-22"},
    ]
    counts = pf.club_bet_counts(bets, since="2026-07-01")
    assert counts["Sunderland"] == 1
    assert "Arsenal" not in counts


# ── The live path honours it ──────────────────────────────────────────────────

def _portfolio(**settings) -> dict:
    p = {
        "season": "2026-27",
        "initial_bankroll": 10_000.0,
        "bankroll": 10_000.0,
        "bets": [],
        "settings": {
            "auto_bet_enabled": True,
            "auto_bet_threshold": 0.40,
            "auto_markets": ["D"],
            "min_prob": 0.21,
            "min_ev": 0.40,
            "kelly_fraction": 1.0,
            "max_stake_pct": 0.25,
            "use_calibrated_probs": False,
            "use_simultaneous_kelly": False,
            "skip_late_season": False,
            "min_team_matches": None,
            "max_bets_per_club": 5,
        },
    }
    p["settings"].update(settings)
    return p


def _candidate(home: str, away: str, date: str = "2026-09-12") -> dict:
    return {
        "home": home, "away": away, "date": date,
        "market": "D", "selection": "Draw",
        "odds": 4.10, "detect_odds": 4.10,
        "model_prob": 0.40, "ev": 0.64,
        "home_elo": 1600, "away_elo": 1600,
    }


def _settled(home: str, away: str) -> dict:
    return {"home": home, "away": away, "market": "D", "status": "won",
            "date": "2026-08-22", "stake": 100.0, "odds": 4.0, "profit": 300.0}


def test_live_path_blocks_a_club_already_at_its_cap():
    p = _portfolio(max_bets_per_club=3)
    p["bets"] = [_settled("Sunderland", "Arsenal"),
                 _settled("Fulham", "Sunderland"),
                 _settled("Sunderland", "Chelsea")]

    placed = pf.auto_place_value_bets(p, [_candidate("Sunderland", "Everton")], 0.40)

    assert placed == []


def test_live_path_still_places_for_a_club_under_its_cap():
    p = _portfolio(max_bets_per_club=3)
    p["bets"] = [_settled("Sunderland", "Arsenal")]

    placed = pf.auto_place_value_bets(p, [_candidate("Sunderland", "Everton")], 0.40)

    assert len(placed) == 1


def test_live_path_records_why_the_club_was_skipped():
    p = _portfolio(max_bets_per_club=1)
    p["bets"] = [_settled("Sunderland", "Arsenal")]
    skip_log: list[dict] = []

    pf.auto_place_value_bets(p, [_candidate("Sunderland", "Everton")], 0.40,
                             skip_log=skip_log)

    assert any(e.get("reason") == "club_exposure" for e in skip_log)


def test_cap_off_by_default_leaves_behaviour_unchanged():
    """Existing portfolios have no cap key, and must behave exactly as before."""
    p = _portfolio()
    del p["settings"]["max_bets_per_club"]
    p["bets"] = [_settled("Sunderland", "Arsenal") for _ in range(9)]

    placed = pf.auto_place_value_bets(p, [_candidate("Sunderland", "Everton")], 0.40)

    assert len(placed) == 1


# ── Both simulators honour it ─────────────────────────────────────────────────
# Parametrised because Main and Mock Two are separate implementations that have
# drifted before. A cap wired into one and not the other reports an identical
# summary either way, so it fails silently — which is exactly what happened on
# the first pass at this: v2 counted the skips but never checked the cap.

def _sim_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twelve priceable draws, all of them involving Sunderland."""
    dates = pd.date_range("2026-01-03", periods=12, freq="7D")
    rows, bt_rows = [], []
    for i, d in enumerate(dates):
        home, away = ("Sunderland", "Everton") if i % 2 == 0 \
            else ("Arsenal", "Sunderland")
        rows.append({
            "Date": d, "HomeTeam": home, "AwayTeam": away,
            "FTHG": 1, "FTAG": 1,
            "B365H": 2.6, "B365D": 3.4, "B365A": 2.9,
        })
        bt_rows.append({
            "Date": d, "Home": home, "Away": away,
            "DC_H": 33.0, "DC_D": 40.0, "DC_A": 27.0, "Actual": "D",
        })
    return pd.DataFrame(bt_rows), pd.DataFrame(rows)


SIMULATORS = [pf.ev_backtest_simulate, pf.ev_backtest_simulate_v2]


@pytest.mark.parametrize("simulate", SIMULATORS)
def test_simulator_stops_at_the_club_cap(simulate):
    bt_df, df = _sim_inputs()
    log, summary = simulate(bt_df, df, min_ev_pct=1.0, min_prob=0.30,
                            allowed_markets={"D"}, max_bets_per_club=4)

    assert len(log) == 4
    assert summary.get("skipped_club_exposure", 0) > 0


@pytest.mark.parametrize("simulate", SIMULATORS)
def test_simulator_is_unrestricted_with_no_cap(simulate):
    bt_df, df = _sim_inputs()
    log, summary = simulate(bt_df, df, min_ev_pct=1.0, min_prob=0.30,
                            allowed_markets={"D"})

    assert len(log) == 12
    assert summary.get("skipped_club_exposure", 0) == 0
