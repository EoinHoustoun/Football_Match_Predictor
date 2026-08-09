"""Promoted-team safety: don't bet on sides the model has never seen.

2026-27 promoted Coventry and Hull, neither of which appears anywhere in the
2021-26 dataset. They have no Dixon-Coles ratings and no Elo at all — the Elo
dict simply has no key for them. Two holes let that through:

  1. The auto-bet loop dropped unknown teams with a bare `continue`, leaving no
     trace that two of the ten opening fixtures were invisible.
  2. `should_skip_elo_profile` waves a fixture through when Elo is missing, on a
     "don't filter on missing data" principle that is exactly backwards for a
     floor whose job is to exclude weak and unrated sides.

The £100k sweep artifact was driven by promoted-side (Sunderland) draws, so
this is the failure mode with form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio as pf
from data import team_match_counts, team_rated_from


# ── Counting top-flight history ───────────────────────────────────────────────

def test_team_match_counts_counts_home_and_away_appearances():
    df = pd.DataFrame({
        "HomeTeam": ["Arsenal", "Everton", "Arsenal"],
        "AwayTeam": ["Everton", "Arsenal", "Fulham"],
    })
    counts = team_match_counts(df)
    assert counts["Arsenal"] == 3
    assert counts["Everton"] == 2
    assert counts["Fulham"] == 1


def test_team_match_counts_omits_a_team_that_never_appears():
    df = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Everton"]})
    assert "Coventry" not in team_match_counts(df)


def test_team_match_counts_survives_an_empty_frame():
    assert team_match_counts(pd.DataFrame(columns=["HomeTeam", "AwayTeam"])) == {}


# ── The no-history predicate ──────────────────────────────────────────────────

COUNTS = {"Arsenal": 190, "Ipswich": 38, "Sunderland": 38, "Newbie": 3}


def test_unrated_team_is_skipped():
    # Coventry is absent from the counts entirely.
    assert pf.should_skip_unrated("Arsenal", "Coventry", COUNTS, 6) is True


def test_team_below_the_threshold_is_skipped():
    assert pf.should_skip_unrated("Newbie", "Arsenal", COUNTS, 6) is True


def test_team_exactly_at_the_threshold_passes():
    assert pf.should_skip_unrated("Arsenal", "Sixer", {**COUNTS, "Sixer": 6}, 6) is False


def test_two_established_teams_pass():
    assert pf.should_skip_unrated("Arsenal", "Ipswich", COUNTS, 6) is False


def test_gate_is_inert_when_the_threshold_is_unset():
    assert pf.should_skip_unrated("Arsenal", "Coventry", COUNTS, None) is False
    assert pf.should_skip_unrated("Arsenal", "Coventry", COUNTS, 0) is False


def test_gate_does_not_filter_when_no_counts_were_supplied():
    """Matches should_skip_elo_profile's stance: never filter on absent data.

    The four call sites are covered separately to prove they always pass counts.
    """
    assert pf.should_skip_unrated("Arsenal", "Coventry", None, 6) is False


# ── Point-in-time history, for honest backtests ───────────────────────────────
#
# A backtest gated on final match counts would treat Ipswich as rated during
# its own first six matches, which is look-ahead. The simulators need to know
# when each team *became* rated, not whether it ever did.

def _season_frame() -> pd.DataFrame:
    """Arsenal plays throughout; Newbie arrives partway and plays 4 matches."""
    rows = [("2026-01-0%d" % d, "Arsenal", "Everton") for d in range(1, 6)]
    rows += [("2026-02-0%d" % d, "Newbie", "Arsenal") for d in range(1, 5)]
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam"]).assign(
        Date=lambda d: pd.to_datetime(d["Date"]))


def test_team_rated_from_gives_the_date_the_threshold_was_reached():
    rated = team_rated_from(_season_frame(), min_matches=3)
    # Arsenal's 3rd match is 3 January.
    assert rated["Arsenal"] == pd.Timestamp("2026-01-03")
    # Newbie's 3rd is 3 February.
    assert rated["Newbie"] == pd.Timestamp("2026-02-03")


def test_team_rated_from_omits_a_team_that_never_reaches_the_threshold():
    rated = team_rated_from(_season_frame(), min_matches=6)
    assert "Newbie" not in rated       # only ever plays 4
    assert "Arsenal" in rated


def test_point_in_time_gate_blocks_before_the_threshold_date():
    rated = team_rated_from(_season_frame(), min_matches=3)
    assert pf.should_skip_unrated_at(
        "Newbie", "Arsenal", pd.Timestamp("2026-02-02"), rated, 3) is True


def test_point_in_time_gate_allows_after_the_threshold_date():
    rated = team_rated_from(_season_frame(), min_matches=3)
    assert pf.should_skip_unrated_at(
        "Newbie", "Arsenal", pd.Timestamp("2026-02-04"), rated, 3) is False


def test_point_in_time_gate_blocks_on_the_threshold_date_itself():
    """The qualifying match hasn't been played when the fixture kicks off."""
    rated = team_rated_from(_season_frame(), min_matches=3)
    assert pf.should_skip_unrated_at(
        "Newbie", "Arsenal", pd.Timestamp("2026-02-03"), rated, 3) is True


def test_point_in_time_gate_blocks_a_team_never_seen():
    rated = team_rated_from(_season_frame(), min_matches=3)
    assert pf.should_skip_unrated_at(
        "Coventry", "Arsenal", pd.Timestamp("2026-06-01"), rated, 3) is True


def test_point_in_time_gate_accepts_a_plain_date_or_string():
    rated = team_rated_from(_season_frame(), min_matches=3)
    for when in ("2026-02-04", pd.Timestamp("2026-02-04").date()):
        assert pf.should_skip_unrated_at("Newbie", "Arsenal", when, rated, 3) is False


def test_point_in_time_gate_is_inert_without_a_threshold_or_table():
    rated = team_rated_from(_season_frame(), min_matches=3)
    assert pf.should_skip_unrated_at("Coventry", "Arsenal", "2026-06-01", rated, None) is False
    assert pf.should_skip_unrated_at("Coventry", "Arsenal", "2026-06-01", None, 3) is False


# ── Elo floor no longer waves missing ratings through ─────────────────────────

def test_missing_elo_is_skipped_when_the_rating_is_required():
    assert pf.should_skip_elo_profile(None, 1600, 1500, require_known_elo=True) is True
    assert pf.should_skip_elo_profile(1600, None, 1500, require_known_elo=True) is True


def test_nan_elo_is_skipped_when_the_rating_is_required():
    assert pf.should_skip_elo_profile(np.nan, 1600, 1500, require_known_elo=True) is True


def test_requiring_a_rating_does_not_disturb_a_known_one():
    assert pf.should_skip_elo_profile(1600, 1700, 1500, require_known_elo=True) is False
    assert pf.should_skip_elo_profile(1400, 1700, 1500, require_known_elo=True) is True


def test_requiring_a_rating_is_inert_when_no_band_is_set():
    """Nothing to enforce, so a missing rating is not grounds to skip."""
    assert pf.should_skip_elo_profile(None, None, None, require_known_elo=True) is False


def test_default_behaviour_is_unchanged():
    """Existing callers that don't opt in keep the old permissive stance."""
    assert pf.should_skip_elo_profile(None, 1600, 1500) is False


# ── The live auto-bet paths ───────────────────────────────────────────────────

def _port(**settings) -> dict:
    base = {
        "kelly_fraction": 0.5, "max_stake_pct": 0.10,
        "auto_markets": ["D"], "min_prob": 0.22,
        "use_calibrated_probs": False, "min_team_matches": 6,
    }
    base.update(settings)
    return {"initial_bankroll": 10_000.0, "bankroll": 10_000.0,
            "bets": [], "settings": base}


def _candidate(home="Arsenal", away="Coventry") -> list[dict]:
    return [{
        "home": home, "away": away, "date": "2026-08-21",
        "market": "D", "selection": "Draw",
        "model_prob": 0.35, "odds": 3.2, "ev": 0.35 * 3.2 - 1,
    }]


def test_main_auto_bet_blocks_an_unrated_team():
    placed = pf.auto_place_value_bets(
        _port(), _candidate(), threshold=0.05, match_counts=COUNTS)
    assert placed == []


def test_main_auto_bet_still_places_on_two_established_teams():
    placed = pf.auto_place_value_bets(
        _port(), _candidate(away="Ipswich"), threshold=0.05, match_counts=COUNTS)
    assert len(placed) == 1


def test_mock_two_auto_bet_blocks_an_unrated_team():
    placed = pf.auto_place_value_bets_v2(
        _port(), _candidate(), threshold=0.05, match_counts=COUNTS)
    assert placed == []


def test_mock_two_auto_bet_still_places_on_two_established_teams():
    placed = pf.auto_place_value_bets_v2(
        _port(), _candidate(away="Ipswich"), threshold=0.05, match_counts=COUNTS)
    assert len(placed) == 1


# ── Skips are recorded, not silent ────────────────────────────────────────────

def test_main_auto_bet_records_why_it_skipped():
    log: list[dict] = []
    pf.auto_place_value_bets(_port(), _candidate(), threshold=0.05,
                             match_counts=COUNTS, skip_log=log)
    assert len(log) == 1
    entry = log[0]
    assert entry["reason"] == "no_history"
    assert entry["home"] == "Arsenal" and entry["away"] == "Coventry"
    assert "Coventry" in entry["detail"]


def test_mock_two_auto_bet_records_why_it_skipped():
    log: list[dict] = []
    pf.auto_place_value_bets_v2(_port(), _candidate(), threshold=0.05,
                                match_counts=COUNTS, skip_log=log)
    assert [e["reason"] for e in log] == ["no_history"]


def test_nothing_is_logged_when_a_bet_goes_through():
    log: list[dict] = []
    pf.auto_place_value_bets(_port(), _candidate(away="Ipswich"), threshold=0.05,
                             match_counts=COUNTS, skip_log=log)
    assert log == []


def test_the_gate_is_off_when_no_counts_reach_the_live_path():
    """Back-compat: existing callers that pass no counts behave as before."""
    placed = pf.auto_place_value_bets(_port(), _candidate(), threshold=0.05)
    assert len(placed) == 1


# ── The simulators, end to end ────────────────────────────────────────────────
#
# Predicate tests prove the logic; these prove it is actually wired in, which
# is the failure mode that matters. A gate that exists but is never called
# looks identical in a summary to a gate that found nothing.

def _sim_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two established sides and one newcomer, all with priceable draws."""
    dates = pd.date_range("2026-01-03", periods=12, freq="7D")
    rows, bt_rows = [], []
    for i, d in enumerate(dates):
        home, away = ("Arsenal", "Everton") if i % 2 == 0 else ("Newbie", "Arsenal")
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


@pytest.mark.parametrize("simulate", [
    pf.ev_backtest_simulate,
    pf.ev_backtest_simulate_v2,
])
def test_simulator_skips_fixtures_involving_an_unrated_side(simulate):
    bt_df, df = _sim_inputs()
    _, summary = simulate(bt_df, df, min_ev_pct=1.0, min_prob=0.30,
                          allowed_markets={"D"}, min_team_matches=4)
    assert summary.get("skipped_no_history", 0) > 0


@pytest.mark.parametrize("simulate", [
    pf.ev_backtest_simulate,
    pf.ev_backtest_simulate_v2,
])
def test_simulator_skips_nothing_when_the_gate_is_off(simulate):
    bt_df, df = _sim_inputs()
    _, summary = simulate(bt_df, df, min_ev_pct=1.0, min_prob=0.30,
                          allowed_markets={"D"})
    assert summary.get("skipped_no_history", 0) == 0
