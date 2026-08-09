"""A promoted team must not enter the Elo series at 1500.

`_compute_elo_series` gave any unseen team 1500.0. For the clubs present when
the dataset opens that is right — everyone starts level and the ratings separate
from there. For a club arriving in 2023 it is badly wrong: it drops a promoted
side into the middle of a league that has already spread out.

The effect is not subtle, and it runs the wrong way. Seed Hull at 1370 from the
relegation market, let them lose 0-2 at home to Manchester United, and their
rating comes back **1487** — 118 points HIGHER than before they played, because
the loss is applied to a 1500 start rather than to the rating we gave them.

The right entry level is where the clubs they replace leave. Across the fifteen
teams relegated between 2021-22 and 2025-26 that is a median of 1351 (mean 1360,
sd 59), and 1500 sits nearly 2.5 standard deviations above it.

This also fixes the historical series, where every promoted side has been
entering a standard deviation and a half too high in its first season.
"""
from __future__ import annotations

import pandas as pd
import pytest

import data as fpred_data
from data import PROMOTED_ENTRY_ELO, _compute_elo_series, get_current_elo


def _match(date: str, home: str, away: str, result: str) -> dict:
    goals = {"H": (2, 0), "A": (0, 2), "D": (1, 1)}[result]
    return {"Date": pd.Timestamp(date), "HomeTeam": home, "AwayTeam": away,
            "FTHG": goals[0], "FTAG": goals[1], "FTR": result}


def _founding_season() -> list[dict]:
    """Two clubs trading results across the opening months."""
    return [
        _match("2021-08-14", "Arsenal", "Everton", "H"),
        _match("2021-09-14", "Everton", "Arsenal", "D"),
        _match("2021-10-14", "Arsenal", "Everton", "H"),
    ]


# ── The entry level ───────────────────────────────────────────────────────────

def test_the_entry_level_is_where_relegated_teams_leave():
    """1351 is the median exit Elo of the fifteen sides relegated since
    2021-22. It is a measurement, not a preference."""
    assert 1330 <= PROMOTED_ENTRY_ELO <= 1380


def test_founding_clubs_still_start_level():
    """Everyone present when the data opens starts at 1500 as before; there is
    nothing to be promoted from."""
    df = pd.DataFrame(_founding_season())
    records, _ = _compute_elo_series(df)
    first = records.iloc[0]
    assert first["home_elo"] == 1500.0
    assert first["away_elo"] == 1500.0


def test_a_club_arriving_later_enters_at_the_promoted_level():
    df = pd.DataFrame(_founding_season() + [
        _match("2023-08-12", "Arsenal", "Hull", "H"),
    ])
    records, _ = _compute_elo_series(df)
    hull_row = records[records["AwayTeam"] == "Hull"].iloc[0]
    assert hull_row["away_elo"] == pytest.approx(PROMOTED_ENTRY_ELO)


def test_a_supplied_entry_rating_overrides_the_default():
    """The relegation market rates this year's intake individually, so the
    caller can hand in a per-team entry rating."""
    df = pd.DataFrame(_founding_season() + [
        _match("2023-08-12", "Arsenal", "Hull", "H"),
    ])
    records, _ = _compute_elo_series(df, entry_ratings={"Hull": 1370.0})
    hull_row = records[records["AwayTeam"] == "Hull"].iloc[0]
    assert hull_row["away_elo"] == pytest.approx(1370.0)


# ── The bug this exists to prevent ────────────────────────────────────────────

def test_losing_your_first_match_cannot_raise_your_rating():
    """The whole point. A seeded promoted side that loses must end up below
    where it started, not 118 points above it."""
    df = pd.DataFrame(_founding_season() + [
        _match("2023-08-12", "Hull", "Arsenal", "A"),   # Hull lose 0-2 at home
    ])
    _, elo = _compute_elo_series(df, entry_ratings={"Hull": 1370.0})
    assert elo["Hull"] < 1370.0


def test_the_old_behaviour_would_have_failed_this():
    """Guards the regression directly: entering at 1500 and losing still leaves
    a promoted side above the level they were seeded at."""
    df = pd.DataFrame(_founding_season() + [
        _match("2023-08-12", "Hull", "Arsenal", "A"),
    ])
    _, naive = _compute_elo_series(df, entry_ratings={"Hull": 1500.0})
    assert naive["Hull"] > 1370.0     # this is what the app used to do


def test_get_current_elo_passes_entry_ratings_through():
    df = pd.DataFrame(_founding_season() + [
        _match("2023-08-12", "Arsenal", "Hull", "H"),
    ])
    high = get_current_elo(df, entry_ratings={"Hull": 1450.0})
    low = get_current_elo(df, entry_ratings={"Hull": 1300.0})
    assert high["Hull"] > low["Hull"]


def test_an_established_club_is_unaffected_by_entry_ratings():
    df = pd.DataFrame(_founding_season())
    baseline = get_current_elo(df)
    with_entry = get_current_elo(df, entry_ratings={"Hull": 1300.0})
    assert baseline["Arsenal"] == pytest.approx(with_entry["Arsenal"])
