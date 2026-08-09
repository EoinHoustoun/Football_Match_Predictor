"""The season simulator must start from the new season, not the old one.

`get_current_table(df)` builds the table from loaded match data. Pre-season that
data still ends in May, because football-data does not publish a season's file
until it starts. So on 2026-08-09 the Season Outlook tab was being handed:

  - a table of the 2025-26 twenty, every club on "Played 38", Arsenal on 85
    points, including Burnley, West Ham and Wolves who have been relegated
  - and 380 remaining fixtures for 2026-27, featuring Hull and Coventry, who
    are not in that table at all

Simulating one on top of the other projects a second full season onto finished
standings, leaves three relegated clubs sitting in the table with no fixtures,
and omits the promoted sides entirely.

Pre-season the correct starting table is simply the new season's clubs on zero.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data import preseason_table, table_is_stale_for


FIXTURES = [
    {"home": "Arsenal", "away": "Coventry"},
    {"home": "Hull", "away": "Man United"},
    {"home": "Coventry", "away": "Hull"},
    {"home": "Man United", "away": "Arsenal"},
]


def _finished_table() -> pd.DataFrame:
    """Last season's final standings, of a different set of clubs."""
    return pd.DataFrame({
        "Team": ["Arsenal", "Man United", "Burnley", "Wolves"],
        "Played": [38, 38, 38, 38], "W": [26, 20, 8, 7], "D": [7, 11, 9, 10],
        "L": [5, 7, 21, 21], "GF": [71, 69, 34, 33], "GA": [27, 50, 70, 72],
        "GD": [44, 19, -36, -39], "Pts": [85, 71, 33, 31],
    })


# ── Detecting the situation ───────────────────────────────────────────────────

def test_a_finished_table_is_stale_against_next_seasons_fixtures():
    assert table_is_stale_for(_finished_table(), FIXTURES) is True


def test_a_table_in_progress_is_not_stale():
    """Once the new season starts, the real table is the right starting point."""
    live = pd.DataFrame({
        "Team": ["Arsenal", "Coventry", "Hull", "Man United"],
        "Played": [3, 3, 3, 3], "W": [2, 0, 1, 1], "D": [1, 1, 1, 1],
        "L": [0, 2, 1, 1], "GF": [7, 2, 4, 5], "GA": [2, 6, 4, 4],
        "GD": [5, -4, 0, 1], "Pts": [7, 1, 4, 4],
    })
    assert table_is_stale_for(live, FIXTURES) is False


def test_an_empty_table_is_stale():
    assert table_is_stale_for(pd.DataFrame(), FIXTURES) is True


def test_no_fixtures_means_nothing_to_judge_against():
    assert table_is_stale_for(_finished_table(), []) is False


# ── Building the replacement ──────────────────────────────────────────────────

def test_preseason_table_uses_the_clubs_in_the_fixtures():
    table = preseason_table(FIXTURES)
    assert set(table["Team"]) == {"Arsenal", "Coventry", "Hull", "Man United"}


def test_relegated_clubs_do_not_survive_into_it():
    table = preseason_table(FIXTURES)
    assert "Burnley" not in set(table["Team"])
    assert "Wolves" not in set(table["Team"])


def test_everyone_starts_on_nothing():
    table = preseason_table(FIXTURES)
    for column in ("Played", "W", "D", "L", "GF", "GA", "GD", "Pts"):
        assert (table[column] == 0).all(), column


def test_it_carries_the_columns_the_simulator_expects():
    expected = {"Team", "Played", "W", "D", "L", "GF", "GA", "GD", "Pts"}
    assert expected.issubset(set(preseason_table(FIXTURES).columns))


def test_no_fixtures_gives_an_empty_table_rather_than_an_error():
    assert preseason_table([]).empty
