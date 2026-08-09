"""Live odds must key on the same team names the fixtures use.

The Odds API says "Coventry City" and "Hull City". `fetch_upcoming_fixtures`
resolves those to football-data's short names, "Coventry" and "Hull", via
`data._resolve_team_name`. `fetch_live_odds` did not resolve anything, so its
keys stayed long-form and the auto-bet loop's `live_odds_map.get((home, away))`
missed — on 2026-08-09, for exactly the two promoted fixtures and nothing else.

It is masked right now because those two fixtures are blocked by the no-history
gate anyway. It stops being masked around October, when Coventry and Hull clear
six Premier League matches and become bettable: they would simply never appear,
with no error and no log line. That is the same silent-drop shape as the bare
`continue` that hid two of ten opening fixtures.

Resolution has to happen inside `fetch_live_odds`, because the cache on disk is
keyed by these names too — resolving at the call site would leave a cache full
of unjoinable keys.
"""
from __future__ import annotations

import pytest

import data as fpred_data
import portfolio as pf


@pytest.fixture(autouse=True)
def _known_teams():
    """`_resolve_team_name` matches against the names in the loaded CSVs, which
    `load_data` populates. Seed it directly so the test needs no network."""
    original = set(fpred_data.KNOWN_FD_TEAMS)
    fpred_data.KNOWN_FD_TEAMS.update(
        {"Arsenal", "Coventry", "Hull", "Man United", "Everton"})
    yield
    fpred_data.KNOWN_FD_TEAMS.clear()
    fpred_data.KNOWN_FD_TEAMS.update(original)


def test_the_resolver_shortens_the_promoted_names():
    """Guards the assumption the join depends on."""
    assert fpred_data._resolve_team_name("Coventry City") == "Coventry"
    assert fpred_data._resolve_team_name("Hull City") == "Hull"


def test_odds_keys_use_resolved_names():
    raw = {("Arsenal", "Coventry City"): {"H": 1.2, "D": 9.2, "A": 15.0},
           ("Hull City", "Man United"):  {"H": 6.0, "D": 4.8, "A": 1.6}}

    resolved = pf.resolve_odds_map_names(raw)

    assert ("Arsenal", "Coventry") in resolved
    assert ("Hull", "Man United") in resolved
    assert ("Arsenal", "Coventry City") not in resolved


def test_already_short_names_are_left_alone():
    raw = {("Everton", "Arsenal"): {"H": 2.0, "D": 3.4, "A": 3.9}}

    resolved = pf.resolve_odds_map_names(raw)

    assert ("Everton", "Arsenal") in resolved
    assert resolved[("Everton", "Arsenal")]["D"] == 3.4


def test_the_quote_survives_resolution():
    raw = {("Arsenal", "Coventry City"): {"H": 1.2, "D": 9.2, "A": 15.0,
                                          "_pinnacle": {"D": 7.83}}}

    resolved = pf.resolve_odds_map_names(raw)

    assert resolved[("Arsenal", "Coventry")]["D"] == 9.2
    assert resolved[("Arsenal", "Coventry")]["_pinnacle"]["D"] == 7.83


def test_an_unresolvable_name_is_kept_rather_than_dropped():
    """A team the resolver cannot place is still better joined on its raw name
    than silently discarded."""
    raw = {("Some New Club FC", "Arsenal"): {"H": 2.0, "D": 3.4, "A": 3.9}}

    resolved = pf.resolve_odds_map_names(raw)

    assert len(resolved) == 1
