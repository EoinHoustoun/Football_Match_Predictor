"""ESPN started answering date-range scoreboard queries (`dates=A-B`) with
400 "Failed to get events endpoint" in September 2026, while single-day
queries kept working. The range call was the only request, so the Weekend
tab went empty with no error. The fetcher must fall back to one query per day.
"""
from datetime import date, timedelta

import data
import pytest


@pytest.fixture(autouse=True)
def _fresh_espn_state():
    data._ESPN_DAY_CACHE.clear()
    data._ESPN_RANGE_FAILED_AT[0] = 0.0
    yield


def _event(day: date, home: str, away: str, status="STATUS_SCHEDULED"):
    return {
        "date": f"{day.isoformat()}T14:00Z",
        "competitions": [{
            "status": {"type": {"name": status}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": home}},
                {"homeAway": "away", "team": {"displayName": away}},
            ],
        }],
    }


def test_falls_back_to_single_day_queries_when_range_fails(monkeypatch):
    today = date.today()
    sat = today + timedelta(days=3)
    later = today + timedelta(days=10)
    by_day = {
        sat.strftime("%Y%m%d"): [_event(sat, "Arsenal", "Chelsea")],
        later.strftime("%Y%m%d"): [_event(later, "Everton", "Fulham")],
    }

    def fake_fetch(url, timeout=10, retries=3):
        q = url.split("dates=")[1]
        if "-" in q:
            return None  # the range query now 400s
        return {"events": by_day.get(q, [])}

    monkeypatch.setattr(data, "_fetch_json_with_backoff", fake_fetch)
    fixtures = data.fetch_upcoming_fixtures(lookahead_days=14)

    # Only the next gameweek cluster comes back
    assert [(f["home"], f["away"]) for f in fixtures] == [("Arsenal", "Chelsea")]
    assert fixtures[0]["date"] == sat


def test_range_query_still_used_when_it_works(monkeypatch):
    today = date.today()
    sat = today + timedelta(days=2)
    calls = []

    def fake_fetch(url, timeout=10, retries=3):
        calls.append(url)
        return {"events": [_event(sat, "Arsenal", "Chelsea")]}

    monkeypatch.setattr(data, "_fetch_json_with_backoff", fake_fetch)
    fixtures = data.fetch_upcoming_fixtures(lookahead_days=14)

    assert len(calls) == 1
    assert len(fixtures) == 1


def test_per_day_fallback_is_cached_between_calls(monkeypatch):
    data._ESPN_DAY_CACHE.clear()
    calls = []

    def fake_fetch(url, timeout=10, retries=3):
        calls.append(url)
        return None if "-" in url.split("dates=")[1] else {"events": []}

    monkeypatch.setattr(data, "_fetch_json_with_backoff", fake_fetch)
    data.fetch_upcoming_fixtures(lookahead_days=10)
    first = len(calls)
    data.fetch_upcoming_fixtures(lookahead_days=10)
    assert len(calls) == first  # range skipped after failing, days cached


def test_promoted_names_resolve_even_before_match_data_loads(monkeypatch):
    """Streamlit reloads data.py when it changes on disk, which empties
    KNOWN_FD_TEAMS while load_data stays cached and never refills it. The
    promoted sides then kept ESPN's long names and auto-bet skipped them as
    unknown teams (16 Sep 2026). The fixed map must cover them on its own."""
    monkeypatch.setattr(data, "KNOWN_FD_TEAMS", set())
    monkeypatch.setattr(data, "KNOWN_SECOND_TIER_TEAMS", set())
    assert data._resolve_team_name("Hull City") == "Hull"
    assert data._resolve_team_name("Coventry City") == "Coventry"
    assert data._resolve_team_name("Ipswich Town") == "Ipswich"
