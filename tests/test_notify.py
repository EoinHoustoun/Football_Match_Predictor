"""Notifications: the right events, sent once, and never able to break a run."""
from __future__ import annotations

import json

import pytest

import notify


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "CONFIG", tmp_path / "notify.json")
    monkeypatch.setattr(notify, "SENT", tmp_path / "sent.json")
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    (tmp_path / "notify.json").write_text(json.dumps({"topic": "t", "enabled": True}))
    return tmp_path


PLACED = {"type": "auto_bet_placed", "portfolio": "main", "match": "Arsenal vs Leeds",
          "market": "D", "selection": "Draw", "stake": 4190.28, "odds": 5.5, "ev": 0.505}
SETTLED = {"type": "settled", "portfolio": "main", "match": "Fulham vs Man United",
           "market": "D", "result": "won", "profit": 4946.49}


def test_formats_the_events_that_matter():
    p = notify.format_event(PLACED)
    assert p["title"] == "Main: Draw backed"
    assert "Arsenal v Leeds @5.50" in p["body"] and "£4,190" in p["body"]
    s = notify.format_event(SETTLED)
    assert s["title"] == "Main: WON +£4,946"
    lost = notify.format_event({**SETTLED, "result": "lost", "profit": -3055.79})
    assert lost["title"] == "Main: lost −£3,056"
    assert notify.format_event({"type": "clv_snapshot"}) is None
    assert notify.format_event({"type": "run_completed"}) is None


def test_each_event_is_sent_once(cfg):
    calls = []
    send = lambda c, m: calls.append(m) or True
    assert notify.notify_event(SETTLED, sender=send)
    assert not notify.notify_event(SETTLED, sender=send)   # runner and app both log it
    assert notify.notify_event(PLACED, sender=send)
    assert len(calls) == 2


def test_a_failing_sender_never_raises(cfg):
    def boom(c, m):
        raise OSError("no network")
    assert notify.notify_event(PLACED, sender=boom) is False
    # and it was not recorded as sent, so it retries next time
    assert notify.notify_event(PLACED, sender=lambda c, m: True)


def test_disabled_or_unconfigured_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "CONFIG", tmp_path / "missing.json")
    monkeypatch.setattr(notify, "SENT", tmp_path / "sent.json")
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert notify.notify_event(PLACED, sender=lambda c, m: True) is False


def test_created_topic_is_unguessable(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "CONFIG", tmp_path / "n.json")
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    c = notify.load_config(create=True)
    assert c["topic"].startswith("fpred-") and len(c["topic"]) == len("fpred-") + 16
