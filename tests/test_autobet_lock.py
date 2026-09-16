"""Only one auto-bet run may hold the portfolios at a time.

Each run loads both portfolio files, decides, and saves them whole. Two browser
sessions, or a session and the launchd runner, running together each saved
their own copy, so the log recorded every event twice (16 Sep 2026) and a run
that decided differently would have silently overwritten the other's bets.
"""
from __future__ import annotations

import portfolio as pf


def test_second_holder_is_refused_until_the_first_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "DATA_DIR", tmp_path)
    with pf.autobet_lock() as first:
        assert first is True
        with pf.autobet_lock() as second:
            assert second is False
    with pf.autobet_lock() as again:
        assert again is True


def test_lock_is_released_when_the_run_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "DATA_DIR", tmp_path)
    try:
        with pf.autobet_lock() as got:
            assert got
            raise RuntimeError("run failed")
    except RuntimeError:
        pass
    with pf.autobet_lock() as got:
        assert got is True
