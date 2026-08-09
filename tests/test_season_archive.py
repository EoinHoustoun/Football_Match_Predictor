"""Season archive — preserve a closed season, then reset the live lines.

The archive is the guard that makes starting a new season non-destructive:
`reset_for_new_season` refuses to run until the outgoing season is safely
archived. These tests pin that guard down, plus the manifest arithmetic and
the byte-identity of the archived copies.
"""
from __future__ import annotations

import json

import pytest

import portfolio as pf
import season_archive as sa


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _bet(bet_id: str, status: str, stake: float, profit: float,
         clv: float | None = None) -> dict:
    b = {
        "id": bet_id, "home": "Everton", "away": "Fulham", "date": "2026-05-01",
        "market": "D", "selection": "Draw", "model_prob": 0.34, "odds": 4.2,
        "ev": 0.43, "stake": stake, "status": status, "profit": profit,
        "placed_at": "2026-04-28T12:00:00", "settled_at": "2026-05-01T20:00:00",
    }
    if clv is not None:
        b["clv"] = clv
    return b


@pytest.fixture
def live_portfolios(tmp_path, monkeypatch):
    """Point both live portfolio files and the archive root at tmp_path."""
    main_file = tmp_path / "portfolio.json"
    two_file  = tmp_path / "portfolio_two.json"
    monkeypatch.setattr(pf, "PORTFOLIO_FILE", main_file)
    monkeypatch.setattr(pf, "MOCK2_PORTFOLIO_FILE", two_file)
    monkeypatch.setattr(pf, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sa, "SEASONS_DIR", tmp_path / "seasons")

    main = {
        "initial_bankroll": 10000.0,
        "bankroll": 12000.0,
        "bets": [
            _bet("a1", "won",  500.0, 1600.0, clv=0.08),
            _bet("a2", "lost", 400.0, -400.0, clv=-0.02),
            _bet("a3", "pending", 300.0, 0.0),
        ],
        "settings": {
            "odds_api_key": "secret-key",
            "auto_markets": ["D", "under25"],
            "min_prob": 0.21,
            "main_settings_label": "2026-27 defaults",
        },
    }
    two = {
        "initial_bankroll": 10000.0,
        "bankroll": 11000.0,
        "bets": [_bet("b1", "won", 250.0, 1000.0, clv=0.05)],
        "settings": {
            "auto_markets": ["D"],
            "v2_settings_label": "K-N stack",
        },
    }
    pf.save_portfolio(main)
    pf.save_portfolio_two(two)
    return {"main": main, "two": two, "main_file": main_file, "two_file": two_file}


# ── Archiving ─────────────────────────────────────────────────────────────────

def test_archive_season_writes_both_portfolios_and_a_manifest(live_portfolios):
    sa.archive_season("2025-26")

    d = sa.SEASONS_DIR / "2025-26"
    assert (d / "portfolio.json").exists()
    assert (d / "portfolio_two.json").exists()
    assert (d / "manifest.json").exists()


def test_archived_copies_are_byte_identical_to_the_live_files(live_portfolios):
    sa.archive_season("2025-26")

    d = sa.SEASONS_DIR / "2025-26"
    assert (d / "portfolio.json").read_bytes() == live_portfolios["main_file"].read_bytes()
    assert (d / "portfolio_two.json").read_bytes() == live_portfolios["two_file"].read_bytes()


def test_manifest_arithmetic_matches_portfolio_stats(live_portfolios):
    manifest = sa.archive_season("2025-26")

    main = manifest["portfolios"]["main"]
    stats = pf.portfolio_stats(live_portfolios["main"])
    assert main["initial_bankroll"] == 10000.0
    assert main["final_bankroll"]   == 12000.0
    assert main["profit"]           == stats["profit"]     # 1600 - 400 = 1200
    assert main["roi_pct"]          == stats["roi"]        # 1200 / 900 staked
    assert main["n_bets"]           == 3
    assert main["n_settled"]        == 2
    assert main["win_rate"]         == stats["win_rate"]   # 1 of 2 settled


def test_manifest_carries_median_clv_and_settings_label(live_portfolios):
    manifest = sa.archive_season("2025-26")

    # Two tagged bets at +0.08 and -0.02 — median is the midpoint.
    assert manifest["portfolios"]["main"]["median_clv"] == pytest.approx(0.03)
    assert manifest["portfolios"]["main"]["settings_label"] == "2026-27 defaults"
    assert manifest["portfolios"]["mock_two"]["settings_label"] == "K-N stack"
    assert manifest["season"] == "2025-26"
    assert manifest["closed_at"]


def test_manifest_does_not_leak_the_odds_api_key(live_portfolios):
    sa.archive_season("2025-26")
    raw = (sa.SEASONS_DIR / "2025-26" / "manifest.json").read_text()
    assert "secret-key" not in raw


def test_archive_season_refuses_to_overwrite_by_default(live_portfolios):
    sa.archive_season("2025-26")
    with pytest.raises(sa.ArchiveError):
        sa.archive_season("2025-26")


def test_archive_season_overwrites_when_asked(live_portfolios):
    sa.archive_season("2025-26")
    manifest = sa.archive_season("2025-26", overwrite=True)
    assert manifest["season"] == "2025-26"


# ── Reading archives back ─────────────────────────────────────────────────────

def test_list_archived_seasons_is_newest_first(live_portfolios):
    sa.archive_season("2024-25")
    sa.archive_season("2025-26")
    assert sa.list_archived_seasons() == ["2025-26", "2024-25"]


def test_load_archived_season_round_trips_the_bets(live_portfolios):
    sa.archive_season("2025-26")
    loaded = sa.load_archived_season("2025-26")

    assert [b["id"] for b in loaded["main"]["bets"]] == ["a1", "a2", "a3"]
    assert loaded["mock_two"]["bets"][0]["id"] == "b1"
    assert loaded["manifest"]["season"] == "2025-26"


def test_load_archived_season_raises_for_an_unknown_season(live_portfolios):
    with pytest.raises(sa.ArchiveError):
        sa.load_archived_season("1999-00")


def test_is_archived_reports_the_truth(live_portfolios):
    assert not sa.is_archived("2025-26")
    sa.archive_season("2025-26")
    assert sa.is_archived("2025-26")


# ── The reset guard ───────────────────────────────────────────────────────────

def test_reset_refuses_when_the_outgoing_season_is_not_archived(live_portfolios):
    with pytest.raises(sa.ArchiveError):
        sa.reset_for_new_season("2026-27", closing_season="2025-26",
                                initial_bankroll=10000.0)

    # The live portfolio must be untouched by the refusal.
    assert pf.load_portfolio()["bankroll"] == 12000.0
    assert len(pf.load_portfolio()["bets"]) == 3


def test_reset_starts_both_lines_fresh(live_portfolios):
    sa.archive_season("2025-26")
    sa.reset_for_new_season("2026-27", closing_season="2025-26",
                            initial_bankroll=10000.0)

    for loaded in (pf.load_portfolio(), pf.load_portfolio_two()):
        assert loaded["initial_bankroll"] == 10000.0
        assert loaded["bankroll"] == 10000.0
        assert loaded["bets"] == []
        assert loaded["season"] == "2026-27"


def test_reset_preserves_the_odds_api_key(live_portfolios):
    sa.archive_season("2025-26")
    sa.reset_for_new_season("2026-27", closing_season="2025-26",
                            initial_bankroll=10000.0)
    assert pf.load_portfolio()["settings"]["odds_api_key"] == "secret-key"


def test_reset_applies_settings_overrides(live_portfolios):
    sa.archive_season("2025-26")
    sa.reset_for_new_season(
        "2026-27", closing_season="2025-26", initial_bankroll=10000.0,
        main_settings={"auto_markets": ["D"]},
        two_settings={"auto_markets": ["D"], "detect_source": "PS"},
    )
    assert pf.load_portfolio()["settings"]["auto_markets"] == ["D"]
    assert pf.load_portfolio_two()["settings"]["detect_source"] == "PS"
    # Untouched keys survive the override.
    assert pf.load_portfolio()["settings"]["min_prob"] == 0.21


def test_reset_backs_up_both_live_files_first(live_portfolios, tmp_path):
    sa.archive_season("2025-26")
    result = sa.reset_for_new_season("2026-27", closing_season="2025-26",
                                     initial_bankroll=10000.0)

    for path in result["backups"].values():
        assert path.exists()
    # The backup holds the pre-reset state, not the fresh one.
    restored = json.loads(result["backups"]["main"].read_text())
    assert restored["bankroll"] == 12000.0
    assert len(restored["bets"]) == 3


def test_reset_is_refused_a_second_time_for_the_same_season(live_portfolios):
    sa.archive_season("2025-26")
    sa.reset_for_new_season("2026-27", closing_season="2025-26",
                            initial_bankroll=10000.0)
    # 2025-26 is archived, but the live line is already on 2026-27 — resetting
    # again would silently discard whatever has been bet since.
    with pytest.raises(sa.ArchiveError):
        sa.reset_for_new_season("2026-27", closing_season="2025-26",
                                initial_bankroll=10000.0)
