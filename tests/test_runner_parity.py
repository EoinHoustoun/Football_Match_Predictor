"""The headless runner must behave exactly like the live app.

`scripts/run_auto_bet.py` is the launchd path: it places bets when nobody has
Streamlit open. It grew as a copy of the auto-bet block in `app.py`, and the two
then drifted. Both promoted-team safety fixes landed in `app.py` only:

  1. `app.py` seeds unrated sides with `PROMOTED_PRIOR` so a promoted fixture is
     priced honestly and stays visible. The runner dropped it with a bare
     `continue`, so two of the ten opening fixtures left no trace.
  2. `app.py` passes `match_counts` into the placement calls, arming the
     no-history gate. The runner passed nothing, and `should_skip_unrated`
     treats absent counts as "gate disabled" — so from a promoted side's second
     Premier League match the runner would stake off a rating fitted on one game.

Neither is live today (auto-bet is off, launchd is not loaded), which is exactly
why it needs a test: the bug only fires the day auto-bet is switched on.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

import portfolio as pf
from models import PROMOTED_PRIOR

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    """Import scripts/run_auto_bet.py as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "run_auto_bet", ROOT / "scripts" / "run_auto_bet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# ── Seeding parity ────────────────────────────────────────────────────────────

def test_seeding_rates_a_promoted_side_in_every_ratings_dict(toy_dc_ratings):
    """A promoted team must be seeded into all three ratings dicts.

    The runner fits standard DC, draw-specialist DC and the KN variant. Seeding
    one and not the others would price the same fixture three different ways.
    """
    fixtures = [{"home": "A", "away": "Hull"}]
    dc, draw, kn = runner.seeded_ratings_for(
        fixtures,
        copy.deepcopy(toy_dc_ratings),
        copy.deepcopy(toy_dc_ratings),
        copy.deepcopy(toy_dc_ratings),
    )
    # The point is agreement across the three fits, not the specific value:
    # Hull is priced by the relegation market, so it is no longer the pooled
    # prior. Seeding one dict differently would quote the fixture three ways.
    for ratings in (dc, draw, kn):
        assert ratings["attacks"]["Hull"] == dc["attacks"]["Hull"]
        assert ratings["defenses"]["Hull"] == dc["defenses"]["Hull"]
        assert ratings["seeded_teams"] == ["Hull"]
    assert dc["attacks"]["Hull"] < PROMOTED_PRIOR["attack"]


def test_seeding_leaves_a_rated_side_untouched(toy_dc_ratings):
    fixtures = [{"home": "A", "away": "B"}]
    dc, _, _ = runner.seeded_ratings_for(
        fixtures,
        copy.deepcopy(toy_dc_ratings),
        copy.deepcopy(toy_dc_ratings),
        copy.deepcopy(toy_dc_ratings),
    )
    assert dc["attacks"]["A"] == toy_dc_ratings["attacks"]["A"]
    assert dc["seeded_teams"] == []


# ── Gate parity ───────────────────────────────────────────────────────────────

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
            "min_team_matches": 6,
        },
    }
    p["settings"].update(settings)
    return p


def _draw_candidate(home: str, away: str) -> dict:
    """A draw priced well above the market: 30% against 9.20 is +176% EV."""
    return {
        "home": home, "away": away, "date": "2026-08-21",
        "market": "D", "selection": "Draw",
        "odds": 9.20, "detect_odds": 9.20,
        "model_prob": 0.30, "ev": 1.76,
        "home_elo": 1800, "away_elo": 1500,
    }


COUNTS = {"Arsenal": 190, "Everton": 190, "Hull": 1}


def test_runner_blocks_a_bet_on_a_side_with_too_little_history():
    main_port, mt_port = _portfolio(), _portfolio()
    cands = [_draw_candidate("Arsenal", "Hull")]

    placed_main, placed_mt = runner.place_candidates(
        main_port, mt_port, cands, cands,
        calibrators=None, bin_variances=None, match_counts=COUNTS,
    )

    assert placed_main == []
    assert placed_mt == []
    assert main_port["bets"] == []


def test_runner_still_places_a_bet_on_two_rated_sides():
    main_port, mt_port = _portfolio(), _portfolio()
    cands = [_draw_candidate("Arsenal", "Everton")]

    placed_main, _ = runner.place_candidates(
        main_port, mt_port, cands, cands,
        calibrators=None, bin_variances=None, match_counts=COUNTS,
    )

    assert len(placed_main) == 1
    assert placed_main[0]["home"] == "Arsenal"


def test_runner_records_why_it_skipped():
    """A silent skip is how two invisible fixtures happened. Log the reason."""
    main_port, mt_port = _portfolio(), _portfolio()
    skip_log: list[dict] = []

    runner.place_candidates(
        main_port, mt_port,
        [_draw_candidate("Arsenal", "Hull")], [],
        calibrators=None, bin_variances=None, match_counts=COUNTS,
        skip_log=skip_log,
    )

    assert any(e.get("reason") == "no_history" for e in skip_log)


def test_runner_leaves_a_disabled_portfolio_alone():
    main_port = _portfolio(auto_bet_enabled=False)
    mt_port = _portfolio()
    cands = [_draw_candidate("Arsenal", "Everton")]

    placed_main, placed_mt = runner.place_candidates(
        main_port, mt_port, cands, cands,
        calibrators=None, bin_variances=None, match_counts=COUNTS,
    )

    assert placed_main == []
    assert len(placed_mt) == 1
