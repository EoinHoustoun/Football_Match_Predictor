"""Mock Two: draw stakes sized from the model pulled toward Pinnacle.

On Main's 2026-27 bets the model expected 12.3 wins, the closing market 9.1,
and 10 landed (median model EV +44%, median CLV +4%). Mock Two tests sizing
from a market-shrunk probability (`v2_market_shrink_beta`) while selection
stays on the model's own EV, so the setting can only make stakes smaller.
Research: scripts/research_shrunk_sizing.py.
"""
from __future__ import annotations

import pytest

import portfolio as pf

SHARP = {"H": 2.10, "D": 4.20, "A": 3.60}


def _port(**settings) -> dict:
    return {"initial_bankroll": 10_000.0, "bankroll": 10_000.0, "bets": [],
            "settings": {"kelly_fraction": 0.5, "max_stake_pct": 0.25,
                         "auto_markets": ["D"], "min_prob": 0.20,
                         "use_calibrated_probs": False,
                         "use_uncertainty_kelly": False, **settings}}


def _cand(raw: float = 0.36, sharp=SHARP) -> dict:
    return {"home": "A", "away": "B", "date": "2026-10-24", "market": "D",
            "selection": "Draw", "model_prob": raw, "odds": 4.5, "ev": 0.5,
            "detect_odds": 4.2, "sharp_1x2": sharp}


def test_beta_one_is_the_model_and_beta_zero_is_the_market():
    assert pf.market_shrunk_prob(0.36, 0.24, 1.0) == pytest.approx(0.36)
    assert pf.market_shrunk_prob(0.36, 0.24, 0.0) == pytest.approx(0.24)
    half = pf.market_shrunk_prob(0.36, 0.24, 0.5)
    assert 0.24 < half < 0.36


def test_shrink_cuts_the_stake_but_places_the_same_bet():
    off = pf.auto_place_value_bets_v2(_port(), [_cand()], 0.05)
    on = pf.auto_place_value_bets_v2(_port(v2_market_shrink_beta=0.5), [_cand()], 0.05)
    assert len(off) == len(on) == 1
    assert 0 < on[0]["stake"] < off[0]["stake"]
    assert on[0]["model_prob"] == off[0]["model_prob"]   # the model's number is recorded
    assert on[0]["v2_market_shrink_beta"] == 0.5
    assert on[0]["v2_market_prob"] < on[0]["v2_size_prob"] < 0.36


def test_off_by_default_and_audit_fields_absent():
    a = pf.auto_place_value_bets_v2(_port(), [_cand()], 0.05)
    b = pf.auto_place_value_bets_v2(_port(), [_cand(sharp=None)], 0.05)
    assert a[0]["stake"] == b[0]["stake"]
    assert "v2_market_prob" not in a[0]


def test_no_full_sharp_triplet_means_no_shrink():
    partial = {"D": 4.2}
    a = pf.auto_place_value_bets_v2(_port(), [_cand()], 0.05)
    b = pf.auto_place_value_bets_v2(_port(v2_market_shrink_beta=0.5),
                                    [_cand(sharp=partial)], 0.05)
    assert a[0]["stake"] == b[0]["stake"]


def test_shrink_never_touches_main():
    a = pf.auto_place_value_bets(_port(), [_cand()], 0.05)
    b = pf.auto_place_value_bets(_port(v2_market_shrink_beta=0.5), [_cand()], 0.05)
    assert a[0]["stake"] == b[0]["stake"]
