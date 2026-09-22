"""Floor on the RAW draw probability, before any calibrator can lift it.

The live isotonic calibrator refits on a trailing window; in a draw-heavy
window it mapped raw 0.22-0.30 to 0.34 and tripled the live bet rate
(data/diagnostics/gw5_review_2026-09-22.md). The floor gates the model's
own number, so no window can talk a raw 0.22 into a bet. Off by default,
so every existing path is unchanged until a portfolio sets it.
"""
from __future__ import annotations

import pandas as pd
import pytest

import portfolio as pf

from tests.test_gameweek_sim import KW, _frames


class _Lift:
    """Stand-in calibrator: maps every raw prob to one fixed value."""

    def __init__(self, to: float):
        self.to = to

    def predict(self, xs):
        return [self.to for _ in xs]


THREE = [
    ("2025-09-13", "A", "B", 25, 4.5, "D"),   # raw 0.25
    ("2025-09-13", "C", "D", 28, 4.5, "H"),   # raw 0.28
    ("2025-09-14", "E", "F", 34, 4.5, "H"),   # raw 0.34
]


def test_floor_is_applied_to_the_raw_prob_not_the_calibrated_one():
    bt, df = _frames(THREE)
    lifted = {"D": _Lift(0.34)}   # a draw-heavy window: everything becomes 0.34
    log_open, s_open = pf.ev_backtest_simulate(bt, df, calibrators=lifted, **KW)
    log_floor, s_floor = pf.ev_backtest_simulate(
        bt, df, calibrators=lifted, min_raw_draw_prob=0.30, **KW)
    # Without the floor the calibrator makes all three look identical.
    assert len(log_open) == 3 and s_open["skipped_raw_floor"] == 0
    # With it, only the match the model itself rated at 0.34 survives.
    assert log_floor["Match"].tolist() == ["E vs F"]
    assert s_floor["skipped_raw_floor"] == 2


def test_floor_off_by_default_leaves_the_simulator_unchanged():
    bt, df = _frames(THREE)
    log_a, s_a = pf.ev_backtest_simulate(bt, df, **KW)
    log_b, s_b = pf.ev_backtest_simulate(bt, df, min_raw_draw_prob=None, **KW)
    pd.testing.assert_frame_equal(log_a, log_b)
    assert s_a["skipped_raw_floor"] == s_b["skipped_raw_floor"] == 0


def test_floor_only_touches_the_draw_market():
    bt, df = _frames([("2025-09-13", "A", "B", 25, 4.5, "H")])
    # Home prob is (100-25)/2 = 37.5% at odds 3.0: +EV, and not a draw.
    kw = {**KW, "allowed_markets": {"D", "H"}}
    log, s = pf.ev_backtest_simulate(bt, df, min_raw_draw_prob=0.30, **kw)
    assert log["Market"].tolist() == ["Home Win"]
    assert s["skipped_raw_floor"] == 1


def _port(**settings) -> dict:
    return {"initial_bankroll": 10_000.0, "bankroll": 10_000.0, "bets": [],
            "settings": {"kelly_fraction": 0.5, "max_stake_pct": 0.10,
                         "auto_markets": ["D"], "min_prob": 0.20,
                         "use_calibrated_probs": True, **settings}}


def _cand(raw: float) -> dict:
    return {"home": "A", "away": "B", "date": "2026-10-10", "market": "D",
            "selection": "Draw", "model_prob": raw, "odds": 4.5, "ev": 0.5,
            "detect_odds": 4.2}


@pytest.mark.parametrize("place", [pf.auto_place_value_bets, pf.auto_place_value_bets_v2])
def test_live_paths_refuse_a_lifted_draw_below_the_raw_floor_and_say_why(place):
    lifted = {"D": _Lift(0.34)}
    skips: list = []
    port = _port(min_raw_draw_prob=0.30)
    placed = place(port, [_cand(0.25)], 0.05, calibrators=lifted, skip_log=skips)
    assert placed == [] and port["bets"] == []
    assert [s["reason"] for s in skips] == ["raw_floor"]
    assert "raw draw 0.250 < 0.30" in skips[0]["detail"]
    assert "calibrated 0.340" in skips[0]["detail"]


@pytest.mark.parametrize("place", [pf.auto_place_value_bets, pf.auto_place_value_bets_v2])
def test_live_paths_still_place_a_draw_the_model_itself_rates_above_the_floor(place):
    lifted = {"D": _Lift(0.34)}
    port = _port(min_raw_draw_prob=0.30)
    placed = place(port, [_cand(0.33)], 0.05, calibrators=lifted)
    assert len(placed) == 1 and placed[0]["market"] == "D"


@pytest.mark.parametrize("place", [pf.auto_place_value_bets, pf.auto_place_value_bets_v2])
def test_live_paths_are_unchanged_when_the_floor_is_not_set(place):
    lifted = {"D": _Lift(0.34)}
    port = _port()
    placed = place(port, [_cand(0.25)], 0.05, calibrators=lifted)
    assert len(placed) == 1
