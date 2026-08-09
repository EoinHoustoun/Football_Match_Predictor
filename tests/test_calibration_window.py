"""A calibrator fitted on too little data must not be used.

The live path fitted isotonic regression on `test_weeks=10`, which pre-season
means the last 80 matches of the previous campaign. Isotonic on 80 points is not
a smooth curve, it is a three-step staircase, and the consequences were visible
in the book on 2026-08-09:

  - Six of nine opening fixtures were calibrated to *exactly* 31.2%, so the
    model's opinion about the match was erased and only the price decided which
    bet got placed. One bet went on.
  - Arsenal v Coventry went from a raw 17.3% to a calibrated **0.5%**, the
    isotonic floor, turning a plausible fixture into -96% EV.

Widening to a full season fixes the shape: 0.17 -> 0.258, 0.25 -> 0.258,
0.32 -> 0.294, 0.36 -> 0.326. Monotone, granular, no cliff.

The existing `DRAW_PROB_CAP` comment already documents this exact failure mode
from a 91-match window in June, where every raw prob above 0.40 mapped to 0.667.
That was capped at the top end. Nothing capped the bottom, and nothing widened
the window, so it came back pointing the other way.

`MIN_CALIBRATION_SAMPLES` is the guard: below it, return no calibrator at all.
Identity calibration is a defensible position. A three-step staircase is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio as pf


def _bt(n: int, draw_rate: float = 0.27, seed: int = 7) -> pd.DataFrame:
    """A backtest frame with `n` rows and a realistic draw rate."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.15, 0.40, n)
    acts = (rng.uniform(0, 1, n) < draw_rate).astype(float)
    return pd.DataFrame({
        "_dc_h": 1 - probs - 0.3, "_dc_d": probs, "_dc_a": np.full(n, 0.3),
        "_act_h": 1 - acts, "_act_d": acts, "_act_a": np.zeros(n),
    })


def test_a_healthy_sample_produces_a_draw_calibrator():
    assert "D" in pf.fit_calibrators_from_backtest(_bt(400))


def test_a_short_window_is_refused():
    """80 matches is what test_weeks=10 gives pre-season, and it is what
    produced the 31.2% plateau."""
    assert "D" not in pf.fit_calibrators_from_backtest(_bt(80))


def test_the_threshold_is_a_full_season_not_a_handful():
    assert pf.MIN_CALIBRATION_SAMPLES >= 200


def test_refusing_to_calibrate_leaves_the_probability_alone():
    """Identity is the safe fallback: no calibrator means the raw model
    probability is used, not a mangled one."""
    cal = pf.fit_calibrators_from_backtest(_bt(80))
    assert pf.calibrate_prob(0.173, "D", cal) == pytest.approx(0.173)


def test_an_empty_frame_is_refused():
    assert pf.fit_calibrators_from_backtest(pd.DataFrame()) == {}


def test_a_sample_with_no_draws_at_all_is_refused():
    """Isotonic on an all-negative sample maps everything to the floor."""
    frame = _bt(400, draw_rate=0.0)
    assert "D" not in pf.fit_calibrators_from_backtest(frame)


def test_the_draw_cap_still_applies_on_a_healthy_fit():
    """The existing top-end guard must survive this change."""
    cal = pf.fit_calibrators_from_backtest(_bt(400, draw_rate=0.95))
    assert pf.calibrate_prob(0.40, "D", cal) <= pf.DRAW_PROB_CAP
