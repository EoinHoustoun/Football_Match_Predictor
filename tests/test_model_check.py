"""Model Check summaries: interval maths and the bucketing the page relies on."""
from __future__ import annotations

import numpy as np
import pandas as pd

import model_check as mc


def test_wilson_interval_brackets_the_rate_and_widens_when_small():
    lo, hi = mc.wilson(30, 100)
    assert lo < 0.30 < hi
    lo_s, hi_s = mc.wilson(3, 10)
    assert (hi_s - lo_s) > (hi - lo)
    assert mc.wilson(0, 0) == (0.0, 1.0)
    lo0, _ = mc.wilson(0, 20)
    assert lo0 == 0.0


def _frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.18, 0.40, n)
    drew = (rng.uniform(size=n) < p).astype(float)
    psd = 1 / (p * 0.95)
    return pd.DataFrame({"Date": pd.date_range("2024-08-01", periods=n), "Season": "2024-25",
                         "p_d": p, "drew": drew, "psd": psd, "mkt_d": p * 0.95})


def test_reliability_bins_cover_every_match_and_stay_ordered():
    f = _frame(n=6000)
    r = mc.reliability(f)
    assert r["n"].sum() == len(f)
    assert list(r["bin"]) == [b for b in mc.BIN_LABELS if b in set(r["bin"])]
    assert (r["lo"] <= r["actual"]).all() and (r["actual"] <= r["hi"]).all()
    # a calibrated synthetic model sits near the diagonal in the well-filled bins
    big = r[r["n"] > 600]
    assert ((big["actual"] - big["predicted"]).abs() < 0.04).all()


def test_edge_by_bucket_settles_flat_stakes_at_the_close():
    f = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01"] * 4), "Season": "x",
                      "p_d": [0.32, 0.32, 0.32, 0.32], "drew": [1, 0, 0, 0],
                      "psd": [4.0, 4.0, 4.0, 4.0], "mkt_d": [0.25] * 4})
    e = mc.edge_by_bucket(f)
    row = e.iloc[0]
    assert row["bin"] == "30 to 34%" and row["n"] == 4
    assert abs(row["roi"] - 0.0) < 1e-9          # +3 -1 -1 -1 = 0 on 4 stakes


def test_season_table_and_rolling_rate():
    f = _frame()
    t = mc.season_table(f)
    assert t.iloc[0]["n"] == len(f)
    assert 0 < t.iloc[0]["brier_model"] < 0.3
    r = mc.rolling_draw_rate(f, window=76)
    assert len(r) > 0 and r["actual"].between(0, 1).all()


def test_missing_pack_is_handled(tmp_path):
    assert mc.load_pack(tmp_path / "nope.pkl") == (None, None)
