"""Which draw calibration should the live model use? Gameweek backtest.

The backtests behind the live config ran with calibration effectively OFF (the
pre-season window was too short, so it switched itself off), while live refits
an isotonic curve on the last 40 weeks and applies it to every bet. So the
thing deciding live stakes had never been backtested. This compares, gameweek
by gameweek on Main's live settings (50% gameweek cap, 5 pending bets):

  none                 raw model probability
  isotonic 40w (LIVE)  refit each gameweek on the prior 280 days, as live does
  isotonic 2y          same method, two years of history
  logistic 2y          smooth Platt curve on logit(raw), two years of history

Every calibrator is fitted only on out-of-sample predictions dated before the
gameweek it prices. Seasons 2023-24 to 2025-26 (2022-23 is history only).
Headline: worst season, deepest drawdown, median CLV vs the Pinnacle close,
and out-of-sample log loss of the draw probability itself.

Usage: python3 scripts/validate_draw_calibration.py   (needs the pack from
scripts/validate_exposure_cap.py build)
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import portfolio as pf  # noqa: E402
import validate_2026_27 as v  # noqa: E402

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
LIVE_KW = dict(gameweek_mode=True, max_pending_bets=5,
               exposure_cap_pct=0.50, exposure_cap_basis="gameweek")


class Logistic:
    """Platt scaling on logit(raw): monotone, smooth, no plateaus."""

    def __init__(self, raw: np.ndarray, actual: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        x = self._logit(raw).reshape(-1, 1)
        self.m = LogisticRegression(C=1e6).fit(x, actual.astype(int))

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def predict(self, xs):
        return self.m.predict_proba(self._logit(xs).reshape(-1, 1))[:, 1]


def make_fn(history: pd.DataFrame, kind: str, days: int):
    cache: dict = {}

    def fn(first_date):
        key = pd.Timestamp(first_date)
        if key in cache:
            return cache[key]
        w = history[(history["Date"] < key)
                    & (history["Date"] >= key - pd.Timedelta(days=days))]
        cal: dict = {}
        if kind == "isotonic":
            cal = pf.fit_calibrators_from_backtest(w)
        elif kind == "logistic" and len(w) >= pf.MIN_CALIBRATION_SAMPLES:
            y = w["_act_d"].to_numpy(float)
            if 0 < y.sum() < len(y):
                cal = {"D": Logistic(w["_dc_d"].to_numpy(float), y)}
        cache[key] = cal
        return cal
    return fn


def logloss_of(fn, frame: pd.DataFrame) -> float:
    blocks = pf._gameweek_blocks(frame["Date"])
    firsts = frame.groupby(blocks.values)["Date"].transform("min")
    p = []
    for raw, first in zip(frame["_dc_d"], firsts):
        cal = fn(first) if fn else None
        p.append(pf.calibrate_prob(raw, "D", cal))
    p = np.clip(np.array(p), 1e-4, 1 - 1e-4)
    y = frame["_act_d"].to_numpy(float)
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5)


def main() -> int:
    pack = pickle.loads(PACK.read_bytes())
    history = pd.concat([pack[s][2] for s in pack if s != "2026-27"],
                        ignore_index=True).sort_values("Date")
    live = v.live_main_config()
    variants = {
        "none":                None,
        "isotonic 40w (LIVE)": make_fn(history, "isotonic", 280),
        "isotonic 2y":         make_fn(history, "isotonic", 730),
        "logistic 2y":         make_fn(history, "logistic", 730),
    }
    results: dict = {}
    print(f"{'variant':22s} " + " ".join(f"{s:>9}" for s in TEST_SEASONS)
          + f" {'WORST':>8} {'maxDD':>6} {'bets':>5} {'CLV':>6} {'logloss':>8}")
    for name, fn in variants.items():
        per: dict = {}
        for s in TEST_SEASONS:
            dfs, ftr, bt, _ = pack[s]
            log, summ = pf.ev_backtest_simulate(
                bt, dfs, df_features=ftr, initial_bankroll=v.INITIAL,
                enable_simultaneous_correction=True, odds_source="Max",
                detect_source="PS", calibrators=None, calibrator_fn=fn,
                **LIVE_KW, **live)
            if "error" in summ:
                per[s] = {"profit": 0.0, "n_bets": 0, "max_dd": 0.0,
                          "median_clv": None, "logloss": logloss_of(fn, bt)}
                continue
            per[s] = {"profit": summ["profit"], "n_bets": summ["n_bets"],
                      "win_rate": summ["win_rate"],
                      "max_dd": round(v._max_drawdown_pct(log), 1),
                      "median_clv": v._median_clv(log, dfs),
                      "skipped_limits": summ.get("skipped_exposure_cap", 0),
                      "logloss": logloss_of(fn, bt)}
        profits = [x["profit"] for x in per.values()]
        clvs = [x["median_clv"] for x in per.values() if x.get("median_clv") is not None]
        r = {"per_season": per, "worst_season": min(profits),
             "total_profit": sum(profits),
             "max_dd": max(x["max_dd"] for x in per.values()),
             "bets": sum(x["n_bets"] for x in per.values()),
             "median_clv": round(float(np.median(clvs)), 2) if clvs else None,
             "mean_logloss": round(float(np.mean([x["logloss"] for x in per.values()])), 5)}
        results[name] = r
        clv_txt = f"{r['median_clv']:+.2f}%" if r["median_clv"] is not None else "n/a"
        print(f"{name:22s} " + " ".join(f"{per[s]['profit']:>+9,.0f}" for s in TEST_SEASONS)
              + f" {r['worst_season']:>+8,.0f} {r['max_dd']:>5.1f}% {r['bets']:>5} "
              f"{clv_txt:>6} {r['mean_logloss']:>8.5f}")
    for name, r in results.items():
        print(f"\n{name}: " + "; ".join(
            f"{s} bets {x['n_bets']} DD {x['max_dd']}% CLV {x['median_clv']} "
            f"skipped-by-limits {x.get('skipped_limits')} ll {x['logloss']}"
            for s, x in r["per_season"].items()))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"validate_draw_calibration_{ts}.json"
    path.write_text(json.dumps({"config": {k: sorted(x) if isinstance(x, set) else x
                                           for k, x in live.items()},
                                "live_kw": LIVE_KW, "seasons": TEST_SEASONS,
                                "results": results}, indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
