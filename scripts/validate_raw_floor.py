"""Should live refuse draws whose RAW model probability is below a floor?

Background (data/diagnostics/gw5_review_2026-09-22.md): the live isotonic
calibrator refits on a trailing 40 weeks. That window is now a 32% draw-rate
stretch, and the fitted curve lifts raw 0.22-0.30 draws to 0.34, so the live
line bets 5.4 times a gameweek where every backtest season managed 1.5-2.5.
Pooled 2023-26 out of sample, raw 0.22-0.30 has no edge over Pinnacle; raw
>= 0.30 does. A floor on the raw number is a tightening that no trailing
window can undo.

This runs the gameweek simulator on the honest per-season pack with Main's
live config, the live-style trailing isotonic calibrator (refit each gameweek
on the prior 280 days, as live does) AND with no calibrator, across floors
None / 0.26 / 0.28 / 0.30 / 0.32. A floor earns its place only if it holds
up under both calibration regimes, which is what "robust, not a spike" means
here.

Headline: worst season, deepest drawdown, bets per gameweek, median CLV vs
the Pinnacle close, compounded multiple across the three seasons.

Usage: python3 scripts/validate_raw_floor.py   (needs the pack from
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
from validate_draw_calibration import make_fn  # noqa: E402

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
LIVE_KW = dict(gameweek_mode=True, max_pending_bets=None,
               exposure_cap_pct=0.50, exposure_cap_basis="gameweek")
FLOORS = [None, 0.26, 0.28, 0.30, 0.32]


def run(pack, fn, floor: float | None, live: dict) -> dict:
    per: dict = {}
    for s in TEST_SEASONS:
        dfs, ftr, bt, _ = pack[s]
        log, summ = pf.ev_backtest_simulate(
            bt, dfs, df_features=ftr, initial_bankroll=v.INITIAL,
            enable_simultaneous_correction=True, odds_source="Max",
            detect_source="PS", calibrators=None, calibrator_fn=fn,
            min_raw_draw_prob=floor, **LIVE_KW, **live)
        if "error" in summ:
            per[s] = {"profit": 0.0, "n_bets": 0, "max_dd": 0.0,
                      "gameweeks": 0, "median_clv": None, "skipped_raw_floor": 0}
            continue
        per[s] = {"profit": summ["profit"], "n_bets": summ["n_bets"],
                  "max_dd": round(v._max_drawdown_pct(log), 1),
                  "gameweeks": int(pf._gameweek_blocks(log["Date"]).nunique()),
                  "median_clv": v._median_clv(log, dfs),
                  "skipped_raw_floor": summ.get("skipped_raw_floor", 0),
                  "win_rate": summ.get("win_rate")}
    profits = [x["profit"] for x in per.values()]
    clvs = [x["median_clv"] for x in per.values() if x.get("median_clv") is not None]
    final = v.INITIAL
    for p in profits:
        final *= 1 + p / v.INITIAL
    bets = sum(x["n_bets"] for x in per.values())
    gws = sum(x["gameweeks"] for x in per.values())
    return {"per_season": per, "worst_season": min(profits), "total_profit": sum(profits),
            "max_dd": max(x["max_dd"] for x in per.values()), "bets": bets,
            "bets_per_gw": round(bets / gws, 2) if gws else 0.0,
            "median_clv": round(float(np.median(clvs)), 2) if clvs else None,
            "compounded_final": round(final, 2)}


def main() -> int:
    pack = pickle.loads(PACK.read_bytes())
    history = pd.concat([pack[s][2] for s in pack if s != "2026-27"],
                        ignore_index=True).sort_values("Date")
    live = v.live_main_config()
    live.pop("min_raw_draw_prob", None)
    regimes = {"isotonic 40w (LIVE)": make_fn(history, "isotonic", 280),
               "no calibration": None}
    results: dict = {}
    for rname, fn in regimes.items():
        print(f"\n== calibration: {rname}")
        print(f"{'raw floor':10s} " + " ".join(f"{s:>9}" for s in TEST_SEASONS)
              + f" {'WORST':>8} {'maxDD':>6} {'bets':>5} {'b/GW':>5} {'CLV':>6} {'final £':>9}")
        for floor in FLOORS:
            r = run(pack, fn, floor, live)
            results[f"{rname} | floor {floor}"] = r
            clv = f"{r['median_clv']:+.2f}%" if r["median_clv"] is not None else "n/a"
            print(f"{str(floor):10s} "
                  + " ".join(f"{r['per_season'][s]['profit']:>+9,.0f}" for s in TEST_SEASONS)
                  + f" {r['worst_season']:>+8,.0f} {r['max_dd']:>5.1f}% {r['bets']:>5}"
                  + f" {r['bets_per_gw']:>5.2f} {clv:>6} {r['compounded_final']:>9,.0f}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"raw_floor_{ts}.json"
    path.write_text(json.dumps({"config": {k: sorted(x) if isinstance(x, set) else x
                                           for k, x in live.items()},
                                "live_kw": LIVE_KW, "seasons": TEST_SEASONS,
                                "results": results}, indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
