"""Is the 35% draw-calibration ceiling real, or an artefact of the fit?

On 16 Sep 2026 the live isotonic calibrator mapped every raw draw probability
above ~33% to exactly 35%, so seven of ten weekend draws tied and the bet order
came down to price alone. This asks three questions, analysis only:

  1. The live curve: which raw values does it pool into the top plateau, and on
     how many matches is that plateau built?
  2. Out of sample: fit a calibrator on one season's out-of-sample predictions,
     apply it to the next season. Does it beat the raw probability (Brier,
     log loss), and does the top plateau hold up on data it never saw?
  3. Against the market: for each raw-probability band, the actual draw rate
     versus Pinnacle's implied draw probability, and flat-stake ROI at the Max
     price. Calibration only matters for betting where it moves this.

Output: stdout + data/diagnostics/draw_calibration_<ts>.json
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

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"
BANDS = [0.0, 0.25, 0.30, 0.33, 0.36, 0.40, 1.0]


def _curve(iso) -> list[tuple[float, float]]:
    return [(round(float(x), 4), round(float(y), 4))
            for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)]


def _with_odds(bt: pd.DataFrame, dfs: pd.DataFrame) -> pd.DataFrame:
    odds = dfs[["Date", "HomeTeam", "AwayTeam", "PSD", "MaxD"]].rename(
        columns={"HomeTeam": "Home", "AwayTeam": "Away"})
    m = bt.merge(odds, on=["Date", "Home", "Away"], how="left")
    return m


def _band_table(frame: pd.DataFrame, cal: dict | None) -> list[dict]:
    f = frame.copy()
    f["band"] = pd.cut(f["_dc_d"], BANDS, right=False)
    if cal and "D" in cal:
        f["cal"] = [pf.calibrate_prob(p, "D", cal) for p in f["_dc_d"]]
    else:
        f["cal"] = f["_dc_d"]
    out = []
    for band, g in f.groupby("band", observed=True):
        ok = g.dropna(subset=["PSD", "MaxD"])
        ok = ok[(ok["PSD"] > 1) & (ok["MaxD"] > 1)]
        # Pinnacle implied draw prob, margin roughly removed by scaling 1/PSD
        implied = float((1 / ok["PSD"]).mean()) if len(ok) else None
        roi = (float(((ok["_act_d"] * ok["MaxD"]) - 1).mean() * 100)
               if len(ok) else None)
        out.append({
            "band": f"{band.left:.2f}-{band.right:.2f}",
            "n": int(len(g)),
            "mean_raw": round(float(g["_dc_d"].mean()), 3),
            "mean_calibrated": round(float(g["cal"].mean()), 3),
            "actual_draw_rate": round(float(g["_act_d"].mean()), 3),
            "pinnacle_implied": round(implied, 3) if implied else None,
            "flat_roi_at_max_pct": round(roi, 1) if roi is not None else None,
        })
    return out


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"brier": round(float(np.mean((p - y) ** 2)), 5),
            "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5)}


def main() -> int:
    pack = pickle.loads(PACK.read_bytes())
    seasons = [s for s in pack if s != "2026-27"]
    frames = {s: _with_odds(pack[s][2], pack[s][0]) for s in seasons}
    report: dict = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    # ── 1. Live curve ──────────────────────────────────────────────────────
    from data import load_data, add_rolling_features
    from models import backtest_models
    df = load_data()
    ftr = add_rolling_features(df)
    bt_live = backtest_models(df, ftr, test_weeks=40)
    live_cal = pf.fit_calibrators_from_backtest(bt_live)
    iso = live_cal["D"]
    curve = _curve(iso)
    top_y = max(y for _, y in curve)
    plateau_x = [x for x, y in curve if y == top_y]
    lo = min(plateau_x)
    pooled = bt_live[bt_live["_dc_d"] >= lo]
    report["live"] = {
        "window": [str(bt_live["Date"].min().date()), str(bt_live["Date"].max().date())],
        "n_matches": int(len(bt_live)),
        "draw_rate": round(float(bt_live["_act_d"].mean()), 3),
        "curve": curve,
        "top_plateau": {"value": top_y, "raw_from": lo, "n_pooled": int(len(pooled)),
                        "draws_in_pool": int(pooled["_act_d"].sum())},
        "cap": pf.DRAW_PROB_CAP,
    }
    print(f"LIVE calibrator window {report['live']['window']}, {len(bt_live)} matches, "
          f"draw rate {report['live']['draw_rate']:.1%}")
    print(f"  curve (raw -> calibrated): {curve}")
    print(f"  top plateau {top_y:.3f} pools raw >= {lo:.3f}: {len(pooled)} matches, "
          f"{int(pooled['_act_d'].sum())} draws")
    print("  bands in the live window:")
    for row in _band_table(_with_odds(bt_live, df), live_cal):
        print(f"    {row}")

    # ── 2 + 3. Out of sample, season to season ────────────────────────────
    report["oos"] = {}
    for prev, nxt in zip(seasons, seasons[1:]):
        cal = pf.fit_calibrators_from_backtest(pack[prev][2])
        f = frames[nxt]
        y = f["_act_d"].to_numpy(float)
        raw = f["_dc_d"].to_numpy(float)
        calp = np.array([pf.calibrate_prob(p, "D", cal) for p in raw]) if "D" in cal else raw
        entry = {
            "fit_on": prev, "tested_on": nxt, "n": int(len(f)),
            "raw": _scores(y, raw), "calibrated": _scores(y, calp),
            "curve": _curve(cal["D"]) if "D" in cal else None,
            "bands": _band_table(f, cal),
        }
        report["oos"][nxt] = entry
        print(f"\nOOS: fit on {prev}, tested on {nxt} ({len(f)} matches)")
        print(f"  raw        {entry['raw']}")
        print(f"  calibrated {entry['calibrated']}")
        for row in entry["bands"]:
            print(f"    {row}")

    # Pooled across all tested seasons, by raw band.
    allf = pd.concat([frames[s] for s in seasons[1:]], ignore_index=True)
    report["pooled_bands_raw"] = _band_table(allf, None)
    print("\nPOOLED out-of-sample seasons, by RAW band:")
    for row in report["pooled_bands_raw"]:
        print(f"    {row}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"draw_calibration_{ts}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
