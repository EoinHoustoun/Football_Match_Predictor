"""Targeted research sweep: what backtest default settings would have reached
£100,000 — honestly?

Two framings, both with leak-free calibration (calibrator fit strictly before
each season):
  A. Single season from £10k: does ANY config reach £100k inside one season?
  B. Compounded: same config run season after season, bankroll carried over
     (stakes are % of bankroll, so growth compounds naturally).

Expands the space beyond honest_random_search_20260510: calibration on/off,
skip_late on/off, odds/detect source choices, lower prob gates. ~800 variants.

Run:  python3 scripts/sweep_100k.py
Out:  data/diagnostics/sweep_100k_<ts>.json
"""
from __future__ import annotations
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import load_data, add_rolling_features   # noqa: E402
from models import backtest_models                  # noqa: E402
import portfolio as pf                              # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)

N_SAMPLES = 800
SEED = 7
random.seed(SEED)


def _season_label(d: pd.Timestamp) -> str:
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def _build_honest_calibrator(df_slice, ftr_slice, test_start_date) -> dict:
    pre = df_slice[df_slice["Date"] < test_start_date].copy()
    pre_ft = ftr_slice[ftr_slice["Date"] < test_start_date].copy()
    if len(pre) < 200:
        return {}
    try:
        bt_pre = backtest_models(pre, pre_ft, test_weeks=10)
        return pf.fit_calibrators_from_backtest(bt_pre)
    except Exception:
        return {}


def _sample_variant(i: int) -> dict:
    return {
        "id":         f"s{i:03d}",
        "mp":         round(random.uniform(0.10, 0.40), 3),
        "ev_pct":     round(random.uniform(0.0, 50.0), 1),
        "kelly":      random.choice([0.50, 0.75, 1.0]),
        "max_stake":  random.choice([0.25, 0.33, 0.50]),
        "dows":       random.choice([None, {"Mon"}, {"Mon", "Fri"}]),
        "skip_late":  random.choice([True, False]),
        "use_cal":    random.choice([True, True, False]),   # 2:1 weighted on
        "odds_src":   random.choice(["Max", "Max", "B365"]),
        "detect_src": random.choice(["PS", None]),
        "markets":    random.choice([{"D", "under25"}, {"D"}]),
        "min_elo":    random.choice([None, 1500]),
        "max_ev":     random.choice([None, 1.0]),
    }


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading data…", flush=True)
    df = load_data().copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    df_features = add_rolling_features(df)
    seasons = sorted(df["SeasonLbl"].unique())[1:]   # drop first (no prior data)
    print(f"  Seasons: {seasons}", flush=True)

    season_pack: dict[str, tuple] = {}
    for season in seasons:
        season_end   = df[df["SeasonLbl"] == season]["Date"].max()
        season_start = df[df["SeasonLbl"] == season]["Date"].min()
        df_slice  = df[df["Date"] <= season_end].copy()
        ftr_slice = df_features[df_features["Date"] <= season_end].copy()
        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty:
            continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty:
            continue
        cal = _build_honest_calibrator(df_slice, ftr_slice, season_start)
        season_pack[season] = (df_slice, ftr_slice, bt_season, cal)
        print(f"  {season}: bt rows {len(bt_season):>4}  cal: "
              f"{sorted(cal.keys()) if cal else 'NONE'}", flush=True)

    ordered_seasons = sorted(season_pack.keys())
    print(f"\n[{datetime.now():%H:%M:%S}] Sweeping {N_SAMPLES} variants × "
          f"{len(ordered_seasons)} seasons (single-season + compounded)…", flush=True)

    def _run(v, season, bankroll):
        df_slice, ftr_slice, bt_season, cal = season_pack[season]
        mg = ({"under25": {"min_prob": 0.50, "min_ev": 0.05}}
              if "under25" in v["markets"] else None)
        log, sum_ = pf.ev_backtest_simulate(
            bt_season, df_slice, df_features=ftr_slice,
            initial_bankroll=bankroll,
            min_ev_pct=v["ev_pct"],
            kelly_frac=v["kelly"],
            max_stake_pct=v["max_stake"],
            min_prob=v["mp"],
            banned_dows=v["dows"],
            min_team_elo=v["min_elo"],
            max_ev_pct=v["max_ev"],
            calibrators=(cal if v["use_cal"] else None),
            allowed_markets=v["markets"],
            market_gates=mg,
            odds_source=v["odds_src"],
            detect_source=v["detect_src"],
            skip_late_season=v["skip_late"],
            skip_home_title_race=False,
            enable_simultaneous_correction=True,
        )
        if "error" in sum_:
            return None, 0
        return float(sum_["final"]), int(sum_["n_bets"])

    results = []
    for i in range(N_SAMPLES):
        v = _sample_variant(i)
        finals: dict[str, float] = {}
        total_n = 0
        for season in ordered_seasons:
            f, n = _run(v, season, 10_000.0)
            finals[season] = f if f is not None else 10_000.0
            total_n += n
        # Compounded path: carry bankroll across seasons (bust stops the chain)
        comp = 10_000.0
        for season in ordered_seasons:
            if comp < 100.0:
                break
            f, _ = _run(v, season, comp)
            if f is not None:
                comp = f
        flist = list(finals.values())
        results.append({
            "id": v["id"],
            "config": {k: (sorted(v[k]) if isinstance(v[k], set) else v[k])
                       for k in ("mp", "ev_pct", "kelly", "max_stake", "dows",
                                  "skip_late", "use_cal", "odds_src",
                                  "detect_src", "markets", "min_elo", "max_ev")},
            "finals": finals,
            "compounded": round(comp, 0),
            "best_season": round(max(flist), 0),
            "median": round(sorted(flist)[len(flist) // 2], 0),
            "min": round(min(flist), 0),
            "total_n": total_n,
        })
        if (i + 1) % 50 == 0:
            best = max(results, key=lambda r: r["compounded"])
            print(f"  {i+1}/{N_SAMPLES}  best compounded so far "
                  f"£{best['compounded']:,.0f} ({best['id']})", flush=True)

    out = OUT_DIR / f"sweep_100k_{ts}.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": N_SAMPLES, "seed": SEED,
        "seasons": ordered_seasons,
        "note": ("Honest calibration (fit pre-season). 'compounded' = bankroll "
                 "carried across seasons in chronological order from £10k."),
        "results": results,
    }, indent=1))
    print(f"\nSaved → {out}", flush=True)

    n100_single = sum(1 for r in results if r["best_season"] >= 100_000)
    n100_comp   = sum(1 for r in results if r["compounded"] >= 100_000)
    print(f"≥£100k single-season: {n100_single}/{N_SAMPLES} · "
          f"compounded: {n100_comp}/{N_SAMPLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
