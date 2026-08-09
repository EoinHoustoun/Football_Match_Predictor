"""Constrained random search:
  - min_ev ≥ 40% (hard floor)
  - n_bets ≥ 30 (must place enough bets)
  - Honest calibration (no leakage)
  - 52-week single window (matches what user runs in UI)

Find config with highest final bankroll that meets both constraints.
Then re-test top 5 across 4 prior seasons for robustness.
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

N_SAMPLES = 500
SEED = 7
random.seed(SEED)

MIN_N_BETS  = 30   # must have at least this many bets
MIN_EV_PCT  = 40.0 # min_ev gate must be ≥ 40%

COMMON = dict(
    initial_bankroll=10_000.0,
    allowed_markets={"D", "under25"},
    enable_simultaneous_correction=True,
    skip_late_season=True,
    skip_home_title_race=False,
    odds_source="Max",
    detect_source="PS",
    market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
)


def _sample_variant(i: int) -> dict:
    return {
        "id":       f"v{i:03d}",
        "mp":       round(random.uniform(0.15, 0.40), 3),
        "ev_pct":   round(random.uniform(MIN_EV_PCT, 80.0), 1),
        "kelly":    random.choice([0.25, 0.50, 0.75, 1.0]),
        "max_stake": random.choice([0.15, 0.25, 0.33, 0.50]),
        "dows":     random.choice([None, {"Mon", "Fri"}, {"Mon"},
                                    {"Fri"}, {"Sun"}, {"Mon", "Fri", "Sun"}]),
        "gap_min":  random.choice([None, 50, 80, 100, 150]),
        "gap_max":  random.choice([None, 200, 250, 350, 500]),
        "min_elo":  random.choice([None, 1450, 1500, 1550, 1600]),
        "max_ev":   random.choice([None, 0.80, 1.00, 1.50, 2.00]),
    }


def _max_drawdown_pct(log: pd.DataFrame, init: float) -> float:
    if log.empty: return 0.0
    s = pd.concat([pd.Series([init]), log["Bankroll"].astype(float)],
                  ignore_index=True)
    return float(((s.cummax() - s) / s.cummax()).max() * 100)


def _season_label(d: pd.Timestamp) -> str:
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading…")
    df = load_data()
    df = df.copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    ftr = add_rolling_features(df)

    print(f"[{datetime.now():%H:%M:%S}] backtest_models 52w + honest calibrator (10w prior)…")
    bt_52w = backtest_models(df, ftr, test_weeks=52)
    bt_cal = backtest_models(df, ftr, test_weeks=10)
    cal_52w = pf.fit_calibrators_from_backtest(bt_cal)

    print(f"\n[{datetime.now():%H:%M:%S}] Sweeping {N_SAMPLES} variants on 52w with honest calibration…")
    variants = [_sample_variant(i) for i in range(N_SAMPLES)]

    results: list[dict] = []
    for i, v in enumerate(variants, 1):
        log, sum_ = pf.ev_backtest_simulate(
            bt_52w, df, df_features=ftr,
            min_ev_pct=v["ev_pct"],
            kelly_frac=v["kelly"],
            max_stake_pct=v["max_stake"],
            min_prob=v["mp"],
            banned_dows=v["dows"],
            elo_gap_min=v["gap_min"],
            elo_gap_max=v["gap_max"],
            min_team_elo=v["min_elo"],
            max_ev_pct=v["max_ev"],
            calibrators=cal_52w,
            **COMMON,
        )
        if "error" in sum_:
            continue
        n = int(sum_["n_bets"])
        # Hard constraint
        if n < MIN_N_BETS:
            continue
        dd = _max_drawdown_pct(log, COMMON["initial_bankroll"])
        results.append({
            "id":       v["id"],
            "config": {
                "mp": v["mp"], "ev_pct": v["ev_pct"],
                "kelly": v["kelly"], "max_stake": v["max_stake"],
                "dows": sorted(v["dows"]) if v["dows"] else None,
                "gap_min": v["gap_min"], "gap_max": v["gap_max"],
                "min_elo": v["min_elo"], "max_ev": v["max_ev"],
            },
            "final":    float(sum_["final"]),
            "roi":      float(sum_["roi"]),
            "n_bets":   n,
            "win_rate": float(sum_["win_rate"]),
            "max_dd":   round(dd, 2),
        })
        if i % 50 == 0:
            print(f"  [{datetime.now():%H:%M:%S}] {i}/{N_SAMPLES}  "
                  f"({len(results)} qualifying so far)")

    print(f"\n[{datetime.now():%H:%M:%S}] {len(results)}/{N_SAMPLES} variants met "
          f"BOTH constraints (n≥{MIN_N_BETS} AND ev≥{MIN_EV_PCT}%)")

    # Top 20 by final
    print("\n" + "=" * 130)
    print(f"TOP 20 QUALIFYING CONFIGS (n≥{MIN_N_BETS}, ev≥{MIN_EV_PCT}%, honest calibration, 52w)")
    print("=" * 130)
    print(f"{'rank':>4}  {'id':>5}  {'final':>10}  {'ROI':>7}  {'n':>4}  "
          f"{'win%':>5}  {'DD%':>5}  {'mp':>5}  {'ev':>5}  {'kelly':>5}")
    print("-" * 130)
    by_final = sorted(results, key=lambda r: r["final"], reverse=True)
    for rank, r in enumerate(by_final[:20], 1):
        c = r["config"]
        print(f"{rank:>4}  {r['id']:>5}  £{r['final']:>9,.0f}  {r['roi']:>+6.1f}%  "
              f"{r['n_bets']:>4}  {r['win_rate']:>4.1f}%  {r['max_dd']:>4.1f}%  "
              f"{c['mp']:>5.2f}  {c['ev_pct']:>5.1f}  {c['kelly']:>5.2f}")

    # Top 5 with full configs
    print("\n" + "=" * 130)
    print("TOP 5 — full configs")
    print("=" * 130)
    for rank, r in enumerate(by_final[:5], 1):
        c = r["config"]
        print(f"\nRank {rank}  ({r['id']}):  £{r['final']:,.0f}  ROI {r['roi']:+.1f}%  "
              f"n={r['n_bets']}  win {r['win_rate']:.1f}%  DD {r['max_dd']}%")
        print(f"  mp={c['mp']}  ev={c['ev_pct']}%  kelly={c['kelly']}  "
              f"max_stake={c['max_stake']}")
        print(f"  dows={c['dows']}  gap_min={c['gap_min']}  gap_max={c['gap_max']}  "
              f"min_elo={c['min_elo']}  max_ev={c['max_ev']}")

    # ── Multi-season re-validation of top 5 ──────────────────────────
    if by_final[:5]:
        print(f"\n[{datetime.now():%H:%M:%S}] Multi-season re-validation of top 5…")
        seasons = sorted(df["SeasonLbl"].unique())[1:]
        season_pack: dict[str, tuple] = {}
        for season in seasons:
            season_end = df[df["SeasonLbl"] == season]["Date"].max()
            season_start = df[df["SeasonLbl"] == season]["Date"].min()
            df_slice = df[df["Date"] <= season_end].copy()
            ftr_slice = ftr[ftr["Date"] <= season_end].copy()
            bt_s = backtest_models(df_slice, ftr_slice, test_weeks=40)
            if bt_s.empty: continue
            bt_season = bt_s[bt_s["Date"].apply(_season_label) == season].copy()
            if bt_season.empty: continue
            # Honest calibrator: data BEFORE season_start
            pre = df_slice[df_slice["Date"] < season_start]
            pre_ft = ftr_slice[ftr_slice["Date"] < season_start]
            if len(pre) < 200:
                continue
            cal_s = pf.fit_calibrators_from_backtest(
                backtest_models(pre, pre_ft, test_weeks=10))
            season_pack[season] = (df_slice, ftr_slice, bt_season, cal_s)

        print("\n" + "=" * 130)
        print("MULTI-SEASON VALIDATION (top 5 from 52w, tested on 4 prior seasons each)")
        print("=" * 130)
        hdr_seasons = "  ".join(f"{s:>10}" for s in season_pack.keys())
        print(f"{'rank':>4}  {'id':>5}  {hdr_seasons}  {'median':>9}  {'min':>9}")
        print("-" * 130)
        for rank, r in enumerate(by_final[:5], 1):
            c = r["config"]
            finals = {}
            for season, (df_slice, ftr_slice, bt_season, cal_s) in season_pack.items():
                log_, sum_ = pf.ev_backtest_simulate(
                    bt_season, df_slice, df_features=ftr_slice,
                    min_ev_pct=c["ev_pct"],
                    kelly_frac=c["kelly"],
                    max_stake_pct=c["max_stake"],
                    min_prob=c["mp"],
                    banned_dows=set(c["dows"]) if c["dows"] else None,
                    elo_gap_min=c["gap_min"],
                    elo_gap_max=c["gap_max"],
                    min_team_elo=c["min_elo"],
                    max_ev_pct=c["max_ev"],
                    calibrators=cal_s,
                    **COMMON,
                )
                finals[season] = (0 if "error" in sum_ else float(sum_["final"]))
            flist = list(finals.values())
            med = sorted(flist)[len(flist) // 2] if flist else 0
            mn  = min(flist) if flist else 0
            season_str = "  ".join(f"£{finals.get(s, 0):>8,.0f}" for s in season_pack.keys())
            print(f"{rank:>4}  {r['id']:>5}  {season_str}  £{med:>7,.0f}  £{mn:>7,.0f}")

    # Persist
    out_path = OUT_DIR / f"constrained_search_{ts}.json"
    out_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": N_SAMPLES,
        "constraints": {"min_n_bets": MIN_N_BETS, "min_ev_pct": MIN_EV_PCT},
        "results": results,
    }, indent=2, default=str))
    print(f"\nWROTE: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
