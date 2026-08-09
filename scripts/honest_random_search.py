"""HONEST random search for the best Main config.

Key statistical fix vs prior sweeps:
  - Calibrator is fit on data ENDING BEFORE the test window (no leakage).
  - Each variant is tested across 4 prior seasons, not one window.
  - Scored by median × min (robust to single-season luck).

Parameter space (random sample):
  - min_prob:        Uniform[0.15, 0.40]
  - min_ev_pct:      Uniform[3, 50]
  - kelly_fraction:  Choice{0.25, 0.50, 0.75, 1.0}
  - max_stake_pct:   Choice{0.15, 0.25, 0.33, 0.50}
  - banned_dows:     Choice{None, {Mon,Fri}, {Mon}, {Fri}, {Sun}, {Mon,Fri,Sun}}
  - elo_gap_min:     Choice{None, 50, 80, 100, 150}
  - elo_gap_max:     Choice{None, 200, 250, 350, 500}
  - min_team_elo:    Choice{None, 1450, 1500, 1550, 1600}
  - max_ev_pct:      Choice{None, 0.50, 0.80, 1.00, 1.50}

N=200 random variants × 4 seasons = 800 honest backtest runs.
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

N_SAMPLES = 200
SEED = 42
random.seed(SEED)


COMMON = dict(
    initial_bankroll=10_000.0,
    allowed_markets={"D", "under25"},
    enable_simultaneous_correction=True,
    skip_late_season=True,           # validated Mar-Apr 0/7
    skip_home_title_race=False,
    odds_source="Max",
    detect_source="PS",
    market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
)


def _sample_variant(i: int) -> dict:
    """Random sample one configuration."""
    mp = round(random.uniform(0.15, 0.40), 3)
    ev = round(random.uniform(3.0, 50.0), 1)
    kelly = random.choice([0.25, 0.50, 0.75, 1.0])
    max_stake = random.choice([0.15, 0.25, 0.33, 0.50])
    dow_options = [None, {"Mon", "Fri"}, {"Mon"}, {"Fri"}, {"Sun"},
                   {"Mon", "Fri", "Sun"}]
    dows = random.choice(dow_options)
    gap_min = random.choice([None, 50, 80, 100, 150])
    gap_max = random.choice([None, 200, 250, 350, 500])
    min_elo = random.choice([None, 1450, 1500, 1550, 1600])
    max_ev = random.choice([None, 0.50, 0.80, 1.00, 1.50])

    if gap_min is not None and gap_max is not None and gap_min >= gap_max:
        gap_max = None  # skip impossible band

    return {
        "id": f"v{i:03d}",
        "mp": mp, "ev_pct": ev,
        "kelly": kelly, "max_stake": max_stake,
        "dows": dows, "gap_min": gap_min, "gap_max": gap_max,
        "min_elo": min_elo, "max_ev": max_ev,
    }


def _max_drawdown_pct(log: pd.DataFrame, init: float) -> float:
    if log.empty: return 0.0
    s = pd.concat([pd.Series([init]), log["Bankroll"].astype(float)],
                  ignore_index=True)
    return float(((s.cummax() - s) / s.cummax()).max() * 100)


def _season_label(d: pd.Timestamp) -> str:
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def _build_honest_calibrator(df_slice: pd.DataFrame,
                              df_features_slice: pd.DataFrame,
                              test_start_date) -> dict:
    """Fit calibrator using ONLY data BEFORE test_start_date.

    This is the honest calibration — no leakage into the test window.
    Uses a 10-week backtest_models on data up to test_start_date.
    """
    pre_test = df_slice[df_slice["Date"] < test_start_date].copy()
    pre_test_ft = df_features_slice[df_features_slice["Date"] < test_start_date].copy()
    if len(pre_test) < 200:  # need enough training data
        return {}
    try:
        bt_pre = backtest_models(pre_test, pre_test_ft, test_weeks=10)
        return pf.fit_calibrators_from_backtest(bt_pre)
    except Exception:
        return {}


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading data…")
    df = load_data()
    df = df.copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    df_features = add_rolling_features(df)
    seasons = sorted(df["SeasonLbl"].unique())[1:]
    print(f"  Seasons: {seasons}")

    print(f"\n[{datetime.now():%H:%M:%S}] Pre-computing per-season fixtures "
          f"(backtest_models + honest calibrator)…")
    season_pack: dict[str, tuple] = {}
    for season in seasons:
        season_end = df[df["SeasonLbl"] == season]["Date"].max()
        season_start = df[df["SeasonLbl"] == season]["Date"].min()
        df_slice  = df[df["Date"] <= season_end].copy()
        ftr_slice = df_features[df_features["Date"] <= season_end].copy()
        # Test window: just this season
        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty: continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty: continue
        # Honest calibrator: fit on data BEFORE season_start
        cal = _build_honest_calibrator(df_slice, ftr_slice, season_start)
        season_pack[season] = (df_slice, ftr_slice, bt_season, cal)
        print(f"  {season}: bt rows {len(bt_season):>4}  "
              f"calibrator markets fit: {list(cal.keys()) if cal else 'NONE'}")

    print(f"\n[{datetime.now():%H:%M:%S}] Random search: {N_SAMPLES} variants "
          f"× {len(season_pack)} seasons = {N_SAMPLES * len(season_pack)} honest sims")
    print()

    variants = [_sample_variant(i) for i in range(N_SAMPLES)]

    results: list[dict] = []
    for i, v in enumerate(variants, 1):
        finals: dict[str, float] = {}
        max_dd_overall = 0.0
        total_n = 0
        for season, (df_slice, ftr_slice, bt_season, cal) in season_pack.items():
            log, sum_ = pf.ev_backtest_simulate(
                bt_season, df_slice, df_features=ftr_slice,
                min_ev_pct=v["ev_pct"],
                kelly_frac=v["kelly"],
                max_stake_pct=v["max_stake"],
                min_prob=v["mp"],
                banned_dows=v["dows"],
                elo_gap_min=v["gap_min"],
                elo_gap_max=v["gap_max"],
                min_team_elo=v["min_elo"],
                max_ev_pct=v["max_ev"],
                calibrators=cal,
                **COMMON,
            )
            if "error" in sum_:
                finals[season] = 0.0
            else:
                finals[season] = float(sum_["final"])
                total_n += int(sum_["n_bets"])
                max_dd_overall = max(max_dd_overall,
                                     _max_drawdown_pct(log, COMMON["initial_bankroll"]))

        if not finals:
            continue
        flist = list(finals.values())
        median_f = sorted(flist)[len(flist) // 2]
        min_f    = min(flist)
        max_f    = max(flist)
        mean_f   = sum(flist) / len(flist)
        robust   = median_f * min_f / 1e8

        results.append({
            "id":         v["id"],
            "config": {
                "mp": v["mp"], "ev_pct": v["ev_pct"],
                "kelly": v["kelly"], "max_stake": v["max_stake"],
                "dows": sorted(v["dows"]) if v["dows"] else None,
                "gap_min": v["gap_min"], "gap_max": v["gap_max"],
                "min_elo": v["min_elo"], "max_ev": v["max_ev"],
            },
            "finals":   finals,
            "median":   round(median_f, 0),
            "min":      round(min_f, 0),
            "max":      round(max_f, 0),
            "mean":     round(mean_f, 0),
            "max_dd":   round(max_dd_overall, 2),
            "total_n":  total_n,
            "robust_score": round(robust, 4),
        })
        if i % 20 == 0:
            print(f"  [{datetime.now():%H:%M:%S}] {i:>3}/{N_SAMPLES}")

    out_path = OUT_DIR / f"honest_random_search_{ts}.json"
    out_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples":    N_SAMPLES,
        "seed":         SEED,
        "seasons":      list(season_pack.keys()),
        "results":      results,
    }, indent=2, default=str))
    print(f"\nWROTE: {out_path.relative_to(ROOT)}")

    seasons_sorted = list(season_pack.keys())

    # ── Top 20 by ROBUST SCORE (median × min — penalises crashes) ───
    print("\n" + "=" * 130)
    print("TOP 20 BY ROBUST SCORE (median × min)")
    print("=" * 130)
    print(f"{'rank':>4}  {'id':>5}  {'median':>9}  {'min':>9}  {'max':>9}  {'mean':>9}  "
          f"{'totN':>5}  {'mp':>5}  {'ev%':>5}  {'kelly':>5}  {'stk':>4}")
    print("-" * 130)
    for i, r in enumerate(sorted(results, key=lambda x: x["robust_score"], reverse=True)[:20], 1):
        c = r["config"]
        print(f"{i:>4}  {r['id']:>5}  "
              f"£{r['median']:>7,.0f}  £{r['min']:>7,.0f}  £{r['max']:>7,.0f}  "
              f"£{r['mean']:>7,.0f}  {r['total_n']:>5}  "
              f"{c['mp']:>5.2f}  {c['ev_pct']:>5.1f}  "
              f"{c['kelly']:>5.2f}  {c['max_stake']:>4.2f}")

    # ── Top 5 with full config ──────────────────────────────────────
    print("\n" + "=" * 130)
    print("TOP 5 BY ROBUST SCORE — full configs + per-season breakdown")
    print("=" * 130)
    top5 = sorted(results, key=lambda x: x["robust_score"], reverse=True)[:5]
    for rank, r in enumerate(top5, 1):
        c = r["config"]
        season_str = "  ".join(f"{s}=£{r['finals'].get(s, 0):>7,.0f}"
                                for s in seasons_sorted)
        print(f"\nRank {rank}  ({r['id']}):  robust={r['robust_score']:,.2f}")
        print(f"  median £{r['median']:,.0f}  min £{r['min']:,.0f}  "
              f"max £{r['max']:,.0f}  total_bets {r['total_n']}  DD {r['max_dd']}%")
        print(f"  PARAMS:  mp={c['mp']}  ev={c['ev_pct']}%  kelly={c['kelly']}  "
              f"max_stake={c['max_stake']}")
        print(f"           dows={c['dows']}  gap_min={c['gap_min']}  "
              f"gap_max={c['gap_max']}  min_elo={c['min_elo']}  max_ev={c['max_ev']}")
        print(f"  SEASONS: {season_str}")

    # ── Top 5 by MEDIAN (peak chasers — likely overfit on one season) ─
    print("\n" + "=" * 130)
    print("TOP 5 BY RAW MEDIAN  (warning: may be one-season fits)")
    print("=" * 130)
    by_median = sorted(results, key=lambda x: x["median"], reverse=True)[:5]
    for rank, r in enumerate(by_median, 1):
        c = r["config"]
        print(f"{rank}.  median £{r['median']:,.0f}  min £{r['min']:,.0f}  "
              f"max £{r['max']:,.0f}  "
              f"mp={c['mp']} ev={c['ev_pct']} k={c['kelly']} "
              f"dows={c['dows']} max_ev={c['max_ev']}")

    # ── Honest recommendation ─────────────────────────────────────────
    print("\n" + "=" * 130)
    print("HONEST RECOMMENDATION")
    print("=" * 130)
    if top5:
        winner = top5[0]
        c = winner["config"]
        print(f"\nBest robust config (id={winner['id']}):")
        print(f"  median £{winner['median']:,.0f}, worst-season £{winner['min']:,.0f}, "
              f"best-season £{winner['max']:,.0f}")
        print(f"  Params: mp={c['mp']}, ev={c['ev_pct']}%, kelly={c['kelly']}, "
              f"max_stake={c['max_stake']}")
        print(f"          dows={c['dows']}, gap_min={c['gap_min']}, "
              f"gap_max={c['gap_max']}, min_elo={c['min_elo']}, max_ev={c['max_ev']}")
        print(f"  Total bets across 4 seasons: {winner['total_n']}  "
              f"(~{winner['total_n']/4:.0f}/season)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
