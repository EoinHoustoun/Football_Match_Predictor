"""Deep ELO-gap multi-season sweep — Eoin's intuition: filtering by team-level
ELO floor throws away David-vs-Goliath upset draws (which pay 3.50–5.00+).
Try ELO gap filtering INSTEAD of team-level floor.

Grid axes (no min_team_elo at all — let weak teams stay in if the gap is right):
  - DOW filter:   none / Mon+Fri
  - ELO gap min:  none / 0 / 30 / 50 / 80 / 100 / 150  (skip too-close)
  - ELO gap max:  none / 150 / 200 / 250 / 300 / 350 / 400 / 500 / 600  (skip too-lopsided)
  - Min prob:     0.30 / 0.32 / 0.35

Reports two leaderboards:
  - Top 30 by raw median (peak-chasers — Eoin wants £100k+ here)
  - Top 30 by robust score (median × min — same as previous sweep)
Plus a "candidate set" filter showing configs with median ≥ £50k AND min ≥ £5k.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                 # noqa: E402
import portfolio as pf                             # noqa: E402
from portfolio import compute_per_bin_variance     # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)


COMMON = dict(
    min_ev_pct=40.0,
    base_kelly_frac=1.0,
    max_stake_pct=0.33,
    initial_bankroll=10_000.0,
    allowed_markets={"D", "under25"},
    enable_simultaneous_correction=True,
    skip_late_season=True,
    skip_home_title_race=False,
    odds_source="Max",
    detect_source="PS",
    market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
)


def _build_grid() -> list[dict]:
    dow_options = [
        ("DOWnone", None),
        ("DOWmf",   {"Mon", "Fri"}),
    ]
    gap_min_options = [None, 0, 30, 50, 80, 100, 150]
    gap_max_options = [None, 150, 200, 250, 300, 350, 400, 500, 600]
    minprob_options = [0.30, 0.32, 0.35]

    variants = []
    for ((dow_name, dow_set), gap_min, gap_max, mp) in product(
            dow_options, gap_min_options, gap_max_options, minprob_options):
        # Skip impossible bands
        if gap_min is not None and gap_max is not None and gap_min >= gap_max:
            continue
        name = f"{dow_name}_mp{int(mp*100)}"
        if gap_min is not None: name += f"_gMin{gap_min}"
        if gap_max is not None: name += f"_gMax{gap_max}"
        overrides = {"min_prob": mp}
        if dow_set:  overrides["banned_dows"] = dow_set
        if gap_min is not None: overrides["elo_gap_min"] = gap_min
        if gap_max is not None: overrides["elo_gap_max"] = gap_max
        variants.append({"name": name, "overrides": overrides})
    return variants


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
    variants = _build_grid()
    print(f"[{datetime.now():%H:%M:%S}] Loading data + grid: {len(variants)} variants")
    df = load_data()
    df = df.copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    df_features = add_rolling_features(df)
    seasons = sorted(df["SeasonLbl"].unique())[1:]
    print(f"  Seasons: {seasons}")

    season_data: dict[str, tuple] = {}
    for season in seasons:
        season_end = df[df["SeasonLbl"] == season]["Date"].max()
        df_slice  = df[df["Date"] <= season_end].copy()
        ftr_slice = df_features[df_features["Date"] <= season_end].copy()
        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty: continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty: continue
        bv = {m: compute_per_bin_variance(bt_season, m) for m in ("H", "D", "A")}
        season_data[season] = (df_slice, ftr_slice, bt_season, bv)
        print(f"  {season}: {len(bt_season)} rows fitted")

    print(f"\n[{datetime.now():%H:%M:%S}] Sweeping {len(variants)} variants × "
          f"{len(season_data)} seasons = {len(variants)*len(season_data)} sims")

    results: list[dict] = []
    for i, v in enumerate(variants, 1):
        finals: dict[str, float] = {}
        n_bets: dict[str, int] = {}
        max_dd = 0.0
        for season, (df_slice, ftr_slice, bt_season, bv) in season_data.items():
            cfg = {**COMMON, **v["overrides"]}
            log_, sum_ = pf.ev_backtest_simulate_v2(
                bt_season, df_slice, bin_variances=bv,
                df_features=ftr_slice, **cfg,
            )
            if "error" in sum_:
                finals[season] = 0.0; n_bets[season] = 0
            else:
                finals[season] = float(sum_["final"])
                n_bets[season] = int(sum_["n_bets"])
                max_dd = max(max_dd, _max_drawdown_pct(log_, COMMON["initial_bankroll"]))

        if not finals: continue
        finals_list = list(finals.values())
        median_f = sorted(finals_list)[len(finals_list) // 2]
        min_f    = min(finals_list)
        max_f    = max(finals_list)
        mean_f   = sum(finals_list) / len(finals_list)
        total_n  = sum(n_bets.values())
        robust_score = median_f * min_f / 1e8
        results.append({
            "name":         v["name"],
            "overrides":    {k: list(v) if isinstance(v, set) else v
                             for k, v in v["overrides"].items()},
            "finals":       finals,
            "median":       round(median_f, 0),
            "mean":         round(mean_f, 0),
            "min":          round(min_f, 0),
            "max":          round(max_f, 0),
            "max_dd":       round(max_dd, 2),
            "total_n":      total_n,
            "robust_score": round(robust_score, 2),
        })
        if i % 30 == 0:
            print(f"  [{datetime.now():%H:%M:%S}] {i}/{len(variants)} done")

    json_path = OUT_DIR / f"wf_elo_gap_deep_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seasons":      list(season_data.keys()),
        "n_variants":   len(variants),
        "results":      results,
    }, indent=2, default=str))
    print(f"\n[{datetime.now():%H:%M:%S}] JSON: {json_path.relative_to(ROOT)}")

    seasons_sorted = list(season_data.keys())

    # ── TOP 30 BY MEDIAN (Eoin's £100k target) ─────────────────────────
    print("\n" + "=" * 130)
    print("TOP 30 BY RAW MEDIAN (peak-chasers — target £100k+)")
    print("=" * 130)
    print(f"{'rank':>4}  {'variant':50s}  {'median':>9}  {'min':>9}  {'max':>9}  {'mean':>9}  {'totN':>5}  {'DD%':>5}")
    print("-" * 130)
    by_median = sorted(results, key=lambda r: r["median"], reverse=True)
    for i, r in enumerate(by_median[:30], 1):
        crash_flag = " 💀" if r["min"] < 5_000 else ("" if r["min"] >= 10_000 else " ⚠")
        print(f"{i:>4}  {r['name']:50s}  £{r['median']:>7,.0f}  "
              f"£{r['min']:>7,.0f}  £{r['max']:>7,.0f}  "
              f"£{r['mean']:>7,.0f}  {r['total_n']:>5}  {r['max_dd']:>4.1f}%{crash_flag}")

    # ── TOP 30 BY ROBUST SCORE (consistency-weighted) ──────────────────
    print("\n" + "=" * 130)
    print("TOP 30 BY ROBUST SCORE (median × min — penalises crashes)")
    print("=" * 130)
    print(f"{'rank':>4}  {'variant':50s}  {'score':>8}  {'median':>9}  {'min':>9}  {'max':>9}")
    print("-" * 130)
    for i, r in enumerate(sorted(results, key=lambda r: r["robust_score"], reverse=True)[:30], 1):
        print(f"{i:>4}  {r['name']:50s}  {r['robust_score']:>8,.0f}  "
              f"£{r['median']:>7,.0f}  £{r['min']:>7,.0f}  £{r['max']:>7,.0f}")

    # ── CANDIDATES: median ≥ £50k AND min ≥ £5k ────────────────────────
    cand = [r for r in results if r["median"] >= 50_000 and r["min"] >= 5_000]
    print("\n" + "=" * 130)
    print(f"CANDIDATES (median ≥ £50k AND min ≥ £5k) — {len(cand)} survive")
    print("=" * 130)
    if cand:
        for r in sorted(cand, key=lambda x: x["median"], reverse=True)[:20]:
            finals_str = "  ".join(f"£{r['finals'].get(s, 0):>7,.0f}"
                                    for s in seasons_sorted)
            print(f"  {r['name']:50s}  median £{r['median']:>7,.0f}  "
                  f"min £{r['min']:>6,.0f}  | {finals_str}")
    else:
        print("  No variants meet the £50k median + £5k floor threshold.")

    # ── 100K+ MEDIAN: any? ──────────────────────────────────────────────
    over_100k = [r for r in results if r["median"] >= 100_000]
    print("\n" + "=" * 130)
    print(f"£100k+ MEDIAN VARIANTS — {len(over_100k)} found")
    print("=" * 130)
    if over_100k:
        for r in sorted(over_100k, key=lambda x: x["median"], reverse=True):
            finals_str = "  ".join(f"£{r['finals'].get(s, 0):>7,.0f}"
                                    for s in seasons_sorted)
            print(f"  {r['name']:50s}  median £{r['median']:>7,.0f}  "
                  f"min £{r['min']:>6,.0f}  | {finals_str}")
    else:
        print("  No variants reached £100k median across 4 seasons.")
        print("  Reality: with £10k start + Kelly compounding, £100k median requires "
              "10× per season consistently — EVERY season above £100k.")
        print("  Best config so far on median:")
        if by_median:
            r = by_median[0]
            print(f"    {r['name']}  median £{r['median']:,.0f}  "
                  f"min £{r['min']:,.0f}  max £{r['max']:,.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
