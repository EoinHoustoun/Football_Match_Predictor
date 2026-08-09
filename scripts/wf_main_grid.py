"""Multi-season grid sweep for MAIN (Mock 1) — same approach as the
Mock Two grid that found min_team_elo=1500. Tests Main's stack with the
ported v2 filters (ELO, DOW, Month, max-EV cap) across 4 prior seasons.

Goal: find Main's best multi-season config. Score by both raw median
(peak-chasers) and robust score (median × min — penalises crashes).
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

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)


COMMON = dict(
    min_ev_pct=40.0,
    kelly_frac=1.0,
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
    """Sweep ELO floor + DOW filter + min_prob + max-EV cap."""
    dow_options = [
        ("DOWnone", None),
        ("DOWmf",   {"Mon", "Fri"}),
    ]
    min_elo_options = [None, 1450, 1500, 1550, 1600]
    gap_max_options = [None, 200, 300, 400]
    minprob_options = [0.30, 0.32, 0.35]
    max_ev_options  = [None, 0.50, 0.80, 1.00]

    variants = []
    for ((dow_name, dow_set), min_elo, gap_max, mp, max_ev) in product(
            dow_options, min_elo_options, gap_max_options,
            minprob_options, max_ev_options):
        name_parts = [dow_name, f"mp{int(mp*100)}"]
        if min_elo is not None: name_parts.append(f"eFloor{min_elo}")
        if gap_max is not None: name_parts.append(f"gMax{gap_max}")
        if max_ev is not None:  name_parts.append(f"evMax{int(max_ev*100)}")
        name = "_".join(name_parts)

        overrides = {"min_prob": mp}
        if dow_set:  overrides["banned_dows"]  = dow_set
        if min_elo is not None: overrides["min_team_elo"] = min_elo
        if gap_max is not None: overrides["elo_gap_max"]  = gap_max
        if max_ev is not None:  overrides["max_ev_pct"]   = max_ev
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
    print(f"[{datetime.now():%H:%M:%S}] Loading + grid: {len(variants)} variants")
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
        season_data[season] = (df_slice, ftr_slice, bt_season)
        print(f"  {season}: {len(bt_season)} rows fitted")

    print(f"\n[{datetime.now():%H:%M:%S}] Sweeping {len(variants)} × "
          f"{len(season_data)} seasons = {len(variants)*len(season_data)} sims")

    results: list[dict] = []
    for i, v in enumerate(variants, 1):
        finals: dict[str, float] = {}
        n_bets: dict[str, int] = {}
        max_dd = 0.0
        for season, (df_slice, ftr_slice, bt_season) in season_data.items():
            cfg = {**COMMON, **v["overrides"]}
            log_, sum_ = pf.ev_backtest_simulate(
                bt_season, df_slice,
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

    json_path = OUT_DIR / f"wf_main_grid_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seasons":      list(season_data.keys()),
        "n_variants":   len(variants),
        "results":      results,
    }, indent=2, default=str))
    print(f"\n[{datetime.now():%H:%M:%S}] JSON: {json_path.relative_to(ROOT)}")

    seasons_sorted = list(season_data.keys())

    # ── TOP 20 BY MEDIAN ─────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("TOP 20 BY RAW MEDIAN")
    print("=" * 130)
    print(f"{'rank':>4}  {'variant':50s}  {'median':>9}  {'min':>9}  {'max':>9}  {'totN':>5}  {'DD%':>5}")
    print("-" * 130)
    by_median = sorted(results, key=lambda r: r["median"], reverse=True)
    for i, r in enumerate(by_median[:20], 1):
        flag = " 💀" if r["min"] < 5_000 else ("" if r["min"] >= 10_000 else " ⚠")
        print(f"{i:>4}  {r['name']:50s}  £{r['median']:>7,.0f}  "
              f"£{r['min']:>7,.0f}  £{r['max']:>7,.0f}  "
              f"{r['total_n']:>5}  {r['max_dd']:>4.1f}%{flag}")

    # ── TOP 20 BY ROBUST SCORE ───────────────────────────────────────
    print("\n" + "=" * 130)
    print("TOP 20 BY ROBUST SCORE (median × min)")
    print("=" * 130)
    print(f"{'rank':>4}  {'variant':50s}  {'score':>8}  {'median':>9}  {'min':>9}  {'max':>9}")
    print("-" * 130)
    for i, r in enumerate(sorted(results, key=lambda r: r["robust_score"], reverse=True)[:20], 1):
        print(f"{i:>4}  {r['name']:50s}  {r['robust_score']:>8,.0f}  "
              f"£{r['median']:>7,.0f}  £{r['min']:>7,.0f}  £{r['max']:>7,.0f}")

    # ── CANDIDATES: median ≥ £40k AND min ≥ £5k ──────────────────────
    cand = [r for r in results if r["median"] >= 40_000 and r["min"] >= 5_000]
    print("\n" + "=" * 130)
    print(f"CANDIDATES (median ≥ £40k AND min ≥ £5k) — {len(cand)} survive")
    print("=" * 130)
    if cand:
        for r in sorted(cand, key=lambda x: x["median"], reverse=True)[:10]:
            finals_str = "  ".join(f"£{r['finals'].get(s, 0):>7,.0f}"
                                    for s in seasons_sorted)
            print(f"  {r['name']:50s}  median £{r['median']:>7,.0f}  "
                  f"min £{r['min']:>6,.0f}  | {finals_str}")
    else:
        print("  No variants meet the £40k median + £5k floor.")

    # ── PER-SEASON BREAKDOWN OF TOP 5 BY MEDIAN ──────────────────────
    print("\n" + "=" * 130)
    print("TOP 5 BY MEDIAN — per-season breakdown")
    print("=" * 130)
    print(f"{'rank':>4}  {'variant':50s}  " + "  ".join(f"{s:>9}" for s in seasons_sorted))
    print("-" * 130)
    for i, r in enumerate(by_median[:5], 1):
        finals_str = "  ".join(f"£{r['finals'].get(s, 0):>7,.0f}" for s in seasons_sorted)
        print(f"{i:>4}  {r['name']:50s}  {finals_str}")

    # ── CURRENT MAIN BASELINE (no Phase 4 filters) ──────────────────
    cur = next((r for r in results if r["name"] == "DOWnone_mp30"), None)
    if cur:
        print("\n" + "=" * 130)
        print("CURRENT MAIN BASELINE (no v2 filters)")
        print("=" * 130)
        print(f"  median £{cur['median']:,.0f}  min £{cur['min']:,.0f}  "
              f"max £{cur['max']:,.0f}  total_n={cur['total_n']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
