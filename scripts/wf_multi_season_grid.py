"""Multi-season grid sweep — find the absolute best Mock Two config that's
NOT overfit. Tests every variant on all 4 prior seasons and ranks by a
robustness score that penalises crashing on any single season.

Grid axes:
  - DOW filter:  none / Mon+Fri
  - ELO gap max: none / 200 / 300 / 400 (skip lopsided)
  - ELO gap min: none / 50 / 100 (skip too-close)
  - Min team ELO: none / 1500 / 1550 / 1600 (skip weak teams)
  - Min prob:    0.30 / 0.32 / 0.35

Robust score: `median_final × min_season_final / 1e8` — rewards configs
that perform well *consistently*, not configs that one season fluked into
£600k while another flopped to £1k.

Output: stdout + data/diagnostics/wf_multi_season_grid_<ts>.json
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
    """Cartesian product of axes. Skip degenerate combos (gap_min ≥ gap_max)."""
    dow_options = [
        ("DOWnone", None),
        ("DOWmf",   {"Mon", "Fri"}),
    ]
    gap_max_options  = [None, 200, 300, 400]
    gap_min_options  = [None, 50, 100]
    min_elo_options  = [None, 1500, 1550, 1600]
    minprob_options  = [0.30, 0.32, 0.35]

    variants = []
    for ((dow_name, dow_set), gap_max, gap_min, min_elo, mp) in product(
            dow_options, gap_max_options, gap_min_options,
            min_elo_options, minprob_options):
        # Skip impossible bands
        if gap_min is not None and gap_max is not None and gap_min >= gap_max:
            continue
        name_parts = [dow_name, f"mp{int(mp*100)}"]
        if gap_max is not None: name_parts.append(f"gMax{gap_max}")
        if gap_min is not None: name_parts.append(f"gMin{gap_min}")
        if min_elo is not None: name_parts.append(f"eFloor{min_elo}")
        name = "_".join(name_parts)

        overrides = {"min_prob": mp}
        if dow_set:  overrides["banned_dows"]  = dow_set
        if gap_max:  overrides["elo_gap_max"]  = gap_max
        if gap_min:  overrides["elo_gap_min"]  = gap_min
        if min_elo:  overrides["min_team_elo"] = min_elo
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

    # Cache the per-season backtest_models output once — same input across all variants
    season_data: dict[str, tuple] = {}
    for season in seasons:
        season_end = df[df["SeasonLbl"] == season]["Date"].max()
        df_slice  = df[df["Date"] <= season_end].copy()
        ftr_slice = df_features[df_features["Date"] <= season_end].copy()
        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty:
            continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty:
            continue
        bv = {m: compute_per_bin_variance(bt_season, m) for m in ("H", "D", "A")}
        season_data[season] = (df_slice, ftr_slice, bt_season, bv)
        print(f"  {season}: {len(bt_season)} rows fitted")

    # Run grid × seasons
    print(f"\n[{datetime.now():%H:%M:%S}] Sweeping {len(variants)} variants × "
          f"{len(season_data)} seasons = {len(variants)*len(season_data)} sims")

    results: list[dict] = []
    for i, v in enumerate(variants, 1):
        finals: dict[str, float] = {}
        dds: dict[str, float] = {}
        n_bets: dict[str, int] = {}
        for season, (df_slice, ftr_slice, bt_season, bv) in season_data.items():
            cfg = {**COMMON, **v["overrides"]}
            log_, sum_ = pf.ev_backtest_simulate_v2(
                bt_season, df_slice, bin_variances=bv,
                df_features=ftr_slice, **cfg,
            )
            if "error" in sum_:
                finals[season] = 0.0
                dds[season]    = 100.0
                n_bets[season] = 0
            else:
                finals[season] = float(sum_["final"])
                dds[season]    = _max_drawdown_pct(log_, COMMON["initial_bankroll"])
                n_bets[season] = int(sum_["n_bets"])

        if not finals:
            continue
        finals_list = list(finals.values())
        median_f  = sorted(finals_list)[len(finals_list) // 2]
        min_f     = min(finals_list)
        max_f     = max(finals_list)
        mean_f    = sum(finals_list) / len(finals_list)
        max_dd    = max(dds.values())
        total_n   = sum(n_bets.values())
        # Robust score: median × min — penalises crashing on any single season
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
        if i % 20 == 0:
            print(f"  [{datetime.now():%H:%M:%S}] {i}/{len(variants)} done")

    # Persist
    json_path = OUT_DIR / f"wf_multi_season_grid_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seasons":      list(season_data.keys()),
        "n_variants":   len(variants),
        "results":      results,
    }, indent=2, default=str))
    print(f"\n[{datetime.now():%H:%M:%S}] JSON: {json_path.relative_to(ROOT)}")

    # ── Top 20 by robust score ─────────────────────────────────────────
    print("\n" + "=" * 110)
    print("TOP 20 BY ROBUST SCORE  (median × min, penalises crashes)")
    print("=" * 110)
    seasons_sorted = list(season_data.keys())
    hdr = f"{'rank':>4}  {'variant':45s}  {'score':>7}  {'median':>9}  {'min':>9}  {'max':>9}  {'totN':>5}"
    print(hdr); print("-" * len(hdr))
    sorted_r = sorted(results, key=lambda r: r["robust_score"], reverse=True)[:20]
    for i, r in enumerate(sorted_r, 1):
        print(f"{i:>4}  {r['name']:45s}  {r['robust_score']:>7,.0f}  "
              f"£{r['median']:>7,.0f}  £{r['min']:>7,.0f}  £{r['max']:>7,.0f}  "
              f"{r['total_n']:>5}")

    # ── Top 10 with per-season breakdown ───────────────────────────────
    print("\n" + "=" * 110)
    print("TOP 10 PER-SEASON BREAKDOWN")
    print("=" * 110)
    s_hdr = f"{'rank':>4}  {'variant':45s}  " + "  ".join(f"{s:>9}" for s in seasons_sorted)
    print(s_hdr); print("-" * len(s_hdr))
    for i, r in enumerate(sorted_r[:10], 1):
        finals_str = "  ".join(f"£{r['finals'].get(s, 0):>7,.0f}" for s in seasons_sorted)
        print(f"{i:>4}  {r['name']:45s}  {finals_str}")

    # ── Highlight: best with constraint min ≥ £5k (no crashes) ────────
    no_crash = [r for r in results if r["min"] >= 5_000]
    if no_crash:
        print("\n" + "=" * 110)
        print(f"BEST WITH NO-CRASH CONSTRAINT (min season ≥ £5,000) — {len(no_crash)} survive")
        print("=" * 110)
        for r in sorted(no_crash, key=lambda x: x["median"], reverse=True)[:10]:
            print(f"  {r['name']:45s}  median £{r['median']:>7,.0f}  "
                  f"min £{r['min']:>6,.0f}  max £{r['max']:>7,.0f}  "
                  f"DD {r['max_dd']:>5.1f}%  n={r['total_n']}")

    # ── Reference: current deployed config ─────────────────────────────
    cur = next((r for r in results if r["name"] == "DOWmf_mp32"), None)
    if cur:
        print("\n" + "=" * 110)
        print("CURRENT DEPLOYED CONFIG (DOWmf + mp=0.32, no ELO filters)")
        print("=" * 110)
        print(f"  median £{cur['median']:,.0f}  min £{cur['min']:,.0f}  "
              f"max £{cur['max']:,.0f}  total_n={cur['total_n']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
