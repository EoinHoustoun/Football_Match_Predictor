"""Comprehensive Main sweep — ELO / DOW / Month / MaxEV combinations,
scored on BOTH:
  (a) 52-week single window (matches what the user sees when they hit Run
      in the Main backtest UI)
  (b) 4-season per-season validation (multi-season robustness)

Goal: find any config that pushes the 52w final toward £100k WITHOUT
crashing on multi-season. The 4-season grid earlier found zero such
configs but used 40-week per-season slices; this sweep ALSO scores
the continuous 52w window the user actually runs in the UI.
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
    """Trimmed grid for tractable runtime: DOW × ELO floor × ELO gap min × ELO gap max × max_ev × min_prob."""
    dow_options    = [("DOWnone", None), ("DOWmf", {"Mon", "Fri"})]
    min_elo_options = [None, 1500, 1550]
    gap_min_options = [None, 80, 100]
    gap_max_options = [None, 250, 350]
    max_ev_options  = [None, 1.00, 1.50]
    minprob_options = [0.30, 0.32]

    variants = []
    for ((dow_name, dow_set), min_elo, gap_min, gap_max, max_ev, mp) in product(
            dow_options, min_elo_options, gap_min_options, gap_max_options,
            max_ev_options, minprob_options):
        if (gap_min is not None and gap_max is not None
                and gap_min >= gap_max):
            continue
        name_parts = [dow_name, f"mp{int(mp*100)}"]
        if min_elo is not None: name_parts.append(f"eFloor{min_elo}")
        if gap_min is not None: name_parts.append(f"gMin{gap_min}")
        if gap_max is not None: name_parts.append(f"gMax{gap_max}")
        if max_ev is not None:  name_parts.append(f"evMax{int(max_ev*100)}")
        name = "_".join(name_parts)
        overrides = {"min_prob": mp}
        if dow_set: overrides["banned_dows"] = dow_set
        if min_elo is not None: overrides["min_team_elo"] = min_elo
        if gap_min is not None: overrides["elo_gap_min"]  = gap_min
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

    # ── 52w single-window (matches UI Run button on test_weeks=52) ──
    print(f"[{datetime.now():%H:%M:%S}] Running 52w continuous backtest_models…")
    bt52 = backtest_models(df, df_features, test_weeks=52)
    print(f"  bt_df rows: {len(bt52)}")

    # ── Per-season cache ───────────────────────────────────────────
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
        print(f"  {season}: {len(bt_season)} rows")

    print(f"\n[{datetime.now():%H:%M:%S}] Sweeping {len(variants)} variants × "
          f"({1+len(season_data)} sims each)")

    results: list[dict] = []
    for i, v in enumerate(variants, 1):
        cfg = {**COMMON, **v["overrides"]}

        # 52w single window
        log52, sum52 = pf.ev_backtest_simulate(
            bt52, df, df_features=df_features, **cfg,
        )
        if "error" in sum52:
            single_52w = 0.0; single_dd = 100.0; single_n = 0
        else:
            single_52w = float(sum52["final"])
            single_dd  = _max_drawdown_pct(log52, COMMON["initial_bankroll"])
            single_n   = int(sum52["n_bets"])

        # Per-season validation
        season_finals: dict[str, float] = {}
        for season, (df_slice, ftr_slice, bt_season) in season_data.items():
            log_, sum_ = pf.ev_backtest_simulate(
                bt_season, df_slice, df_features=ftr_slice, **cfg,
            )
            season_finals[season] = (0.0 if "error" in sum_
                                      else float(sum_["final"]))

        if not season_finals:
            continue
        finals_list = list(season_finals.values())
        median_ms = sorted(finals_list)[len(finals_list) // 2]
        min_ms    = min(finals_list)
        max_ms    = max(finals_list)
        results.append({
            "name":       v["name"],
            "overrides":  {k: list(v) if isinstance(v, set) else v
                           for k, v in v["overrides"].items()},
            "single_52w":   round(single_52w, 0),
            "single_n":     single_n,
            "single_dd":    round(single_dd, 2),
            "season_finals": season_finals,
            "ms_median":    round(median_ms, 0),
            "ms_min":       round(min_ms, 0),
            "ms_max":       round(max_ms, 0),
        })
        if i % 50 == 0:
            print(f"  [{datetime.now():%H:%M:%S}] {i}/{len(variants)}")

    # Persist
    json_path = OUT_DIR / f"wf_main_full_sweep_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seasons":      list(season_data.keys()),
        "n_variants":   len(variants),
        "results":      results,
    }, indent=2, default=str))
    print(f"\n[{datetime.now():%H:%M:%S}] JSON: {json_path.relative_to(ROOT)}")

    seasons_sorted = list(season_data.keys())

    # ── Top 20 by 52w (what user sees) ─────────────────────────────
    print("\n" + "=" * 130)
    print("TOP 20 BY 52-WEEK SINGLE WINDOW (= what you see hitting Run on test_weeks=52)")
    print("=" * 130)
    print(f"{'rank':>4}  {'variant':52s}  {'52w':>10}  {'msMed':>9}  {'msMin':>9}  {'n':>4}  {'DD%':>5}")
    print("-" * 130)
    by_52w = sorted(results, key=lambda r: r["single_52w"], reverse=True)
    for i, r in enumerate(by_52w[:20], 1):
        flag = " 💀" if r["ms_min"] < 5_000 else ("" if r["ms_min"] >= 10_000 else " ⚠")
        print(f"{i:>4}  {r['name']:52s}  £{r['single_52w']:>8,.0f}  "
              f"£{r['ms_median']:>7,.0f}  £{r['ms_min']:>7,.0f}  "
              f"{r['single_n']:>4}  {r['single_dd']:>4.1f}%{flag}")

    # ── Sweet spot: 52w ≥ £80k AND ms_min ≥ £5k ───────────────────
    sweet = [r for r in results if r["single_52w"] >= 80_000 and r["ms_min"] >= 5_000]
    print("\n" + "=" * 130)
    print(f"SWEET SPOT (52w ≥ £80k AND ms_min ≥ £5k) — {len(sweet)} survive")
    print("=" * 130)
    if sweet:
        for r in sorted(sweet, key=lambda x: x["single_52w"], reverse=True)[:15]:
            seasons_str = "  ".join(f"£{r['season_finals'].get(s, 0):>7,.0f}"
                                    for s in seasons_sorted)
            print(f"  {r['name']:52s}  52w £{r['single_52w']:>8,.0f}  "
                  f"msMed £{r['ms_median']:>7,.0f}  | {seasons_str}")
    else:
        print("  No config achieved BOTH £80k 52w AND £5k floor.")

    # ── £100k+ on 52w ──────────────────────────────────────────────
    over_100k = [r for r in results if r["single_52w"] >= 100_000]
    print("\n" + "=" * 130)
    print(f"£100k+ ON 52W — {len(over_100k)} variants")
    print("=" * 130)
    if over_100k:
        for r in sorted(over_100k, key=lambda x: x["single_52w"], reverse=True)[:15]:
            seasons_str = "  ".join(f"£{r['season_finals'].get(s, 0):>7,.0f}"
                                    for s in seasons_sorted)
            print(f"  {r['name']:52s}  52w £{r['single_52w']:>8,.0f}  "
                  f"msMin £{r['ms_min']:>7,.0f}  | {seasons_str}")
    else:
        print("  No variant reached £100k on 52w continuous window.")

    # ── Current Main baseline ──────────────────────────────────────
    cur = next((r for r in results if r["name"] == "DOWnone_mp30"), None)
    if cur:
        print("\n" + "=" * 130)
        print("CURRENT MAIN BASELINE (no v2 filters)")
        print("=" * 130)
        print(f"  52w £{cur['single_52w']:,.0f}  ms_median £{cur['ms_median']:,.0f}  "
              f"ms_min £{cur['ms_min']:,.0f}  n={cur['single_n']}  DD={cur['single_dd']:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
