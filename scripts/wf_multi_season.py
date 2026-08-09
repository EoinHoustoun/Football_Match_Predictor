"""Multi-season walk-forward validation of the Phase 4 winning config.

The Phase 4 tune was a single 52-week window — risk of overfitting to the
2025-26 regime. This script runs the same simulator on prior seasons by
slicing the dataset to end at each season's last match, then `backtest_models`
naturally trains on prior data and tests on that season.

For each season we compare:
  - v0_baseline       → no Phase 4 filters (current Mock Two minus the ban filters)
  - v_deployed        → dowMF + ev≤1.00 + octX + mp=0.32 (deployed config)
  - v_conservative    → dowMFS + ev≤0.50 + octX + mp=0.32 (lower-DD alt)
  - v_dow_only        → just dowMF (isolates DOW lever)
  - v_oct_only        → just ban Oct (isolates October lever)
  - v_maxev_only      → just ev≤1.00 cap (isolates EV cap lever)

If `v_deployed` ranks well across multiple seasons, the config is real.
If it only wins on 2025-26, we deployed an overfit.

Output: stdout + data/diagnostics/wf_multi_season_<ts>.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
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


VARIANTS: list[dict] = [
    {"name": "v0_baseline",
     "overrides": {"min_prob": 0.30}},
    {"name": "v_deployed (dowMF+ev1+oct+mp32)",
     "overrides": {"min_prob": 0.32,
                   "banned_dows": {"Mon", "Fri"},
                   "banned_months": {"Oct"},
                   "max_ev_pct": 1.00}},
    {"name": "v_conservative (dowMFS+ev0.5+oct+mp32)",
     "overrides": {"min_prob": 0.32,
                   "banned_dows": {"Mon", "Fri", "Sun"},
                   "banned_months": {"Oct"},
                   "max_ev_pct": 0.50}},
    {"name": "v_dow_only (dowMF)",
     "overrides": {"min_prob": 0.30,
                   "banned_dows": {"Mon", "Fri"}}},
    {"name": "v_oct_only (oct)",
     "overrides": {"min_prob": 0.30,
                   "banned_months": {"Oct"}}},
    {"name": "v_maxev_only (ev1)",
     "overrides": {"min_prob": 0.30,
                   "max_ev_pct": 1.00}},
]


def _max_drawdown_pct(bet_log: pd.DataFrame, initial: float) -> float:
    if bet_log.empty:
        return 0.0
    series = pd.concat([pd.Series([initial]),
                        bet_log["Bankroll"].astype(float)], ignore_index=True)
    peaks = series.cummax()
    dd = (peaks - series) / peaks
    return float(dd.max() * 100)


def _season_label(d: pd.Timestamp) -> str:
    """August-onward = same year start; pre-August = previous season."""
    if d.month >= 8:
        return f"{d.year}-{str(d.year+1)[-2:]}"
    return f"{d.year-1}-{str(d.year)[-2:]}"


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading full dataset…")
    df_full = load_data()
    df_full = df_full.copy()
    df_full["SeasonLbl"] = df_full["Date"].apply(_season_label)
    df_features_full = add_rolling_features(df_full)

    seasons = sorted(df_full["SeasonLbl"].unique())
    print(f"  Seasons in data: {seasons}")
    # Need ≥ 1 prior season to train on. Skip the first (2021-22) since the
    # backtest_models train window would be empty.
    test_seasons = seasons[1:]
    print(f"  Validating on: {test_seasons}\n")

    season_results: dict[str, list[dict]] = {}

    for season in test_seasons:
        print(f"[{datetime.now():%H:%M:%S}] === Season {season} ===")
        # End of this season — last match date
        season_end = df_full[df_full["SeasonLbl"] == season]["Date"].max()
        df_slice  = df_full[df_full["Date"] <= season_end].copy()
        ftr_slice = df_features_full[df_features_full["Date"] <= season_end].copy()
        # test_weeks = full season ≈ 38 match weeks; cap at 42 to be safe
        # Use 40 weeks so we cover August → end of season cleanly
        test_weeks = 40

        bt_df = backtest_models(df_slice, ftr_slice, test_weeks=test_weeks)
        if bt_df.empty:
            print(f"  → empty bt_df, skipping")
            continue
        # Restrict bt_df to matches actually in this season (defensive)
        bt_df_season = bt_df[bt_df["Date"].apply(_season_label) == season].copy()
        if bt_df_season.empty:
            print(f"  → no matches in season after filter, skipping")
            continue
        print(f"  bt_df rows in season: {len(bt_df_season)}  "
              f"(span {bt_df_season['Date'].min().date()} → "
              f"{bt_df_season['Date'].max().date()})")

        bin_variances = {
            "H": compute_per_bin_variance(bt_df_season, "H"),
            "D": compute_per_bin_variance(bt_df_season, "D"),
            "A": compute_per_bin_variance(bt_df_season, "A"),
        }

        rows = []
        for v in VARIANTS:
            cfg = {**COMMON, **v["overrides"]}
            log_df, summary = pf.ev_backtest_simulate_v2(
                bt_df_season, df_slice, bin_variances=bin_variances, **cfg,
            )
            if "error" in summary:
                rows.append({"name": v["name"], "error": summary["error"]})
                continue
            max_dd = _max_drawdown_pct(log_df, COMMON["initial_bankroll"])
            rows.append({
                "name":     v["name"],
                "final":    summary["final"],
                "profit":   summary["profit"],
                "roi":      summary["roi"],
                "n_bets":   summary["n_bets"],
                "win_rate": summary["win_rate"],
                "max_dd":   round(max_dd, 2),
            })
            print(f"  {v['name']:48s}  £{summary['final']:>9,.0f}  "
                  f"ROI {summary['roi']:>+6.1f}%  "
                  f"n={summary['n_bets']:>3}  win {summary['win_rate']:>4.1f}%  "
                  f"DD {max_dd:>5.1f}%")

        season_results[season] = rows
        print()

    # ── Persist ────────────────────────────────────────────────────────
    json_path = OUT_DIR / f"wf_multi_season_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seasons":      test_seasons,
        "common_cfg":   {k: list(v) if isinstance(v, set) else v
                         for k, v in COMMON.items()},
        "results":      season_results,
    }, indent=2, default=str))
    print(f"[{datetime.now():%H:%M:%S}] JSON: {json_path.relative_to(ROOT)}")

    # ── Per-variant rank stability ─────────────────────────────────────
    print("\n" + "=" * 110)
    print("RANK STABILITY (rank within each season; lower is better)")
    print("=" * 110)
    var_names = [v["name"] for v in VARIANTS]
    rank_matrix: dict[str, dict[str, int]] = {n: {} for n in var_names}
    for season, rows in season_results.items():
        valid = [r for r in rows if "error" not in r]
        ranked = sorted(valid, key=lambda r: r["final"], reverse=True)
        for i, r in enumerate(ranked, 1):
            rank_matrix[r["name"]][season] = i

    # Header
    header = f"{'variant':50s}  " + "  ".join(f"{s:>10s}" for s in test_seasons) + "  | mean rank | mean £"
    print(header)
    print("-" * len(header))
    rows_for_print = []
    for n in var_names:
        ranks = [rank_matrix[n].get(s, "—") for s in test_seasons]
        finals = [next((r["final"] for r in season_results.get(s, []) if r["name"] == n), 0)
                  for s in test_seasons]
        mean_rank = (sum(r for r in ranks if isinstance(r, int))
                     / max(1, sum(1 for r in ranks if isinstance(r, int))))
        mean_final = sum(finals) / len(finals)
        rows_for_print.append((n, ranks, mean_rank, mean_final, finals))

    # Sort by mean rank
    rows_for_print.sort(key=lambda x: x[2])
    for n, ranks, mr, mf, finals in rows_for_print:
        rank_str = "  ".join(f"{r if isinstance(r, int) else '—':>10}" for r in ranks)
        print(f"{n:50s}  {rank_str}  | {mr:>9.1f} | £{mf:>8,.0f}")

    # ── Per-season final bankrolls ─────────────────────────────────────
    print("\n" + "=" * 110)
    print("FINAL BANKROLLS BY SEASON (£)")
    print("=" * 110)
    print(f"{'variant':50s}  " + "  ".join(f"{s:>10s}" for s in test_seasons))
    print("-" * 110)
    for n in var_names:
        finals = [next((r["final"] for r in season_results.get(s, []) if r["name"] == n), 0)
                  for s in test_seasons]
        finals_str = "  ".join(f"£{f:>9,.0f}" for f in finals)
        print(f"{n:50s}  {finals_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
