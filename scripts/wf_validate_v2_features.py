"""Phase 3 — walk-forward validate Mock Two Phase 2 features.

Runs `ev_backtest_simulate_v2` over the last 52 weeks of match data with
multiple filter configurations layered on top of the K-N + uncertainty-Kelly +
sim-correction stack. Reports final bankroll / ROI / DD per variant so we can
keep only filters that actually transfer OOS.

The sim itself is causally rolling — each bet date sees the model fitted on
prior weeks (via backtest_models), and the team-ROI filter only sees bets
already settled within the sim. So the variant comparison is honestly
out-of-sample.

Output: stdout + data/diagnostics/wf_v2_features_<ts>.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from data import load_data, add_rolling_features       # noqa: E402
from models import backtest_models                     # noqa: E402
import portfolio as pf                                  # noqa: E402
from portfolio import compute_per_bin_variance          # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)


# Common settings shared across all v2 variants (mirrors current live config)
COMMON = dict(
    min_ev_pct=40.0,           # +40% EV gate
    base_kelly_frac=1.0,       # full Kelly
    max_stake_pct=0.33,        # cap at 33% bankroll
    initial_bankroll=10_000.0,
    allowed_markets={"D", "under25"},
    min_prob=0.30,
    enable_simultaneous_correction=True,
    skip_late_season=True,
    skip_home_title_race=False,
    odds_source="Max",
    detect_source="PS",
    market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
)


# Variants to evaluate (overrides applied on top of COMMON)
VARIANTS: list[dict] = [
    # ── Baselines ────────────────────────────────────────────────────────
    {"name": "v0_baseline_v2",   "overrides": {}},

    # ── Single-feature variants ──────────────────────────────────────────
    {"name": "v1_team_roi_-25",
     "overrides": {"team_roi_filter": True,
                   "team_roi_threshold_pct": -25.0,
                   "team_roi_min_n": 3}},
    {"name": "v1_team_roi_-50",
     "overrides": {"team_roi_filter": True,
                   "team_roi_threshold_pct": -50.0,
                   "team_roi_min_n": 4}},

    {"name": "v2_max_ev_080",
     "overrides": {"max_ev_pct": 0.80}},
    {"name": "v2_max_ev_060",
     "overrides": {"max_ev_pct": 0.60}},
    {"name": "v2_max_ev_050",
     "overrides": {"max_ev_pct": 0.50}},

    {"name": "v3_dow_MonFri",
     "overrides": {"banned_dows": {"Mon", "Fri"}}},
    {"name": "v3_dow_MonFriSun",
     "overrides": {"banned_dows": {"Mon", "Fri", "Sun"}}},

    {"name": "v4_dd_throttle_20pct",
     "overrides": {"enable_drawdown_throttle": True,
                   "drawdown_at_pct": 0.20,
                   "drawdown_min_factor": 0.25}},
    {"name": "v4_dd_throttle_30pct",
     "overrides": {"enable_drawdown_throttle": True,
                   "drawdown_at_pct": 0.30,
                   "drawdown_min_factor": 0.30}},

    {"name": "v5_minprob_035",
     "overrides": {"min_prob": 0.35}},
    {"name": "v5_minprob_040",
     "overrides": {"min_prob": 0.40}},

    # ── Stacked variants — best singles combined ─────────────────────────
    {"name": "v6_safe_stack",
     "overrides": {
         "team_roi_filter": True, "team_roi_threshold_pct": -25.0,
         "team_roi_min_n": 3,
         "max_ev_pct": 0.80,
         "enable_drawdown_throttle": True,
         "drawdown_at_pct": 0.20, "drawdown_min_factor": 0.25,
     }},
    {"name": "v6_aggressive_stack",
     "overrides": {
         "team_roi_filter": True, "team_roi_threshold_pct": -25.0,
         "team_roi_min_n": 3,
         "max_ev_pct": 0.80,
         "banned_dows": {"Mon", "Fri", "Sun"},
         "enable_drawdown_throttle": True,
         "drawdown_at_pct": 0.20, "drawdown_min_factor": 0.25,
         "min_prob": 0.35,
     }},
    {"name": "v6_minimal_stack",
     "overrides": {
         "max_ev_pct": 0.80,
         "enable_drawdown_throttle": True,
         "drawdown_at_pct": 0.20, "drawdown_min_factor": 0.25,
     }},
    {"name": "v6_team_only_drawdown",
     "overrides": {
         "team_roi_filter": True, "team_roi_threshold_pct": -25.0,
         "team_roi_min_n": 3,
         "enable_drawdown_throttle": True,
         "drawdown_at_pct": 0.20, "drawdown_min_factor": 0.25,
     }},
]


def _max_drawdown_pct(bet_log: pd.DataFrame, initial: float) -> float:
    if bet_log.empty:
        return 0.0
    series = pd.concat([pd.Series([initial]),
                        bet_log["Bankroll"].astype(float)], ignore_index=True)
    peaks = series.cummax()
    dd = (peaks - series) / peaks
    return float(dd.max() * 100)


def main(test_weeks: int = 52) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{datetime.now():%H:%M:%S}] Loading data + fitting models…")
    df = load_data()
    df_features = add_rolling_features(df)

    # Single backtest_models call shared across all variants — keeps comparison
    # apples-to-apples (same OOS predictions, only the gating layer changes)
    bt_df = backtest_models(df, df_features, test_weeks=test_weeks)
    if bt_df.empty:
        print("EMPTY backtest output — aborting")
        return 1
    print(f"[{datetime.now():%H:%M:%S}] backtest_models produced {len(bt_df)} match rows")

    # Bin variances: precompute once on full bt_df (same input for all variants)
    bin_variances = {
        "H": compute_per_bin_variance(bt_df, "H"),
        "D": compute_per_bin_variance(bt_df, "D"),
        "A": compute_per_bin_variance(bt_df, "A"),
    }

    results: list[dict] = []

    for v in VARIANTS:
        name = v["name"]
        cfg = {**COMMON, **v["overrides"]}
        print(f"\n[{datetime.now():%H:%M:%S}] Running variant: {name}")
        log_df, summary = pf.ev_backtest_simulate_v2(
            bt_df, df,
            bin_variances=bin_variances,
            **cfg,
        )
        if "error" in summary:
            print(f"  → ERROR: {summary['error']}")
            results.append({"name": name, "error": summary["error"]})
            continue
        max_dd = _max_drawdown_pct(log_df, COMMON["initial_bankroll"])
        results.append({
            "name":          name,
            "overrides":     {k: list(v) if isinstance(v, set) else v
                              for k, v in v["overrides"].items()},
            "final":         summary["final"],
            "profit":        summary["profit"],
            "roi_pct":       summary["roi"],
            "n_bets":        summary["n_bets"],
            "win_rate":      summary["win_rate"],
            "max_dd_pct":    round(max_dd, 2),
            "skipped_min_prob":   summary.get("skipped_min_prob"),
            "skipped_calendar":   summary.get("skipped_calendar"),
            "skipped_team_roi":   summary.get("skipped_team_roi"),
            "skipped_max_ev":     summary.get("skipped_max_ev"),
            "skipped_late_season": summary.get("skipped_late_season"),
            "mean_shrinkage":  summary.get("mean_shrinkage"),
            "mean_sim_factor": summary.get("mean_sim_factor"),
            "mean_dd_factor":  summary.get("mean_dd_factor"),
        })
        print(f"  → £{summary['final']:,.0f}  "
              f"(profit £{summary['profit']:+,.0f},  "
              f"ROI {summary['roi']:+.1f}%,  "
              f"n={summary['n_bets']},  "
              f"win {summary['win_rate']}%,  "
              f"DD {max_dd:.1f}%)")

    # Persist
    json_path = OUT_DIR / f"wf_v2_features_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_weeks":   test_weeks,
        "common_cfg":   {k: list(v) if isinstance(v, set) else v
                         for k, v in COMMON.items()},
        "results":      results,
    }, indent=2, default=str))
    print(f"\n[{datetime.now():%H:%M:%S}] JSON written: {json_path.relative_to(ROOT)}")

    # ── Leaderboard ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("LEADERBOARD (sorted by final bankroll)")
    print("=" * 90)
    sorted_r = sorted([r for r in results if "error" not in r],
                      key=lambda r: r["final"], reverse=True)
    print(f"{'rank':>4}  {'variant':30s}  {'final £':>10}  "
          f"{'profit':>10}  {'ROI%':>7}  {'n':>3}  {'win%':>5}  {'DD%':>6}")
    print("-" * 90)
    for i, r in enumerate(sorted_r, 1):
        print(f"{i:>4}  {r['name']:30s}  £{r['final']:>9,.0f}  "
              f"£{r['profit']:>+9,.0f}  {r['roi_pct']:>+6.1f}%  "
              f"{r['n_bets']:>3}  {r['win_rate']:>4.1f}%  {r['max_dd_pct']:>5.1f}%")

    # ── Filter survival: which variants beat baseline ───────────────────
    baseline = next((r for r in results if r["name"] == "v0_baseline_v2"), None)
    if baseline:
        print("\n" + "=" * 90)
        print(f"BASELINE (Mock Two no-new-filters): "
              f"£{baseline['final']:,.0f}  (DD {baseline['max_dd_pct']:.1f}%)")
        print("=" * 90)
        print("Variants that BEAT the baseline (higher final AND lower DD):")
        for r in sorted_r:
            if r["name"] == "v0_baseline_v2":
                continue
            beat_profit = r["final"] > baseline["final"]
            beat_dd     = r["max_dd_pct"] < baseline["max_dd_pct"]
            tag = ("✅ both " if beat_profit and beat_dd
                   else "💰 profit" if beat_profit
                   else "🛡 DD-only" if beat_dd
                   else "❌ worse")
            delta_pct = (r["final"] - baseline["final"]) / baseline["final"] * 100
            print(f"  {tag:8s}  {r['name']:30s}  Δ £{r['final']-baseline['final']:>+8,.0f}  "
                  f"({delta_pct:>+5.1f}%)  DD {r['max_dd_pct']-baseline['max_dd_pct']:+5.1f}pp")

    return 0


if __name__ == "__main__":
    sys.exit(main(test_weeks=52))
