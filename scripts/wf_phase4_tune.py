"""Phase 4 — focused tune on the survivors from Phase 3.

Phase 3 narrowed the winning levers to two:
  - DOW filter (banning Mon, Fri, and optionally Sun)
  - Max-EV cap (best at 0.50)

This script sweeps those two dimensions plus a few add-ons (October ban,
min_prob nudges), to pick the most ROBUST combination — highest profit AND
lowest DD — rather than just the peak of one window.

Output: stdout + data/diagnostics/wf_phase4_tune_<ts>.json
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
    min_prob=0.30,
    enable_simultaneous_correction=True,
    skip_late_season=True,
    skip_home_title_race=False,
    odds_source="Max",
    detect_source="PS",
    market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
)


def _max_drawdown_pct(bet_log: pd.DataFrame, initial: float) -> float:
    if bet_log.empty:
        return 0.0
    series = pd.concat([pd.Series([initial]),
                        bet_log["Bankroll"].astype(float)], ignore_index=True)
    peaks = series.cummax()
    dd = (peaks - series) / peaks
    return float(dd.max() * 100)


# Build the variant grid: DOW × max_ev × (Oct ban) × min_prob
VARIANTS: list[dict] = []

dow_options = [
    ("dowMF",   {"Mon", "Fri"}),
    ("dowMFS",  {"Mon", "Fri", "Sun"}),
    ("dowOnly_Sun", {"Sun"}),
    ("dowMFSat",   {"Mon", "Fri", "Sat"}),  # sanity check — should be terrible
    ("dowNone", set()),
]

ev_caps = [None, 1.0, 0.80, 0.60, 0.50, 0.40]

oct_options = [False, True]

minprob_options = [0.30, 0.32, 0.35]

for dow_name, dow_set in dow_options:
    for ev in ev_caps:
        for ban_oct in oct_options:
            for mp in minprob_options:
                # Skip degenerate baseline-equivalent variants except the one
                # explicit baseline
                if (dow_name == "dowNone" and ev is None
                        and not ban_oct and mp == 0.30):
                    name = "Z_baseline"
                else:
                    name = (f"{dow_name}"
                            f"_ev{int(ev*100) if ev is not None else 'INF'}"
                            f"_oct{'X' if ban_oct else '_'}"
                            f"_mp{int(mp*100)}")
                overrides = {}
                if dow_set:
                    overrides["banned_dows"] = dow_set
                if ban_oct:
                    overrides["banned_months"] = {"Oct"}
                if ev is not None:
                    overrides["max_ev_pct"] = ev
                if mp != 0.30:
                    overrides["min_prob"] = mp
                VARIANTS.append({"name": name, "overrides": overrides})


def main(test_weeks: int = 52) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{datetime.now():%H:%M:%S}] Loading + fitting…  ({len(VARIANTS)} variants)")
    df = load_data()
    df_features = add_rolling_features(df)
    bt_df = backtest_models(df, df_features, test_weeks=test_weeks)
    bin_variances = {
        "H": compute_per_bin_variance(bt_df, "H"),
        "D": compute_per_bin_variance(bt_df, "D"),
        "A": compute_per_bin_variance(bt_df, "A"),
    }

    results: list[dict] = []
    for v in VARIANTS:
        name = v["name"]
        cfg = {**COMMON, **v["overrides"]}
        log_df, summary = pf.ev_backtest_simulate_v2(
            bt_df, df, bin_variances=bin_variances, **cfg,
        )
        if "error" in summary:
            results.append({"name": name, "error": summary["error"]})
            continue
        max_dd = _max_drawdown_pct(log_df, COMMON["initial_bankroll"])
        results.append({
            "name":       name,
            "overrides":  {k: list(v) if isinstance(v, set) else v
                           for k, v in v["overrides"].items()},
            "final":      summary["final"],
            "profit":     summary["profit"],
            "roi_pct":    summary["roi"],
            "n_bets":     summary["n_bets"],
            "win_rate":   summary["win_rate"],
            "max_dd_pct": round(max_dd, 2),
        })

    # Persist
    json_path = OUT_DIR / f"wf_phase4_tune_{ts}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_weeks":   test_weeks,
        "common_cfg":   {k: list(v) if isinstance(v, set) else v
                         for k, v in COMMON.items()},
        "n_variants":   len(VARIANTS),
        "results":      results,
    }, indent=2, default=str))
    print(f"\n[{datetime.now():%H:%M:%S}] JSON: {json_path.relative_to(ROOT)}")

    valid = [r for r in results if "error" not in r]

    # ── Top 20 by final bankroll ───────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"TOP 20 BY FINAL BANKROLL  (out of {len(valid)} variants)")
    print("=" * 100)
    print(f"{'rank':>4}  {'variant':40s}  {'final £':>10}  "
          f"{'ROI%':>7}  {'n':>3}  {'win%':>5}  {'DD%':>6}")
    print("-" * 100)
    for i, r in enumerate(sorted(valid, key=lambda x: x["final"], reverse=True)[:20], 1):
        print(f"{i:>4}  {r['name']:40s}  £{r['final']:>9,.0f}  "
              f"{r['roi_pct']:>+6.1f}%  {r['n_bets']:>3}  "
              f"{r['win_rate']:>4.1f}%  {r['max_dd_pct']:>5.1f}%")

    # ── Top 20 by combined score (final / DD ratio) ────────────────────
    def _score(r): return r["final"] / max(r["max_dd_pct"], 1.0)
    print("\n" + "=" * 100)
    print("TOP 20 BY ROBUSTNESS SCORE (final £ / max DD%)")
    print("=" * 100)
    print(f"{'rank':>4}  {'variant':40s}  {'score':>8}  {'final £':>10}  "
          f"{'DD%':>6}  {'n':>3}  {'ROI%':>7}")
    print("-" * 100)
    for i, r in enumerate(sorted(valid, key=_score, reverse=True)[:20], 1):
        print(f"{i:>4}  {r['name']:40s}  {_score(r):>7,.0f}  "
              f"£{r['final']:>9,.0f}  {r['max_dd_pct']:>5.1f}%  "
              f"{r['n_bets']:>3}  {r['roi_pct']:>+6.1f}%")

    # ── Sweet-spot: ≥£100k AND ≤30% DD ─────────────────────────────────
    sweet = [r for r in valid if r["final"] >= 100_000 and r["max_dd_pct"] <= 30.0]
    if sweet:
        print("\n" + "=" * 100)
        print(f"SWEET SPOT (final ≥ £100k AND DD ≤ 30%) — {len(sweet)} matches")
        print("=" * 100)
        for r in sorted(sweet, key=lambda x: x["final"], reverse=True):
            print(f"  {r['name']:40s}  £{r['final']:>10,.0f}  "
                  f"DD {r['max_dd_pct']:>5.1f}%  ROI {r['roi_pct']:>+6.1f}%  "
                  f"n={r['n_bets']}")

    return 0


if __name__ == "__main__":
    sys.exit(main(test_weeks=52))
