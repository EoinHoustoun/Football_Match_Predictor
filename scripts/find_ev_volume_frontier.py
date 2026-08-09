"""Find the EV gate frontier — the highest min_ev that still produces
n_bets ≥ 30 in 52w with honest calibration. Sweeps min_ev from 5% to 50%
in 1% steps, holding other params at sensible defaults.

This answers: 'what's the highest EV we can ACTUALLY demand under honest
calibration while still placing enough bets?'
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                 # noqa: E402
import portfolio as pf                             # noqa: E402

print("Loading + honest calibrator…")
df = load_data()
ftr = add_rolling_features(df)
bt_52w = backtest_models(df, ftr, test_weeks=52)
cal = pf.fit_calibrators_from_backtest(backtest_models(df, ftr, test_weeks=10))
print(f"  bt rows: {len(bt_52w)}\n")

# Sweep min_ev with other params held at modest defaults
print(f"{'min_ev':>7}  {'final':>10}  {'ROI':>7}  {'n':>4}  {'win%':>5}")
print("-" * 50)
results = []
for ev in range(5, 51):
    log, sum_ = pf.ev_backtest_simulate(
        bt_52w, df, df_features=ftr,
        min_ev_pct=float(ev),
        kelly_frac=1.0,
        max_stake_pct=0.33,
        initial_bankroll=10_000.0,
        allowed_markets={"D", "under25"},
        min_prob=0.20,
        skip_late_season=True,
        odds_source="Max", detect_source="PS",
        enable_simultaneous_correction=True,
        market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
        calibrators=cal,
    )
    if "error" in sum_:
        continue
    results.append({
        "ev": ev, "final": sum_["final"], "roi": sum_["roi"],
        "n": sum_["n_bets"], "win": sum_["win_rate"],
    })
    marker = " ★" if sum_["n_bets"] >= 30 else ""
    print(f"{ev:>5}%  £{sum_['final']:>9,.0f}  {sum_['roi']:>+6.1f}%  "
          f"{sum_['n_bets']:>4}  {sum_['win_rate']:>4.1f}%{marker}")

print("\n★ = meets n≥30 constraint")
qualifying = [r for r in results if r["n"] >= 30]
if qualifying:
    print(f"\nHighest min_ev with n≥30: {max(r['ev'] for r in qualifying)}%")
    best = max(qualifying, key=lambda r: r["final"])
    print(f"Best qualifying config: min_ev={best['ev']}%, final=£{best['final']:,.0f}, "
          f"ROI={best['roi']:+.1f}%, n={best['n']}")
else:
    print("\nNo config with n≥30 found across the EV range tested.")
