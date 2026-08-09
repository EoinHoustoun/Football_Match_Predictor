"""Find a Main config that bets 4-5 times per week on 2025-26 (≈200 bets in 52w)
while still making profit. Sweeps gate looseness."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                 # noqa: E402
import portfolio as pf                             # noqa: E402

print("Loading...")
df = load_data()
df_features = add_rolling_features(df)
print(f"  {len(df)} matches. Fitting backtest_models (52w)…")
bt = backtest_models(df, df_features, test_weeks=52)
print(f"  bt rows: {len(bt)}\n")

calibrators = pf.fit_calibrators_from_backtest(bt)

VARIANTS = [
    # (name, min_ev_pct, min_prob, gMin100, evCap, dows, mkts, u25_gates)
    ("CURRENT aggressive",        40, 0.32, 100,  1.00, {"Mon","Fri"}, {"D","under25"}, True),
    ("Baseline (no v2 filters)",  40, 0.30, None, None, None,           {"D","under25"}, True),
    ("Loose EV 20%",              20, 0.25, None, None, None,           {"D","under25"}, True),
    ("Loose EV 10%",              10, 0.22, None, None, None,           {"D","under25"}, True),
    ("Very loose EV 5%",           5, 0.20, None, None, None,           {"D","under25"}, True),
    ("All markets, EV 5%",         5, 0.20, None, None, None,           {"D","H","A","over25","under25"}, True),
    ("All markets, EV 10%",       10, 0.22, None, None, None,           {"D","H","A","over25","under25"}, True),
    ("All markets, EV 20%",       20, 0.25, None, None, None,           {"D","H","A","over25","under25"}, True),
]

print(f"{'config':35s}  {'final':>10}  {'ROI':>7}  {'n':>4}  {'win%':>5}  {'/week':>6}  {'DD%':>5}")
print("-" * 90)
for (name, ev, mp, gmin, evcap, dows, mkts, u25g) in VARIANTS:
    market_gates = ({"under25": {"min_prob": 0.50, "min_ev": 0.05}} if u25g else None)
    log, sum_ = pf.ev_backtest_simulate(
        bt, df, df_features=df_features,
        min_ev_pct=float(ev),
        kelly_frac=1.0,
        max_stake_pct=0.33,
        initial_bankroll=10_000.0,
        allowed_markets=mkts,
        min_prob=mp,
        skip_late_season=True,
        skip_home_title_race=False,
        odds_source="Max", detect_source="PS",
        enable_simultaneous_correction=True,
        market_gates=market_gates,
        banned_dows=dows,
        max_ev_pct=evcap,
        elo_gap_min=gmin,
        calibrators=calibrators,
    )
    if "error" in sum_:
        print(f"{name:35s}  ERROR: {sum_['error']}")
        continue
    s = pd.concat([pd.Series([10_000.0]), log["Bankroll"].astype(float)],
                   ignore_index=True)
    dd = float(((s.cummax() - s) / s.cummax()).max() * 100)
    per_week = sum_['n_bets'] / 52.0
    print(f"{name:35s}  £{sum_['final']:>9,.0f}  "
          f"{sum_['roi']:>+6.1f}%  {sum_['n_bets']:>4}  "
          f"{sum_['win_rate']:>4.1f}%  {per_week:>5.1f}  {dd:>4.1f}%")
