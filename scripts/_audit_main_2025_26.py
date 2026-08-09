"""Audit-mode backtest — runs the deployed Main config on the 2025-26
season and prints the skip-counter breakdown so we can see exactly where
candidates are being dropped.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                 # noqa: E402
import portfolio as pf                             # noqa: E402

settings = json.load(open(ROOT / "data" / "portfolio.json"))["settings"]
print("CONFIG:")
for k in ("min_prob", "min_ev", "main_banned_dows", "main_elo_gap_min",
         "main_max_ev_pct"):
    print(f"  {k}: {settings.get(k)}")
print()

df = load_data()
df_features = add_rolling_features(df)
print(f"[load_data] {len(df)} rows total. Latest: {df.Date.max().date()}")

# 39-week test window — should land on Aug 2025 cutoff
bt = backtest_models(df, df_features, test_weeks=39)
print(f"[backtest_models] test set: {len(bt)} matches")
print(f"  span: {bt.Date.min().date()} → {bt.Date.max().date()}")
print()

# Run with the deployed config + print skip counters
log, summary = pf.ev_backtest_simulate(
    bt, df, df_features=df_features,
    min_ev_pct=float(settings.get("min_ev", 0.40)) * 100,
    kelly_frac=float(settings.get("kelly_fraction", 1.0)),
    max_stake_pct=float(settings.get("max_stake_pct", 0.33)),
    initial_bankroll=10_000.0,
    allowed_markets=set(settings.get("auto_markets", ["D", "under25"])),
    min_prob=float(settings.get("min_prob", 0.32)),
    skip_late_season=bool(settings.get("skip_late_season", True)),
    skip_home_title_race=bool(settings.get("skip_home_title_race", False)),
    odds_source="Max", detect_source="PS",
    enable_simultaneous_correction=True,
    market_gates=settings.get("market_gates"),
    banned_dows=set(settings.get("main_banned_dows", [])) or None,
    banned_months=set(settings.get("main_banned_months", [])) or None,
    max_ev_pct=settings.get("main_max_ev_pct"),
    min_team_elo=settings.get("main_min_team_elo"),
    max_team_elo=settings.get("main_max_team_elo"),
    elo_gap_min=settings.get("main_elo_gap_min"),
    elo_gap_max=settings.get("main_elo_gap_max"),
)

print("=" * 70)
print("SKIP COUNTERS — where candidates get filtered out")
print("=" * 70)
print(f"  Total matches in test:        {len(bt)}")
print(f"  Candidates per match:         ~3 (D + over25 + under25, since Main")
print(f"                                 auto_markets = ['D','under25'])")
print(f"  Theoretical candidate count:  ~{len(bt)*3}")
print()
print(f"  Skipped (March-April filter): {summary.get('skipped_late_season', 0):>5}")
print(f"  Skipped (Mon/Fri DOW ban):    {summary.get('skipped_calendar', 0):>5}")
print(f"  Skipped (ELO gap < 100):      {summary.get('skipped_elo', 0):>5}")
print(f"  Skipped (prob < 32% gate):    {summary.get('skipped_min_prob', 0):>5}")
print(f"  Skipped (EV > 100% cap):      {summary.get('skipped_max_ev', 0):>5}")
print()
print(f"  BETS PLACED:                  {summary['n_bets']:>5}")
print(f"  Final bankroll:               £{summary['final']:>10,.0f}")
print(f"  Win rate:                     {summary['win_rate']}%")
print()

# Show every bet placed
print("=" * 70)
print("EVERY BET PLACED IN THE 2025-26 SEASON BACKTEST")
print("=" * 70)
if not log.empty:
    for i, r in log.iterrows():
        print(f"  {r['Date'].strftime('%Y-%m-%d %a'):14s}  "
              f"{r['Match']:35s}  {r['Market']:10s}  "
              f"prob {r['Model%']:>6s}  EV {r['EV']:>6s}  "
              f"odds {r['Odds']:.2f}  stake £{r['Stake']:>7,.2f}  "
              f"{r['Result']}")

print()
# Now run the same window WITHOUT the new filters for comparison
print("=" * 70)
print("FOR COMPARISON — BASELINE (no v2 filters) on same 39-week window")
print("=" * 70)
log_b, sum_b = pf.ev_backtest_simulate(
    bt, df,
    min_ev_pct=float(settings.get("min_ev", 0.40)) * 100,
    kelly_frac=float(settings.get("kelly_fraction", 1.0)),
    max_stake_pct=float(settings.get("max_stake_pct", 0.33)),
    initial_bankroll=10_000.0,
    allowed_markets=set(settings.get("auto_markets", ["D", "under25"])),
    min_prob=0.30,  # baseline used 0.30
    skip_late_season=True,
    skip_home_title_race=False,
    odds_source="Max", detect_source="PS",
    enable_simultaneous_correction=True,
    market_gates=settings.get("market_gates"),
)
print(f"  Bets placed:    {sum_b['n_bets']}")
print(f"  Final bankroll: £{sum_b['final']:,.0f}")
print(f"  Win rate:       {sum_b['win_rate']}%")
