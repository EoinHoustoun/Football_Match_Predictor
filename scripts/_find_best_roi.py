"""Focused sweep — find the absolute best Main ROI config with calibration
applied (i.e., matching live behaviour). User just rejected £29k as "bad"
and wants the £93k feel back — but that was inflated. This finds the true
honest best.

Sweeps: min_prob × min_ev × ELO gap min × max-EV cap × DOW bans
All with calibrators applied.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from itertools import product
from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                 # noqa: E402
import portfolio as pf                             # noqa: E402

print("Loading data + fitting backtest_models (52w)…")
df = load_data()
df_features = add_rolling_features(df)
bt = backtest_models(df, df_features, test_weeks=52)
calibrators = pf.fit_calibrators_from_backtest(bt)
print(f"  bt rows: {len(bt)}\n")

variants = []
for (mp, ev, gMin, evCap, dows) in product(
    [0.20, 0.22, 0.25, 0.30],            # min_prob
    [5, 10, 15, 20, 30, 40],             # min_ev_pct
    [None, 50, 80, 100],                  # ELO gap min
    [None, 0.80, 1.00, 1.50],             # max-EV cap
    [None, frozenset({"Mon","Fri"})],     # DOW bans
):
    parts = [f"mp{int(mp*100)}", f"ev{ev}"]
    if gMin is not None: parts.append(f"gMin{gMin}")
    if evCap is not None: parts.append(f"evCap{int(evCap*100)}")
    if dows is not None: parts.append("DOWmf")
    variants.append({
        "name": "_".join(parts),
        "mp": mp, "ev": ev, "gMin": gMin, "evCap": evCap,
        "dows": set(dows) if dows else None,
    })

print(f"Sweeping {len(variants)} variants × calibrated backtest…")
results = []
for i, v in enumerate(variants, 1):
    log, sum_ = pf.ev_backtest_simulate(
        bt, df, df_features=df_features,
        min_ev_pct=float(v["ev"]),
        kelly_frac=1.0,
        max_stake_pct=0.33,
        initial_bankroll=10_000.0,
        allowed_markets={"D", "under25"},
        min_prob=v["mp"],
        skip_late_season=True,
        skip_home_title_race=False,
        odds_source="Max", detect_source="PS",
        enable_simultaneous_correction=True,
        market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
        banned_dows=v["dows"],
        max_ev_pct=v["evCap"],
        elo_gap_min=v["gMin"],
        calibrators=calibrators,
    )
    if "error" in sum_:
        continue
    s = pd.concat([pd.Series([10_000.0]), log["Bankroll"].astype(float)],
                   ignore_index=True)
    dd = float(((s.cummax() - s) / s.cummax()).max() * 100)
    results.append({
        "name": v["name"],
        "final": sum_["final"],
        "roi": sum_["roi"],
        "n": sum_["n_bets"],
        "win_rate": sum_["win_rate"],
        "dd": round(dd, 2),
        "per_week": round(sum_["n_bets"] / 52.0, 2),
        "overrides": {k: list(vv) if isinstance(vv, set) else vv
                      for k, vv in v.items() if k != "name"},
    })
    if i % 30 == 0:
        print(f"  {i}/{len(variants)}")

print()
print("=" * 110)
print("TOP 20 BY FINAL BANKROLL (52w, calibration on)")
print("=" * 110)
print(f"{'rank':>4}  {'config':50s}  {'final':>10}  {'ROI':>7}  "
      f"{'n':>4}  {'win%':>5}  {'/wk':>4}  {'DD%':>5}")
for i, r in enumerate(sorted(results, key=lambda x: x["final"], reverse=True)[:20], 1):
    print(f"{i:>4}  {r['name']:50s}  £{r['final']:>9,.0f}  "
          f"{r['roi']:>+6.1f}%  {r['n']:>4}  {r['win_rate']:>4.1f}%  "
          f"{r['per_week']:>3.1f}  {r['dd']:>4.1f}%")
