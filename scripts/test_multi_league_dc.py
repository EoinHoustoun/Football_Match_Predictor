"""Smoke test: fit Dixon-Coles on multi-league corpus + compare to EPL-only.

Confirms:
  - DC fit converges on 9.8k matches without issue
  - EPL team parameters look sensible (Arsenal/Liverpool should be top attacks)
  - Global home_advantage + rho are in expected range
  - Predictions for a known EPL fixture are plausible
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data_multi_league import load_multi_league, disambiguate_team_names, summary  # noqa: E402
from models import compute_dixon_coles_ratings, predict_dixon_coles  # noqa: E402

print("Loading multi-league dataset…")
df_all = load_multi_league()
stats = summary(df_all)
print(f"  Total: {stats['n_matches']:,} matches")
print(f"  Leagues: {stats['per_league']}")
print(f"  Date span: {stats['date_min']} → {stats['date_max']}")
print()

# Disambiguate team names (e.g. "E0::Arsenal")
df_all = disambiguate_team_names(df_all)

print("Fitting Dixon-Coles on multi-league corpus…")
dc_all = compute_dixon_coles_ratings(df_all)
print(f"  Converged: {dc_all['converged']}")
print(f"  Home advantage: {dc_all['home_adv']:.4f}  "
      f"(typical PL ~ 0.27 → exp = 1.31× attack boost)")
print(f"  Rho (low-score corr): {dc_all['rho']:.4f}")
print(f"  Teams in model: {len(dc_all['teams'])}")
print()

print("Sanity check — top 5 attacks across all leagues:")
attacks = sorted(dc_all["attacks"].items(), key=lambda x: x[1], reverse=True)[:5]
for team, score in attacks:
    print(f"  {team:35s}  {score:+.3f}")

print("\nTop 5 defences (LOWER value = more solid):")
defs = sorted(dc_all["defenses"].items(), key=lambda x: x[1])[:5]
for team, score in defs:
    print(f"  {team:35s}  {score:+.3f}")

print("\nEPL prediction sanity — Liverpool (home) vs Wolves (away):")
try:
    pred = predict_dixon_coles("E0::Liverpool", "E0::Wolves", dc_all)
    print(f"  P(Home Win): {pred['home_win']*100:.1f}%")
    print(f"  P(Draw):     {pred['draw']*100:.1f}%")
    print(f"  P(Away Win): {pred['away_win']*100:.1f}%")
    print(f"  P(Over 2.5): {pred.get('over_25', 0)*100:.1f}%")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nFor comparison — fit DC on EPL-only:")
df_epl = df_all[df_all["HomeTeam"].str.startswith("E0::")].copy()
print(f"  EPL-only rows: {len(df_epl):,}")
dc_epl = compute_dixon_coles_ratings(df_epl)
print(f"  Home advantage: {dc_epl['home_adv']:.4f}")
print(f"  Rho:            {dc_epl['rho']:.4f}")

try:
    pred_epl = predict_dixon_coles("E0::Liverpool", "E0::Wolves", dc_epl)
    print(f"\n  Liverpool vs Wolves (EPL-only model):")
    print(f"    P(Home Win): {pred_epl['home_win']*100:.1f}%")
    print(f"    P(Draw):     {pred_epl['draw']*100:.1f}%")
    print(f"    P(Away Win): {pred_epl['away_win']*100:.1f}%")
except Exception as e:
    print(f"  ERROR: {e}")
