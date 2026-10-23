#!/usr/bin/env python3
"""Mock Two paper line · anchored vs carried season simulation for 2026-27.

Runs the full-season Monte Carlo twice on identical draws — once on the
carried-over Dixon-Coles ratings, once with the preseason market anchor
(`research_market_anchor`) blended in — and freezes both to
`data/research/mock2_market_anchor_<season>.json`.

That file IS the experiment: in May the pair is scored against the real
table (mean points error, Brier on title/top-4/relegation) and the anchor
graduates, retunes, or dies on that evidence. Nothing here touches the live
app, portfolio, or bet placement.

    python3 scripts/mock2_market_anchor_sim.py [--beta 0.5] [--sims 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import load_data                                    # noqa: E402
from models import (MARKET_ODDS_CAPTURED, MARKET_RELEGATION_ODDS,  # noqa: E402
                    compute_dixon_coles_ratings, market_odds_are_stale,
                    seed_promoted_teams, simulate_season)
from research_market_anchor import DEFAULT_BETA, anchored_dc_ratings  # noqa: E402

OUT_DIR = ROOT / "data" / "research"


def _probs(sim_positions: np.ndarray, teams: list[str]) -> dict:
    out = {}
    for i, t in enumerate(teams):
        pos = sim_positions[:, i]
        out[t] = {
            "mean_pts": None,           # filled by caller from summary
            "p_title": float((pos == 1).mean()),
            "p_top4": float((pos <= 4).mean()),
            "p_releg": float((pos >= len(teams) - 2).mean()),
        }
    return out


def run(beta: float, n_sims: int) -> Path:
    df = load_data()
    season_label = "2026-27"

    clubs = sorted(MARKET_RELEGATION_ODDS)      # the market's twenty IS the league
    dc = compute_dixon_coles_ratings(df)
    dc = seed_promoted_teams(dc, clubs)          # promoted priors, market-ordered

    fixtures = [{"home": h, "away": a} for h, a in permutations(clubs, 2)]
    table = pd.DataFrame({"Team": clubs, "Pts": 0, "GF": 0, "GA": 0,
                          "GD": 0, "Played": 0})

    result = {"season": season_label, "run_date": date.today().isoformat(),
              "beta": beta, "n_sims": n_sims,
              "odds_captured": MARKET_ODDS_CAPTURED,
              "odds_stale_at_run": bool(market_odds_are_stale()),
              "relegation_odds": dict(MARKET_RELEGATION_ODDS),
              "lines": {}}

    for name, ratings in (("carried", dc),
                          ("anchored", anchored_dc_ratings(dc, beta))):
        summary, _pts, _fx, positions = simulate_season(
            fixtures, table, ratings, n_sims=n_sims, rng_seed=42,
            return_samples=True)
        line = _probs(positions, clubs)
        mean_pts = dict(zip(summary["Team"], summary["mean_pts"]))
        for t in line:
            line[t]["mean_pts"] = round(float(mean_pts.get(t, float("nan"))), 2)
        result["lines"][name] = line

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mock2_market_anchor_{season_label.replace('-', '_')}.json"
    out.write_text(json.dumps(result, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--sims", type=int, default=10_000)
    args = ap.parse_args()

    if market_odds_are_stale():
        print("NOTE: the relegation odds table is stale "
              f"(captured {MARKET_ODDS_CAPTURED}) · refresh "
              "models.MARKET_RELEGATION_ODDS before trusting this run.")

    out = run(args.beta, args.sims)
    res = json.loads(out.read_text())
    car, anc = res["lines"]["carried"], res["lines"]["anchored"]
    print(f"\nMock Two market anchor · beta {args.beta} · {args.sims:,} sims "
          f"· odds captured {res['odds_captured']}")
    print(f"{'club':<15} {'pts':>6} {'→':^1} {'pts':>6} | "
          f"{'title':>6} {'→':^1} {'title':>6} | {'releg':>6} {'→':^1} {'releg':>6}")
    for t in sorted(car, key=lambda t: -anc[t]["mean_pts"]):
        c, a = car[t], anc[t]
        print(f"{t:<15} {c['mean_pts']:>6.1f} → {a['mean_pts']:>6.1f} | "
              f"{c['p_title']:>6.1%} → {a['p_title']:>6.1%} | "
              f"{c['p_releg']:>6.1%} → {a['p_releg']:>6.1%}")
    print(f"\nfrozen to {out}")


if __name__ == "__main__":
    main()
