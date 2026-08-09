"""Does the betting market predict a promoted side better than a flat prior?

`scripts/promoted_team_prior.py` established that Championship form carries no
usable out-of-sample signal about the step up: DC attack, DC defence, goal
difference, points and promotion route all lost to the pooled mean in
leave-one-out. So every promoted side currently gets an identical rating.

That is uncomfortable, because the market plainly does not think they are
identical. For 2026-27 it prices Hull at 1/4 to be relegated and Coventry at
4/6 — an implied 80% against 60%.

The market knows things the Championship table cannot: who was bought and sold
over the summer, who changed manager, who is injured, how the squad is judged
against top-flight opposition. This asks whether that knowledge actually
predicts, using the one form of it available for every promoted team back to
2021-22: **the odds on their opening Premier League fixtures**.

Those odds are set before a ball is kicked, so using them is not hindsight. And
they are available for Hull and Coventry right now, which is the point.

Method:
  - For each promoted team, take the first N Premier League matches of its first
    season and read the closing 1X2 odds (Pinnacle where present, else the panel
    average, else Bet365).
  - Strip the overround, convert to implied points per game, and subtract the
    implied points per game an average side would be given in those same
    fixtures, so a brutal opening run is not mistaken for a bad team.
  - Leave-one-out: fit on the other fourteen teams, predict the held-out one,
    and compare RMSE against simply predicting the pooled mean.

Predicting the pooled mean is the incumbent. The market prior only earns its
place by beating it out of sample.

**Result, and the catch.** Over three matches the market beats the pooled mean
on both ratings (attack 0.208 vs 0.226, defence 0.189 vs 0.236) while
Championship goal difference still loses. But the window matters, and not
smoothly:

    N=1  attack 0.235 vs 0.226  lose      defence 0.241 vs 0.236  lose
    N=2  attack 0.233 vs 0.226  lose      defence 0.236 vs 0.236  tie
    N=3  attack 0.208 vs 0.226  WIN       defence 0.189 vs 0.236  WIN
    N=5  attack 0.207 vs 0.226  WIN       defence 0.194 vs 0.236  WIN

The closing odds for match 2 are set after match 1 has been played, so anything
past N=1 carries early-season form as well as pre-season judgement. The step
from lose at N=2 to win at N=3, rather than a gradual improvement, is what you
would expect if the gain comes from results rather than from foresight — and on
fifteen teams that step is not far outside noise.

So this does NOT license changing the prior before kickoff. The only strictly
pre-season version, N=1, loses. What it does support is reassessing a promoted
side around matchday 3, which sits naturally before the no-history gate lifts at
six matches.

The untested idea worth pursuing is pre-season relegation odds, which are real
pre-kickoff market judgement and currently separate Hull (1/4) from Coventry
(4/6). Testing it needs historical pre-season relegation prices for the fifteen
teams, which this repo does not have.

Output: stdout + data/diagnostics/promoted_market_prior_<ts>.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_data                                    # noqa: E402
from scripts.promoted_team_prior import build_cohort, PROMOTED, NEXT_SEASON  # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sharpest first. Pinnacle is the benchmark the rest of the project detects at.
ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"),
             ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]

N_MATCHES = 3   # enough to blunt one freak fixture, still all pre-form pricing


def _implied(row) -> tuple[float, float, float] | None:
    """Overround-stripped (home, draw, away) probabilities for one match."""
    for h_col, d_col, a_col in ODDS_SETS:
        if not all(c in row.index for c in (h_col, d_col, a_col)):
            continue
        try:
            o = [float(row[h_col]), float(row[d_col]), float(row[a_col])]
        except (ValueError, TypeError):
            continue
        if any(np.isnan(v) or v <= 1.0 for v in o):
            continue
        raw = [1.0 / v for v in o]
        total = sum(raw)
        return tuple(v / total for v in raw)
    return None


def _team_ppg(probs: tuple[float, float, float], at_home: bool) -> float:
    p_h, p_d, p_a = probs
    win = p_h if at_home else p_a
    return 3.0 * win + 1.0 * p_d


def market_view(df: pd.DataFrame, team: str, season: str,
                n: int = N_MATCHES) -> dict | None:
    """Implied points per game for `team` over its first `n` matches, and the
    points per game a neutral side would be given in the same fixtures.

    The difference is the part that is about the team rather than the draw.
    """
    season_df = df[df["Season"] == season].sort_values("Date")
    played = season_df[(season_df["HomeTeam"] == team) |
                       (season_df["AwayTeam"] == team)].head(n)
    if played.empty:
        return None

    team_ppg, neutral_ppg = [], []
    for _, row in played.iterrows():
        probs = _implied(row)
        if probs is None:
            continue
        at_home = row["HomeTeam"] == team
        team_ppg.append(_team_ppg(probs, at_home))
        # A neutral side in the same fixture: swap the favourite role out by
        # taking the average of the two sides' implied points.
        neutral_ppg.append((_team_ppg(probs, True) + _team_ppg(probs, False)) / 2)

    if not team_ppg:
        return None
    return {
        "n_matches":   len(team_ppg),
        "market_ppg":  float(np.mean(team_ppg)),
        "neutral_ppg": float(np.mean(neutral_ppg)),
        "edge":        float(np.mean(team_ppg) - np.mean(neutral_ppg)),
    }


def loo_rmse(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Leave-one-out RMSE for (a) predicting the pooled mean and (b) a linear
    fit on x. Both are scored on the held-out team only."""
    pooled_err, model_err = [], []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        pooled_err.append(y[i] - y[keep].mean())
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        model_err.append(y[i] - (slope * x[i] + intercept))
    return (float(np.sqrt(np.mean(np.square(pooled_err)))),
            float(np.sqrt(np.mean(np.square(model_err)))))


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading data and rebuilding the cohort…")
    df = load_data()
    cohort = build_cohort(df)

    rows = []
    for _, r in cohort.iterrows():
        view = market_view(df, r["team"], r["pl_season"])
        if view is None:
            continue
        rows.append({**r.to_dict(), **view})
    table = pd.DataFrame(rows)
    observed = table.dropna(subset=["pl_att", "pl_def"]).copy()

    print(f"\nMarket view over each promoted side's first {N_MATCHES} "
          f"Premier League matches\n")
    print(f"{'season':9s} {'team':16s} {'mkt ppg':>8s} {'neutral':>8s} "
          f"{'edge':>7s} {'PL att':>8s} {'PL def':>8s}")
    print("-" * 70)
    for _, r in table.sort_values(["pl_season", "team"]).iterrows():
        att = f"{r['pl_att']:+.3f}" if pd.notna(r["pl_att"]) else "     —"
        dfc = f"{r['pl_def']:+.3f}" if pd.notna(r["pl_def"]) else "     —"
        print(f"{r['pl_season']:9s} {r['team']:16s} {r['market_ppg']:8.2f} "
              f"{r['neutral_ppg']:8.2f} {r['edge']:+7.2f} {att:>8s} {dfc:>8s}")

    print(f"\nLeave-one-out, n={len(observed)} promoted teams")
    print("-" * 70)
    results = {}
    for target, label in (("pl_att", "PL attack"), ("pl_def", "PL defence")):
        y = observed[target].to_numpy(dtype=float)
        best = None
        for feature, fname in (("edge", "market edge"),
                               ("market_ppg", "market ppg"),
                               ("champ_gd", "Championship GD")):
            x = observed[feature].to_numpy(dtype=float)
            if np.isnan(x).any():
                continue
            pooled, model = loo_rmse(x, y)
            verdict = "BEATS pooled" if model < pooled else "loses to pooled"
            print(f"  {label:11s} from {fname:17s} "
                  f"RMSE {model:.3f} vs pooled {pooled:.3f}   {verdict}")
            results[f"{target}__{feature}"] = {"model_rmse": round(model, 4),
                                               "pooled_rmse": round(pooled, 4),
                                               "beats_pooled": bool(model < pooled)}
            if best is None or model < best[1]:
                best = (fname, model, pooled)
        corr = observed[["edge", target]].corr().iloc[0, 1]
        print(f"  {label:11s} corr(market edge, target) = {corr:+.3f}\n")
        results[f"{target}__corr_edge"] = round(float(corr), 4)

    # ── The current intake ────────────────────────────────────────────────
    incoming = table[table["pl_season"] == "2026-27"]
    if not incoming.empty:
        print("2026-27 intake, priced by the market on their opening fixtures")
        print("-" * 70)
        for _, r in incoming.iterrows():
            print(f"  {r['team']:16s} market ppg {r['market_ppg']:.2f}  "
                  f"neutral {r['neutral_ppg']:.2f}  edge {r['edge']:+.2f}  "
                  f"({int(r['n_matches'])} fixtures priced)")

    # Window sensitivity is the finding, so record it rather than a single N.
    print("Window sensitivity — the odds for match 2 onward already know how")
    print("match 1 went, so only N=1 is strictly pre-season information.")
    print("-" * 70)
    window = {}
    for n in (1, 2, 3, 5):
        cells = []
        for target, label in (("pl_att", "attack"), ("pl_def", "defence")):
            rows_n = []
            for _, r in cohort.iterrows():
                v = market_view(df, r["team"], r["pl_season"], n=n)
                if v:
                    rows_n.append({**r.to_dict(), **v})
            t_n = pd.DataFrame(rows_n).dropna(subset=["pl_att", "pl_def"])
            y = t_n[target].to_numpy(dtype=float)
            x = t_n["edge"].to_numpy(dtype=float)
            pooled, model = loo_rmse(x, y)
            cells.append(f"{label} {model:.3f} vs {pooled:.3f} "
                         f"{'WIN ' if model < pooled else 'lose'}")
            window[f"N{n}__{target}"] = {"model_rmse": round(model, 4),
                                         "pooled_rmse": round(pooled, 4),
                                         "beats_pooled": bool(model < pooled)}
        print(f"  N={n}   " + "   ".join(cells))
    print("\nVERDICT: the pre-season-only window (N=1) does not beat a flat")
    print("prior. Do not change the prior before kickoff on this evidence.")

    path = OUT_DIR / f"promoted_market_prior_{ts}.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_matches_per_team": N_MATCHES,
        "loo": results,
        "window_sensitivity": window,
        "teams": table.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
