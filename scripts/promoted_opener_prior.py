"""Back a promoted side's Dixon-Coles rating out of its opening-day odds.

The market-edge test (`scripts/promoted_market_prior.py`) beat a flat prior over
three matches but LOST over one, and only one match is strictly pre-season: the
closing odds for match 2 already know how match 1 went. So the version that won
is not usable before kickoff, which is exactly when it is needed.

That test used a crude estimator: implied points per game, compared against the
points a "neutral" side would be given in the same fixture. Approximating the
opponent away like that throws information out. One fixture against Manchester
City and one against Luton are worlds apart, and the neutral baseline only
half-corrects for it.

This conditions on the opponent properly. For a promoted side T opening away to
a known opponent O:

    lambda_home = exp(att_O - def_T + home_adv)
    lambda_away = exp(att_T - def_O)

O's attack and defence, and the home advantage, come from a Dixon-Coles fit on
the PREVIOUS season only, so nothing from the season being predicted is used.
That leaves two unknowns, att_T and def_T, against two independent market
probabilities, so the rating is identified. Solve for the pair that reproduces
the market's prices.

The result is a prior in the model's own units, derived from the sharpest
pre-season information that exists, using only data already in the repo.

Whether it is better than assuming every promoted side is identical is the
question — answered the same way as before, leave-one-out against the pooled
mean, because that is the incumbent.

Output: stdout + data/diagnostics/promoted_opener_prior_<ts>.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_data                                  # noqa: E402
from models import compute_dixon_coles_ratings              # noqa: E402
from scripts.promoted_team_prior import build_cohort        # noqa: E402
from scripts.promoted_market_prior import _implied, loo_rmse  # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NO_DECAY = 10_000.0   # whole-season rating, matching the prior study
MAX_GOALS = 8


def _dc_probs(att_h: float, def_h: float, att_a: float, def_a: float,
              home_adv: float, rho: float) -> tuple[float, float, float]:
    """Home, draw, away probabilities for a Dixon-Coles pair of ratings."""
    lam_h = np.exp(att_h - def_a + home_adv)
    lam_a = np.exp(att_a - def_h)
    goals = np.arange(MAX_GOALS + 1)
    p_h_goals = np.exp(-lam_h) * lam_h ** goals / np.array(
        [np.math.factorial(g) for g in goals])
    p_a_goals = np.exp(-lam_a) * lam_a ** goals / np.array(
        [np.math.factorial(g) for g in goals])
    matrix = np.outer(p_h_goals, p_a_goals)

    # Dixon-Coles low-score correction, same shape the model itself uses.
    tau = np.ones_like(matrix)
    tau[0, 0] = 1 - lam_h * lam_a * rho
    tau[0, 1] = 1 + lam_h * rho
    tau[1, 0] = 1 + lam_a * rho
    tau[1, 1] = 1 - rho
    matrix = matrix * tau
    matrix = matrix / matrix.sum()

    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return home, draw, away


def solve_rating(market: tuple[float, float, float], opponent: dict,
                 team_at_home: bool, home_adv: float, rho: float
                 ) -> tuple[float, float] | None:
    """Find the (attack, defence) that reproduces the market's prices."""
    target = np.array(market)

    def loss(params: np.ndarray) -> float:
        att_t, def_t = params
        if team_at_home:
            probs = _dc_probs(att_t, def_t, opponent["att"], opponent["def"],
                              home_adv, rho)
        else:
            probs = _dc_probs(opponent["att"], opponent["def"], att_t, def_t,
                              home_adv, rho)
        return float(np.sum((np.array(probs) - target) ** 2))

    best = minimize(loss, x0=np.array([-0.6, 0.8]), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
    if not best.success and best.fun > 1e-4:
        return None
    return float(best.x[0]), float(best.x[1])


def previous_season_fit(df: pd.DataFrame, pl_season: str) -> dict | None:
    """Dixon-Coles on the season before `pl_season`. No look-ahead."""
    seasons = sorted(df["Season"].unique())
    if pl_season not in seasons:
        # The season being predicted may not be in the data yet (2026-27).
        prior_seasons = [s for s in seasons if s < pl_season]
    else:
        prior_seasons = seasons[:seasons.index(pl_season)]
    if not prior_seasons:
        return None
    prev = df[df["Season"] == prior_seasons[-1]]
    if prev.empty:
        return None
    return compute_dixon_coles_ratings(prev, decay_weeks=NO_DECAY)


def opener_rating(df: pd.DataFrame, team: str, pl_season: str) -> dict | None:
    """Market-implied rating for `team` from its first fixture of `pl_season`."""
    fit = previous_season_fit(df, pl_season)
    if fit is None:
        return None

    season_df = df[df["Season"] == pl_season].sort_values("Date")
    played = season_df[(season_df["HomeTeam"] == team) |
                       (season_df["AwayTeam"] == team)].head(1)
    if played.empty:
        return None
    row = played.iloc[0]

    probs = _implied(row)
    if probs is None:
        return None

    at_home = row["HomeTeam"] == team
    opponent_name = row["AwayTeam"] if at_home else row["HomeTeam"]
    if opponent_name not in fit["attacks"]:
        # Two promoted sides cannot rate each other; neither is known yet.
        return None
    opponent = {"att": fit["attacks"][opponent_name],
                "def": fit["defenses"][opponent_name]}

    solved = solve_rating(probs, opponent, at_home,
                          fit["home_adv"], fit["rho"])
    if solved is None:
        return None
    return {"opener_att": solved[0], "opener_def": solved[1],
            "opponent": opponent_name, "at_home": bool(at_home)}


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading data…")
    df = load_data()
    cohort = build_cohort(df)

    rows = []
    for _, r in cohort.iterrows():
        solved = opener_rating(df, r["team"], r["pl_season"])
        if solved:
            rows.append({**r.to_dict(), **solved})
    table = pd.DataFrame(rows)
    observed = table.dropna(subset=["pl_att", "pl_def"]).copy()

    print("\nRating implied by the opening fixture's price, against the rating "
          "the season actually produced\n")
    print(f"{'season':9s} {'team':16s} {'opponent':14s} {'H/A':4s} "
          f"{'mkt att':>8s} {'PL att':>8s} {'mkt def':>8s} {'PL def':>8s}")
    print("-" * 82)
    for _, r in table.sort_values(["pl_season", "team"]).iterrows():
        a = f"{r['pl_att']:+.3f}" if pd.notna(r["pl_att"]) else "     —"
        d = f"{r['pl_def']:+.3f}" if pd.notna(r["pl_def"]) else "     —"
        print(f"{r['pl_season']:9s} {r['team']:16s} {r['opponent']:14s} "
              f"{'H' if r['at_home'] else 'A':4s} "
              f"{r['opener_att']:+8.3f} {a:>8s} {r['opener_def']:+8.3f} {d:>8s}")

    print(f"\nLeave-one-out, n={len(observed)} promoted teams, "
          f"strictly pre-season information")
    print("-" * 82)
    results = {}
    for target, feature, label in (("pl_att", "opener_att", "PL attack"),
                                   ("pl_def", "opener_def", "PL defence")):
        y = observed[target].to_numpy(dtype=float)
        x = observed[feature].to_numpy(dtype=float)
        pooled, model = loo_rmse(x, y)
        verdict = "BEATS pooled" if model < pooled else "loses to pooled"
        corr = float(np.corrcoef(x, y)[0, 1])
        print(f"  {label:11s} from the opening price   RMSE {model:.3f} "
              f"vs pooled {pooled:.3f}   corr {corr:+.3f}   {verdict}")
        results[target] = {"model_rmse": round(model, 4),
                           "pooled_rmse": round(pooled, 4),
                           "corr": round(corr, 4),
                           "beats_pooled": bool(model < pooled)}

    both = all(v["beats_pooled"] for v in results.values())
    print(f"\nVERDICT: {'usable pre-kickoff' if both else 'does not beat a flat prior'}")

    path = OUT_DIR / f"promoted_opener_prior_{ts}.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "loo": results,
        "teams": table.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"Written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
