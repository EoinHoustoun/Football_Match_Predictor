"""Estimate the Championship-to-Premier-League downgrade, empirically.

A promoted side has no Premier League history, so the model has nothing to rate
it with. It does have a full Championship season. The question this answers is:
how much of a Championship rating survives the step up?

Method. For every team promoted between 2021-22 and 2025-26 we fit Dixon-Coles
twice: once on its final Championship season, once on its first Premier League
season. Both fits use actual goals (never xG) so the two leagues are measured
on the same instrument, and both run without time decay so the rating reflects
the whole season rather than its last few weeks.

DC attack and defence are identified up to an additive constant and normalised
to sum to zero, so each rating is already relative to its own league's average.
The difference between the two fits is therefore exactly what we want: how far
a team moves relative to the field when the field gets better.

Output is a per-team table, the fitted shift with its spread, and PL-equivalent
priors for the 2026-27 intake. The spread matters more than the mean here — a
sample of fifteen promoted teams cannot support a precise point estimate, and
pretending otherwise is how the £100k artifact happened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import compute_dixon_coles_ratings   # noqa: E402

CHAMP_DIR = ROOT / "data" / "championship"
CHAMP_URL = "https://www.football-data.co.uk/mmz4281/{code}/E1.csv"

# No decay: we want a whole-season rating, not an end-of-season form reading.
NO_DECAY = 10_000.0

# season label -> football-data code
SEASONS = {
    "2020-21": "2021", "2021-22": "2122", "2022-23": "2223",
    "2023-24": "2324", "2024-25": "2425", "2025-26": "2526",
}

# Who went up, and by which route. Route is kept because finishing 2nd and
# scraping through the playoffs from 6th are not the same event.
PROMOTED = {
    "2020-21": [("Norwich", "auto"), ("Watford", "auto"), ("Brentford", "playoff")],
    "2021-22": [("Fulham", "auto"), ("Bournemouth", "auto"), ("Nott'm Forest", "playoff")],
    "2022-23": [("Burnley", "auto"), ("Sheffield United", "auto"), ("Luton", "playoff")],
    "2023-24": [("Leicester", "auto"), ("Ipswich", "auto"), ("Southampton", "playoff")],
    "2024-25": [("Leeds", "auto"), ("Burnley", "auto"), ("Sunderland", "playoff")],
    "2025-26": [("Coventry", "auto"), ("Ipswich", "auto"), ("Hull", "playoff")],
}

# The Premier League season each cohort arrived in.
NEXT_SEASON = {
    "2020-21": "2021-22", "2021-22": "2022-23", "2022-23": "2023-24",
    "2023-24": "2024-25", "2024-25": "2025-26", "2025-26": "2026-27",
}


def load_championship(season: str) -> pd.DataFrame:
    """Championship matches for one season, cached on disk."""
    CHAMP_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAMP_DIR / f"{SEASONS[season]}.csv"
    if not path.exists():
        import requests
        r = requests.get(CHAMP_URL.format(code=SEASONS[season]), timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def goals_only(df: pd.DataFrame) -> pd.DataFrame:
    """Strip xG so both leagues are fitted on the same instrument.

    The Premier League frame carries Understat xG and the Championship frame
    does not. Fitting one on xG and the other on goals would fold the
    difference between the two measurements into the league-shift estimate.
    """
    return df.drop(columns=[c for c in ("xg_h", "xg_a") if c in df.columns])


def season_ratings(df: pd.DataFrame) -> dict:
    return compute_dixon_coles_ratings(goals_only(df), decay_weeks=NO_DECAY)


def table_from(df: pd.DataFrame) -> pd.DataFrame:
    """Points and goal difference, for context alongside the ratings."""
    acc: dict[str, dict] = {}
    for _, r in df.iterrows():
        h, a, hg, ag = r["HomeTeam"], r["AwayTeam"], int(r["FTHG"]), int(r["FTAG"])
        for t in (h, a):
            acc.setdefault(t, {"P": 0, "GF": 0, "GA": 0, "Pts": 0})
        acc[h]["P"] += 1; acc[a]["P"] += 1
        acc[h]["GF"] += hg; acc[h]["GA"] += ag
        acc[a]["GF"] += ag; acc[a]["GA"] += hg
        if hg > ag:   acc[h]["Pts"] += 3
        elif ag > hg: acc[a]["Pts"] += 3
        else:         acc[h]["Pts"] += 1; acc[a]["Pts"] += 1
    t = pd.DataFrame(acc).T
    t["GD"] = t["GF"] - t["GA"]
    return t.sort_values(["Pts", "GD"], ascending=False)


def build_cohort(pl_df: pd.DataFrame) -> pd.DataFrame:
    """One row per promoted team with its Championship and Premier League fits."""
    rows = []
    for champ_season, promoted in PROMOTED.items():
        pl_season = NEXT_SEASON[champ_season]
        champ_df = load_championship(champ_season)
        champ_r  = season_ratings(champ_df)
        champ_t  = table_from(champ_df)

        pl_slice = pl_df[pl_df["Season"] == pl_season]
        pl_r = season_ratings(pl_slice) if len(pl_slice) else None

        for team, route in promoted:
            row = {
                "champ_season": champ_season, "pl_season": pl_season,
                "team": team, "route": route,
                "champ_att": champ_r["attacks"].get(team),
                "champ_def": champ_r["defenses"].get(team),
                "champ_pts": champ_t.loc[team, "Pts"] if team in champ_t.index else np.nan,
                "champ_gd":  champ_t.loc[team, "GD"] if team in champ_t.index else np.nan,
                "pl_att": None, "pl_def": None,
            }
            if pl_r is not None and team in pl_r["attacks"]:
                row["pl_att"] = pl_r["attacks"][team]
                row["pl_def"] = pl_r["defenses"][team]
            rows.append(row)

    out = pd.DataFrame(rows)
    out["d_att"] = out["pl_att"] - out["champ_att"]
    out["d_def"] = out["pl_def"] - out["champ_def"]
    return out


def main() -> None:
    from data import load_data

    pl_df = load_data()

    cohort = build_cohort(pl_df)
    observed = cohort.dropna(subset=["d_att", "d_def"])

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("PROMOTED TEAMS: CHAMPIONSHIP RATING vs FIRST PREMIER LEAGUE SEASON")
    print("=" * 78)
    show = observed[["pl_season", "team", "route", "champ_pts", "champ_gd",
                     "champ_att", "pl_att", "d_att",
                     "champ_def", "pl_def", "d_def"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print("\n" + "-" * 78)
    print(f"SAMPLE: {len(observed)} promoted teams across "
          f"{observed['pl_season'].nunique()} seasons")
    print("-" * 78)
    for label, col in (("attack", "d_att"), ("defence", "d_def")):
        v = observed[col]
        print(f"  {label:8s} shift   mean {v.mean():+.3f}   median {v.median():+.3f}   "
              f"sd {v.std():.3f}   range [{v.min():+.3f}, {v.max():+.3f}]")

    print("\n  By promotion route:")
    for route, g in observed.groupby("route"):
        print(f"    {route:8s} (n={len(g):2d})  attack {g['d_att'].mean():+.3f}   "
              f"defence {g['d_def'].mean():+.3f}")

    # Does the Championship season predict how well the step up goes?
    print("\n  Does Championship goal difference predict the PL rating?")
    for col, label in (("pl_att", "PL attack"), ("pl_def", "PL defence")):
        r = observed["champ_gd"].corr(observed[col])
        print(f"    corr(champ GD, {label:11s}) = {r:+.3f}")

    # ── Apply to the 2026-27 intake ───────────────────────────────────────
    d_att, d_def = observed["d_att"].mean(), observed["d_def"].mean()
    sd_att, sd_def = observed["d_att"].std(), observed["d_def"].std()

    incoming = cohort[cohort["pl_season"] == "2026-27"]
    print("\n" + "=" * 78)
    print("2026-27 INTAKE — PREMIER LEAGUE PRIORS FROM THE 2025-26 CHAMPIONSHIP")
    print("=" * 78)
    for _, r in incoming.iterrows():
        pa, pd_ = r["champ_att"] + d_att, r["champ_def"] + d_def
        print(f"\n  {r['team']}  ({r['route']}, {int(r['champ_pts'])} pts, "
              f"GD {int(r['champ_gd']):+d})")
        print(f"    Championship   attack {r['champ_att']:+.3f}   defence {r['champ_def']:+.3f}")
        print(f"    PL prior       attack {pa:+.3f}   defence {pd_:+.3f}")
        print(f"    68% interval   attack [{pa - sd_att:+.3f}, {pa + sd_att:+.3f}]   "
              f"defence [{pd_ - sd_def:+.3f}, {pd_ + sd_def:+.3f}]")

    # Where those priors would sit in the current Premier League.
    latest = pl_df[pl_df["Season"] == pl_df["Season"].max()]
    latest_r = season_ratings(latest)
    ranked = sorted(latest_r["attacks"].items(), key=lambda kv: kv[1], reverse=True)
    print("\n  For scale, 2025-26 Premier League attack ratings:")
    print(f"    best  {ranked[0][0]:14s} {ranked[0][1]:+.3f}")
    print(f"    median{'':14s} {np.median([v for _, v in ranked]):+.3f}")
    print(f"    worst {ranked[-1][0]:14s} {ranked[-1][1]:+.3f}")

    out_path = ROOT / "data" / "diagnostics" / "promoted_team_prior.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out_path, index=False)
    print(f"\n  Full cohort written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
