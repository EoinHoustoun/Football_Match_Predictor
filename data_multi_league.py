"""Multi-league data loader — extends the EPL-only `data.py` to cover
Championship, La Liga, Bundesliga, and Serie A (plus EPL).

Use `load_multi_league()` to get all 9,871 matches as a single DataFrame,
optionally filtered to a subset of leagues. The schema matches `data.py`
output close enough that downstream models (DC, XGB, Draw Specialist)
can train on it with minimal changes — every match has Date, HomeTeam,
AwayTeam, FTHG, FTAG, FTR, and the standard odds columns.

Important: team-names are unique per league in football-data.co.uk's
naming convention, so pooling all leagues into one DC fit gives each
team its own attack/defence — but the global home_advantage and rho
parameters get averaged across leagues. That's a trade-off — single
global rho usually works fine, but if league-specific tuning matters
you'd want a more elaborate hierarchical fit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "multi_league"

LEAGUE_NAMES = {
    "E0":  "EPL",
    "E1":  "Championship",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
}


def load_multi_league(
    leagues: list[str] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the unified multi-league match dataset.

    Args:
        leagues: optional subset of league codes to include
                 (e.g. ["E0","E1"] for EPL+Championship). None = all.
        refresh: if True, re-run the download script to refetch (slow).

    Returns:
        DataFrame with columns: Date, League, LeagueName, Season,
        HomeTeam, AwayTeam, FTHG, FTAG, FTR + standard odds columns.
    """
    csv_path = DATA_DIR / "matches.csv"
    if refresh or not csv_path.exists():
        import subprocess, sys
        subprocess.run(
            [sys.executable, str(DATA_DIR.parent.parent / "scripts" / "download_multi_league.py")],
            check=True,
        )

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if leagues:
        df = df[df["League"].isin(leagues)].copy()
    df = df.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)
    return df


def disambiguate_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix each team name with its league code so models treat teams
    in different leagues as distinct entities.

    Example: "Arsenal" → "E0::Arsenal", "Arsenal" (Brazilian club, hypothetical)
    in I1 → "I1::Arsenal". For football-data.co.uk's actual naming this is
    rarely necessary (PL teams have distinct names from La Liga teams), but
    Spanish/Italian shared first-words (e.g. "Atletico") can collide.
    """
    df = df.copy()
    df["HomeTeam"] = df["League"].astype(str) + "::" + df["HomeTeam"].astype(str)
    df["AwayTeam"] = df["League"].astype(str) + "::" + df["AwayTeam"].astype(str)
    return df


def summary(df: pd.DataFrame) -> dict:
    """Quick stats about the loaded dataset."""
    return {
        "n_matches":     int(len(df)),
        "n_leagues":     int(df["League"].nunique()),
        "n_seasons":     int(df["Season"].nunique()),
        "date_min":      df["Date"].min().date(),
        "date_max":      df["Date"].max().date(),
        "per_league":    df["League"].value_counts().to_dict(),
        "per_season":    df["Season"].value_counts().sort_index().to_dict(),
    }
