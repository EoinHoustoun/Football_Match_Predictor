"""Download football-data.co.uk CSVs for multiple leagues across 5 seasons.

Currently the F_PRED model is trained only on EPL (E0) — ~1869 matches over
5 seasons. Adding Championship (E1), La Liga (SP1), Bundesliga (D1), and
Serie A (I1) brings the corpus to ~10k+ matches: 5× more data for
DC/XGB/draw-specialist training, with the same column schema.

This script saves a unified parquet/csv `data/multi_league_matches.csv`
keyed by League × Season × Date × HomeTeam × AwayTeam, ready for
downstream model training that wants more samples.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd       # noqa: E402
import requests           # noqa: E402

OUT_DIR = ROOT / "data" / "multi_league"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = {
    "E0":  "EPL",
    "E1":  "Championship",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
}

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
SEASON_LABEL = {
    "2122": "2021-22", "2223": "2022-23", "2324": "2023-24",
    "2425": "2024-25", "2526": "2025-26",
}

URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Common columns we expect across leagues. Keep this small but useful.
KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HC", "AC",
    # Closing/place odds
    "B365H", "B365D", "B365A",
    "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA",
    "PSH",  "PSD",  "PSA",
    # Over/under
    "B365>2.5", "B365<2.5",
    "Max>2.5",  "Max<2.5",
    "Avg>2.5",  "Avg<2.5",
    "P>2.5",    "P<2.5",
    # BTTS (when available)
    "B365BTSY", "B365BTSN",
    "MaxBTSY",  "MaxBTSN",
    "AvgBTSY",  "AvgBTSN",
]


def _fetch_one(league: str, season: str) -> pd.DataFrame:
    url = URL_TEMPLATE.format(season=season, league=league)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  [SKIP] {league}/{season}: {e}")
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text))
    # First column sometimes has BOM
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def main() -> int:
    all_frames: list[pd.DataFrame] = []
    print(f"Fetching {len(LEAGUES)} leagues × {len(SEASONS)} seasons "
          f"= {len(LEAGUES) * len(SEASONS)} CSVs...")
    print()
    for season in SEASONS:
        print(f"== {SEASON_LABEL[season]} ==")
        for league_code, league_name in LEAGUES.items():
            df = _fetch_one(league_code, season)
            if df.empty:
                continue
            # Restrict to known columns that exist in this slice
            kept = [c for c in KEEP_COLS if c in df.columns]
            slim = df[kept].copy()
            slim["League"] = league_code
            slim["LeagueName"] = league_name
            slim["Season"] = SEASON_LABEL[season]
            # Parse Date — football-data uses dd/mm/yyyy
            slim["Date"] = pd.to_datetime(slim["Date"], dayfirst=True,
                                           errors="coerce")
            # Drop rows where Date couldn't parse OR HomeTeam is missing
            slim = slim.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
            print(f"  {league_code} ({league_name:13s}): "
                  f"{len(slim):>4} matches, "
                  f"span {slim['Date'].min().date()} → {slim['Date'].max().date()}")
            all_frames.append(slim)

    if not all_frames:
        print("\nNo data fetched. Aborting.")
        return 1

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["Date", "League", "HomeTeam"])
    combined = combined.reset_index(drop=True)

    out_csv = OUT_DIR / "matches.csv"
    combined.to_csv(out_csv, index=False)
    print()
    print(f"WROTE: {out_csv.relative_to(ROOT)}")
    print(f"  Total matches: {len(combined):>5,}")
    print(f"  Date span:     {combined['Date'].min().date()} → {combined['Date'].max().date()}")
    print(f"  Per-league counts:")
    for lg, n in combined["League"].value_counts().items():
        print(f"    {lg} ({LEAGUES[lg]:13s}): {n:>5,}")
    print(f"  Per-season counts:")
    for s, n in combined["Season"].value_counts().sort_index().items():
        print(f"    {s}: {n:>5,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
