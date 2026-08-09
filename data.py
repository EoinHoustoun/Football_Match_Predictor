"""
Data loading, feature engineering, and helper functions for PL Predictor.
"""
from __future__ import annotations

import difflib
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")

_FIRST_SEASON_START = 2021   # earliest season in the dataset (2021-22)


def _season_start_year(today: date | None = None) -> int:
    """Start year of the current football season. PL seasons run Aug-May;
    July onwards counts as the new season so the pipeline rolls over
    automatically before the first fixtures."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _build_seasons(today: date | None = None) -> dict[str, str]:
    """SEASONS dict from 2021-22 through the current season — auto-extends
    every July so no manual edit is needed at season rollover."""
    out: dict[str, str] = {}
    for y in range(_FIRST_SEASON_START, _season_start_year(today) + 1):
        code = f"{str(y)[-2:]}{str(y + 1)[-2:]}"
        out[_season_label(y)] = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
    return out


SEASONS = _build_seasons()

# Current + previous season — always re-download for freshness
_CUR_SEASON_START = _season_start_year()
_LIVE_SEASONS = {_season_label(_CUR_SEASON_START), _season_label(_CUR_SEASON_START - 1)}

# Understat year key → football season  (2021 → 2021-22, …)
_UNDERSTAT_YEARS = list(range(_FIRST_SEASON_START, _CUR_SEASON_START + 1))

_US_TO_FD = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man United",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest":       "Nott'm Forest",
}


def _compute_elo_series(
    df: pd.DataFrame, k: float = 32.0, home_adv: float = 100.0
):
    """
    Compute running Elo ratings chronologically.
    Returns (records_df, current_elo_dict).
    records_df columns: Date, HomeTeam, AwayTeam, home_elo, away_elo
    current_elo_dict: {team: final_rating after all matches}
    """
    elo: dict[str, float] = {}
    records = []

    for _, row in df.sort_values("Date").iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        r_h = elo.get(h, 1500.0)
        r_a = elo.get(a, 1500.0)

        # Expected score for home team (with home advantage bump)
        e_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h - home_adv) / 400.0))
        e_a = 1.0 - e_h

        records.append({
            "Date": row["Date"], "HomeTeam": h, "AwayTeam": a,
            "home_elo": r_h, "away_elo": r_a,
        })

        ftr = row["FTR"]
        s_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        s_a = 1.0 - s_h

        elo[h] = r_h + k * (s_h - e_h)
        elo[a] = r_a + k * (s_a - e_a)

    return pd.DataFrame(records), elo


def get_current_elo(df: pd.DataFrame) -> dict[str, float]:
    """Return the current (post-last-match) Elo rating for each team."""
    _, elo_dict = _compute_elo_series(df)
    return elo_dict

# ESPN full names → football-data.co.uk short names
_ESPN_TO_FD = {
    "AFC Bournemouth":           "Bournemouth",
    "Brighton & Hove Albion":    "Brighton",
    "Leeds United":              "Leeds",
    "Manchester City":           "Man City",
    "Manchester United":         "Man United",
    "Newcastle United":          "Newcastle",
    "Nottingham Forest":         "Nott'm Forest",
    "Tottenham Hotspur":         "Tottenham",
    "West Ham United":           "West Ham",
    "Wolverhampton Wanderers":   "Wolves",
    "Sunderland AFC":            "Sunderland",
    "Brighton and Hove Albion":  "Brighton",
    "Ipswich Town":              "Ipswich",
    "Leicester City":            "Leicester",
}

# Team names present in the loaded football-data CSVs — populated by
# load_data() and used to auto-resolve names the hardcoded maps don't cover
# (newly promoted sides each season).
KNOWN_FD_TEAMS: set[str] = set()

# Second-tier names, used only as a fallback when a fixture team matches
# nothing in the Premier League data. A promoted side appears in the new
# season's fixtures under an API display name ("Coventry City") months before
# it appears in any Premier League CSV, and the top-flight names alone cannot
# resolve it. Populated lazily from the cached Championship season by
# `load_promoted_team_names()`.
KNOWN_SECOND_TIER_TEAMS: set[str] = set()

# Per-season real-xG coverage, populated by load_data():
# {"2025-26": {"matched": 380, "total": 380}, ...}. Lets the app warn when
# Understat has silently failed and the model is running on the
# shots-conversion proxy instead of real xG.
XG_COVERAGE: dict[str, dict[str, int]] = {}


def _resolve_team_name(raw: str) -> str:
    """API display name → football-data short name.

    Hardcoded map first; for names it doesn't cover, falls back to matching
    against the team names actually present in the loaded CSVs — substring
    containment, then difflib — so promoted teams resolve without a manual
    mapping update every August.
    """
    if raw in _ESPN_TO_FD:
        return _ESPN_TO_FD[raw]
    if not KNOWN_FD_TEAMS or raw in KNOWN_FD_TEAMS:
        return raw
    # "Leeds United" → "Leeds", "Burnley FC" → "Burnley"
    contained = [t for t in KNOWN_FD_TEAMS if t.lower() in raw.lower()]
    if contained:
        return max(contained, key=len)
    close = difflib.get_close_matches(raw, list(KNOWN_FD_TEAMS), n=1, cutoff=0.75)
    if close:
        return close[0]
    # Last resort: a newly promoted side, which has no top-flight history at
    # all. Its football-data name lives in the Championship data.
    second = [t for t in KNOWN_SECOND_TIER_TEAMS if t.lower() in raw.lower()]
    if second:
        return max(second, key=len)
    close = difflib.get_close_matches(raw, list(KNOWN_SECOND_TIER_TEAMS),
                                      n=1, cutoff=0.85)
    return close[0] if close else raw


def load_promoted_team_names(season: str | None = None) -> set[str]:
    """Register last season's Championship names so promoted sides resolve.

    Reads the CSV cached by `scripts/promoted_team_prior.py`. Silent no-op when
    it isn't there — resolution simply falls back to the raw API name, which is
    the behaviour this replaces.
    """
    season = season or _season_label(_CUR_SEASON_START - 1)
    code = f"{season[2:4]}{season[-2:]}"
    path = DATA_DIR / "championship" / f"{code}.csv"
    if not path.exists():
        return set()
    try:
        champ = pd.read_csv(path, encoding="latin-1")
        names = set(champ["HomeTeam"].dropna().unique())
    except Exception:
        return set()
    KNOWN_SECOND_TIER_TEAMS.update(names)
    return names


def _fetch_json_with_backoff(url: str, timeout: int = 10, retries: int = 3):
    """GET → JSON with exponential backoff so a single timeout or 429 doesn't
    silently drop a day of fixtures. Returns None when all attempts fail."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _fetch_understat_xg() -> pd.DataFrame:
    """Fetch (or load cached) match-level xG from Understat for PL seasons."""
    DATA_DIR.mkdir(exist_ok=True)
    records = []

    for year in _UNDERSTAT_YEARS:
        cache = DATA_DIR / f"understat_{year}.json"

        # Re-fetch the current season's xG cache daily
        is_current = year >= datetime.now().year - 1
        if cache.exists() and is_current:
            age_hours = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
            if age_hours > 24:
                cache.unlink()

        if cache.exists():
            try:
                data = json.loads(cache.read_text())
            except Exception:
                cache.unlink()
                data = None
        else:
            data = None

        if data is None:
            url = f"https://understat.com/getLeagueData/EPL/{year}"
            headers = {
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            }
            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                data = r.json()
                cache.write_text(json.dumps(data))
            except Exception as e:
                print(f"Warning: Understat fetch failed for {year}: {e}")
                continue

        for m in data.get("dates", []):
            if not m.get("isResult"):
                continue
            try:
                records.append({
                    "HomeTeam": _US_TO_FD.get(m["h"]["title"], m["h"]["title"]),
                    "AwayTeam": _US_TO_FD.get(m["a"]["title"], m["a"]["title"]),
                    "Date":     pd.to_datetime(m["datetime"]).normalize(),
                    "xg_h_us": float(m["xG"]["h"]),
                    "xg_a_us": float(m["xG"]["a"]),
                })
            except (KeyError, ValueError):
                continue

    if not records:
        return pd.DataFrame(columns=["HomeTeam", "AwayTeam", "Date", "xg_h_us", "xg_a_us"])
    return pd.DataFrame(records)


def load_data() -> pd.DataFrame:
    """Download Premier League CSVs and return a clean combined DataFrame."""
    DATA_DIR.mkdir(exist_ok=True)
    dfs = []

    for season, url in SEASONS.items():
        fname = season.replace("-", "") + ".csv"
        path  = DATA_DIR / fname

        # Always re-download live seasons
        if season in _LIVE_SEASONS and path.exists():
            path.unlink()

        if not path.exists():
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                path.write_text(r.text, encoding="utf-8")
            except Exception as e:
                print(f"Warning: could not download {season}: {e}")
                continue

        try:
            df = pd.read_csv(path, encoding="latin-1")
            df["Season"] = season
            dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {season}: {e}")

    if not dfs:
        raise RuntimeError("No data loaded — check internet connection.")

    raw = pd.concat(dfs, ignore_index=True)

    keep = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HTHG", "HTAG", "HTR"]   # half-time goals/result for equaliser study
    for col in ["HS", "AS", "HST", "AST",
                "B365H", "B365D", "B365A",
                "MaxH", "MaxD", "MaxA",
                "B365>2.5", "B365<2.5",
                # Closing-line columns for CLV (Pinnacle = sharp benchmark)
                "PSH", "PSD", "PSA",
                "AvgH", "AvgD", "AvgA",
                "P>2.5", "P<2.5",
                "Avg>2.5", "Avg<2.5"]:
        if col in raw.columns:
            keep.append(col)

    df = raw[[c for c in keep if c in raw.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce").astype("Int64")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    df["Result"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    df = df.sort_values("Date").reset_index(drop=True)

    # ── xG enrichment ────────────────────────────────────────────────────────
    us_df = _fetch_understat_xg()
    if len(us_df) > 0:
        df = df.merge(us_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")
        print(f"  Understat xG matched {df['xg_h_us'].notna().sum()}/{len(df)} rows")
    else:
        df["xg_h_us"] = np.nan
        df["xg_a_us"] = np.nan

    # Record per-season real-xG coverage so the app can surface a degraded
    # state (Understat down → model silently on shots-conversion proxy).
    XG_COVERAGE.clear()
    for season, grp in df.groupby("Season"):
        XG_COVERAGE[str(season)] = {
            "matched": int(grp["xg_h_us"].notna().sum()),
            "total":   int(len(grp)),
        }

    if "HST" in df.columns and "AST" in df.columns:
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        valid = df[["HST", "AST", "FTHG", "FTAG"]].dropna()
        if len(valid) > 0 and valid["HST"].sum() > 0:
            cr_h = valid["FTHG"].sum() / valid["HST"].sum()
            cr_a = valid["FTAG"].sum() / valid["AST"].sum()
            sot_h = df["HST"] * cr_h
            sot_a = df["AST"] * cr_a
        else:
            sot_h = df["FTHG"].astype(float)
            sot_a = df["FTAG"].astype(float)
    else:
        sot_h = df["FTHG"].astype(float)
        sot_a = df["FTAG"].astype(float)

    df["xg_h"] = df["xg_h_us"].fillna(sot_h).fillna(df["FTHG"].astype(float))
    df["xg_a"] = df["xg_a_us"].fillna(sot_a).fillna(df["FTAG"].astype(float))
    df = df.drop(columns=["xg_h_us", "xg_a_us"], errors="ignore")

    # ── Days since last match per team ───────────────────────────────────────
    home_d = df[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"})
    away_d = df[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})
    all_d  = pd.concat([home_d, away_d], ignore_index=True).sort_values(["Team", "Date"])
    all_d["days_rest"] = all_d.groupby("Team")["Date"].diff().dt.days
    median_rest = all_d["days_rest"].median()
    all_d["days_rest"] = all_d["days_rest"].fillna(median_rest)
    all_d = all_d.drop_duplicates(subset=["Date", "Team"])

    df = df.merge(
        all_d.rename(columns={"Team": "HomeTeam", "days_rest": "home_days_rest"}),
        on=["Date", "HomeTeam"], how="left",
    )
    df = df.merge(
        all_d.rename(columns={"Team": "AwayTeam", "days_rest": "away_days_rest"}),
        on=["Date", "AwayTeam"], how="left",
    )
    df["home_days_rest"] = df["home_days_rest"].fillna(median_rest)
    df["away_days_rest"] = df["away_days_rest"].fillna(median_rest)

    # ── Elo ratings (pre-match) ───────────────────────────────────────────
    elo_records, _ = _compute_elo_series(df)
    df = df.merge(elo_records, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    # Feed the fuzzy team-name resolver with every name the CSVs use
    KNOWN_FD_TEAMS.update(df["HomeTeam"].unique())
    KNOWN_FD_TEAMS.update(df["AwayTeam"].unique())
    load_promoted_team_names()   # so promoted sides resolve from August

    return df


def get_current_teams(df: pd.DataFrame) -> list[str]:
    """Return the 20 teams active in the current (most recent) season."""
    current_season = df["Season"].max()
    current_df = df[df["Season"] == current_season]
    return sorted(set(current_df["HomeTeam"]) | set(current_df["AwayTeam"]))


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Append rolling EWM form columns to every match row.
    Shift by 1 to prevent lookahead — stats represent form *before* the match.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    has_xg = "xg_h" in df.columns

    home = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
    home.columns = ["Date", "Team", "GF", "GA", "FTR_raw"]
    home["Venue"]    = "H"
    home["Points"]   = home["FTR_raw"].map({"H": 3, "D": 1, "A": 0})
    home["DrawFlag"] = (home["FTR_raw"] == "D").astype(float)
    if has_xg:
        home["xG"]  = df["xg_h"].values
        home["xGA"] = df["xg_a"].values

    away = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR"]].copy()
    away.columns = ["Date", "Team", "GF", "GA", "FTR_raw"]
    away["Venue"]    = "A"
    away["Points"]   = away["FTR_raw"].map({"A": 3, "D": 1, "H": 0})
    away["DrawFlag"] = (away["FTR_raw"] == "D").astype(float)
    if has_xg:
        away["xG"]  = df["xg_a"].values
        away["xGA"] = df["xg_h"].values

    long = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["Team", "Date"])
        .reset_index(drop=True)
    )

    roll_cols = ["GF", "GA", "Points", "DrawFlag"] + (["xG", "xGA"] if has_xg else [])
    for col in roll_cols:
        long[f"roll_{col}"] = long.groupby("Team")[col].transform(
            lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
        )

    base  = ["Date", "Team", "roll_GF", "roll_GA", "roll_Points", "roll_DrawFlag"]
    xg_c  = ["roll_xG", "roll_xGA"] if has_xg else []

    home_feats = (
        long[long["Venue"] == "H"][base + xg_c]
        .drop_duplicates(subset=["Date", "Team"])
        .rename(columns={
            "Team": "HomeTeam",
            "roll_GF": "home_roll_gf", "roll_GA": "home_roll_ga", "roll_Points": "home_roll_pts",
            "roll_DrawFlag": "home_draw_rate",
            **({ "roll_xG": "home_roll_xg", "roll_xGA": "home_roll_xga" } if has_xg else {}),
        })
    )
    away_feats = (
        long[long["Venue"] == "A"][base + xg_c]
        .drop_duplicates(subset=["Date", "Team"])
        .rename(columns={
            "Team": "AwayTeam",
            "roll_GF": "away_roll_gf", "roll_GA": "away_roll_ga", "roll_Points": "away_roll_pts",
            "roll_DrawFlag": "away_draw_rate",
            **({ "roll_xG": "away_roll_xg", "roll_xGA": "away_roll_xga" } if has_xg else {}),
        })
    )

    df = df.merge(home_feats, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_feats, on=["Date", "AwayTeam"], how="left")

    # ── Venue-specific rolling form ───────────────────────────────────────
    # Home team's form in home games only (home_venue_*), away team's in away games only
    home_only = long[long["Venue"] == "H"].copy()
    away_only = long[long["Venue"] == "A"].copy()

    for col in ["GF", "GA", "Points", "DrawFlag"]:
        home_only[f"hv_{col}"] = home_only.groupby("Team")[col].transform(
            lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
        )
        away_only[f"av_{col}"] = away_only.groupby("Team")[col].transform(
            lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
        )

    home_venue_feats = (
        home_only[["Date", "Team", "hv_GF", "hv_GA", "hv_Points", "hv_DrawFlag"]]
        .drop_duplicates(subset=["Date", "Team"])
        .rename(columns={
            "Team": "HomeTeam",
            "hv_GF": "home_venue_gf", "hv_GA": "home_venue_ga", "hv_Points": "home_venue_pts",
            "hv_DrawFlag": "home_venue_draw_rate",
        })
    )
    away_venue_feats = (
        away_only[["Date", "Team", "av_GF", "av_GA", "av_Points", "av_DrawFlag"]]
        .drop_duplicates(subset=["Date", "Team"])
        .rename(columns={
            "Team": "AwayTeam",
            "av_GF": "away_venue_gf", "av_GA": "away_venue_ga", "av_Points": "away_venue_pts",
            "av_DrawFlag": "away_venue_draw_rate",
        })
    )

    df = df.merge(home_venue_feats, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_venue_feats, on=["Date", "AwayTeam"], how="left")

    # ── Elo differential ─────────────────────────────────────────────────
    if "home_elo" in df.columns and "away_elo" in df.columns:
        df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # ── Draw-specific features ────────────────────────────────────────────
    if "home_roll_xg" in df.columns and "away_roll_xg" in df.columns:
        df["xg_convergence"] = (df["home_roll_xg"] - df["away_roll_xg"]).abs()
    if "elo_diff" in df.columns:
        df["elo_diff_abs"] = df["elo_diff"].abs()
    if "home_draw_rate" in df.columns and "away_draw_rate" in df.columns:
        df["both_draw_prone"] = df["home_draw_rate"] * df["away_draw_rate"]

    return df


def get_team_form(df: pd.DataFrame, team: str, n: int = 5) -> list[dict]:
    """Return the last N results for a team, enriched with opponent and venue."""
    mask   = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    recent = df[mask].tail(n)
    rows   = []
    for _, row in recent.iterrows():
        is_home = row["HomeTeam"] == team
        if is_home:
            gf, ga = int(row["FTHG"]), int(row["FTAG"])
            result   = {"H": "W", "D": "D", "A": "L"}[row["FTR"]]
            opponent = row["AwayTeam"]
            venue    = "H"
        else:
            gf, ga = int(row["FTAG"]), int(row["FTHG"])
            result   = {"A": "W", "D": "D", "H": "L"}[row["FTR"]]
            opponent = row["HomeTeam"]
            venue    = "A"
        rows.append({"date": row["Date"].strftime("%d %b"), "opponent": opponent,
                     "venue": venue, "gf": gf, "ga": ga, "result": result})
    return rows


def get_head_to_head(df: pd.DataFrame, home: str, away: str, n: int = 10) -> pd.DataFrame:
    """Return last N meetings between two teams (either venue)."""
    mask = (
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
        | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    )
    return df[mask].tail(n).copy()


def get_current_stats(
    df: pd.DataFrame,
    team: str,
    n: int = 6,
    elo_dict: dict | None = None,
) -> dict:
    """Compute a team's rolling averages from their last N matches.
    Optionally accepts a pre-computed elo_dict for Elo rating lookup.
    """
    mask   = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    recent = df[mask].tail(n)
    has_xg = "xg_h" in df.columns

    avg_gf_default = float(df["FTHG"].mean())
    avg_ga_default = float(df["FTAG"].mean())

    if len(recent) == 0:
        elo = elo_dict.get(team, 1500.0) if elo_dict else 1500.0
        return {
            "avg_gf":    avg_gf_default,
            "avg_ga":    avg_ga_default,
            "avg_pts":   1.2,
            "avg_xg":    float(df["xg_h"].mean()) if has_xg else avg_gf_default,
            "avg_xga":   float(df["xg_a"].mean()) if has_xg else avg_ga_default,
            "days_rest": 7.0,
            "home_venue_gf": avg_gf_default, "home_venue_ga": avg_ga_default, "home_venue_pts": 1.2,
            "away_venue_gf": avg_gf_default, "away_venue_ga": avg_ga_default, "away_venue_pts": 1.2,
            "draw_rate": 0.27,
            "elo": elo,
        }

    gf_l, ga_l, pts_l, xg_l, xga_l, draw_l = [], [], [], [], [], []
    for _, row in recent.iterrows():
        is_home = row["HomeTeam"] == team
        if is_home:
            gf, ga = row["FTHG"], row["FTAG"]
            pts = {"H": 3, "D": 1, "A": 0}[row["FTR"]]
            if has_xg: xg_l.append(row["xg_h"]); xga_l.append(row["xg_a"])
        else:
            gf, ga = row["FTAG"], row["FTHG"]
            pts = {"A": 3, "D": 1, "H": 0}[row["FTR"]]
            if has_xg: xg_l.append(row["xg_a"]); xga_l.append(row["xg_h"])
        gf_l.append(gf); ga_l.append(ga); pts_l.append(pts)
        draw_l.append(1.0 if row["FTR"] == "D" else 0.0)

    # Venue-specific stats
    home_games = df[df["HomeTeam"] == team].tail(n)
    away_games = df[df["AwayTeam"] == team].tail(n)

    def _venue_stats(games, gf_col, ga_col, pts_map):
        gf = [int(r[gf_col]) for _, r in games.iterrows()] if len(games) > 0 else [avg_gf_default]
        ga = [int(r[ga_col]) for _, r in games.iterrows()] if len(games) > 0 else [avg_ga_default]
        pts = [pts_map[r["FTR"]] for _, r in games.iterrows()] if len(games) > 0 else [1.2]
        return float(np.mean(gf)), float(np.mean(ga)), float(np.mean(pts))

    h_vgf, h_vga, h_vpts = _venue_stats(home_games, "FTHG", "FTAG", {"H": 3, "D": 1, "A": 0})
    a_vgf, a_vga, a_vpts = _venue_stats(away_games, "FTAG", "FTHG", {"A": 3, "D": 1, "H": 0})

    # Venue-specific draw rates
    h_vdraw = float(np.mean([1.0 if r["FTR"] == "D" else 0.0 for _, r in home_games.iterrows()])) if len(home_games) > 0 else 0.27
    a_vdraw = float(np.mean([1.0 if r["FTR"] == "D" else 0.0 for _, r in away_games.iterrows()])) if len(away_games) > 0 else 0.27

    last_date   = df[mask]["Date"].max()
    latest_date = df["Date"].max()
    days_rest   = float(min((latest_date - last_date).days, 21))
    avg_gf = float(np.mean(gf_l))
    avg_ga = float(np.mean(ga_l))

    elo = elo_dict.get(team, 1500.0) if elo_dict else 1500.0

    return {
        "avg_gf":    avg_gf,
        "avg_ga":    avg_ga,
        "avg_pts":   float(np.mean(pts_l)),
        "avg_xg":    float(np.mean(xg_l))  if xg_l  else avg_gf,
        "avg_xga":   float(np.mean(xga_l)) if xga_l else avg_ga,
        "days_rest": days_rest,
        "home_venue_gf": h_vgf, "home_venue_ga": h_vga, "home_venue_pts": h_vpts,
        "away_venue_gf": a_vgf, "away_venue_ga": a_vga, "away_venue_pts": a_vpts,
        "draw_rate": float(np.mean(draw_l)),
        "home_venue_draw_rate": h_vdraw,
        "away_venue_draw_rate": a_vdraw,
        "elo": elo,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upcoming fixtures scraper (ESPN public API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_upcoming_fixtures(lookahead_days: int = 30) -> list[dict]:
    """
    Fetch upcoming Premier League fixtures from ESPN's public scoreboard API.
    Returns fixtures grouped into the next gameweek (cluster of match dates).
    Each fixture: {"home": str, "away": str, "date": date, "time_utc": str}
    Uses date-range query to fetch all fixtures in a single request.
    """
    today = date.today()
    end = today + timedelta(days=lookahead_days)
    all_fixtures: list[dict] = []

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
        f"?dates={today.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    )
    data = _fetch_json_with_backoff(url, timeout=15)
    if data is None:
        return []

    for event in data.get("events", []):
        try:
            comps  = event["competitions"][0]
            status = comps["status"]["type"]["name"]
            if status == "STATUS_POSTPONED":
                continue
            # Only include scheduled (unplayed) fixtures
            if status not in ("STATUS_SCHEDULED",):
                continue

            home_name = away_name = None
            event_date = datetime.fromisoformat(
                event["date"].replace("Z", "+00:00")
            ).date()
            for comp in comps["competitors"]:
                raw = comp["team"]["displayName"]
                name = _resolve_team_name(raw)
                if comp["homeAway"] == "home":
                    home_name = name
                else:
                    away_name = name

            if home_name and away_name:
                all_fixtures.append({
                    "home":     home_name,
                    "away":     away_name,
                    "date":     event_date,
                    "time_utc": event["date"],
                    "status":   status,
                })
        except (KeyError, IndexError):
            continue

    if not all_fixtures:
        return []

    # Group into the next gameweek: find the earliest match date,
    # then include all matches within 4 days of that date.
    dates_with_games = sorted({f["date"] for f in all_fixtures})
    first_date = dates_with_games[0]
    cutoff = first_date + timedelta(days=4)
    gameweek = [f for f in all_fixtures if f["date"] <= cutoff]

    # Sort by date then kick-off time
    gameweek.sort(key=lambda x: (x["date"], x["time_utc"]))
    return gameweek


def team_match_counts(df: pd.DataFrame) -> dict[str, int]:
    """How many matches each team has in the loaded dataset, home plus away.

    Drives the no-history gate: a newly promoted side has no Dixon-Coles rating
    and no Elo until it has played enough top-flight matches for the model to
    learn anything, and a team absent from the data is simply missing from this
    dict rather than present with a zero.
    """
    if df.empty or "HomeTeam" not in df.columns:
        return {}
    counts = df["HomeTeam"].value_counts().add(
        df["AwayTeam"].value_counts(), fill_value=0)
    return {team: int(n) for team, n in counts.items()}


def team_rated_from(df: pd.DataFrame, min_matches: int) -> dict[str, pd.Timestamp]:
    """The date each team reached `min_matches` top-flight matches.

    The live gate can use final counts, because every match in the dataset has
    already happened. A backtest cannot: gating on final counts would treat a
    promoted side as rated during its own first matches, which is look-ahead of
    exactly the kind that produced the £100k calibration artifact. Teams that
    never reach the threshold are absent from the result.
    """
    if df.empty or "HomeTeam" not in df.columns or min_matches < 1:
        return {}
    appearances = pd.concat([
        df[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
        df[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"}),
    ]).sort_values("Date")

    out: dict[str, pd.Timestamp] = {}
    for team, group in appearances.groupby("Team", sort=False):
        if len(group) >= min_matches:
            out[team] = group["Date"].iloc[min_matches - 1]
    return out


def get_current_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the current live league table from all matches in df.
    Returns DataFrame sorted by position with columns:
        Team, Played, W, D, L, GF, GA, GD, Pts
    Only includes teams from the most recent season.
    """
    current_season = df["Season"].max()
    season_df = df[df["Season"] == current_season].copy()

    rows = []
    for _, r in season_df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag, ftr = int(r["FTHG"]), int(r["FTAG"]), r["FTR"]
        rows.append({"Team": h, "W": ftr=="H", "D": ftr=="D", "L": ftr=="A", "GF": hg, "GA": ag})
        rows.append({"Team": a, "W": ftr=="A", "D": ftr=="D", "L": ftr=="H", "GF": ag, "GA": hg})

    tbl = (
        pd.DataFrame(rows)
        .groupby("Team")
        .agg(Played=("W","count"), W=("W","sum"), D=("D","sum"), L=("L","sum"),
             GF=("GF","sum"), GA=("GA","sum"))
        .assign(GD=lambda x: x["GF"] - x["GA"],
                Pts=lambda x: x["W"]*3 + x["D"])
        .sort_values(["Pts","GD","GF"], ascending=False)
        .reset_index()
    )
    return tbl


def fetch_remaining_season_fixtures(season_end_date: str | None = None) -> list[dict]:
    """
    Fetch all remaining unplayed PL fixtures through the end of the season.
    Results are cached locally for 12 hours (daily schedule rarely changes).
    Returns list of {"home": str, "away": str, "date": date, "time_utc": str}

    `season_end_date` defaults to June 1 of the current season's end year —
    computed dynamically so it rolls over each July with no manual edit
    (the buffer past late May also catches rescheduled final fixtures).
    """
    if season_end_date is None:
        season_end_date = date(_season_start_year() + 1, 6, 1).isoformat()
    cache_path = DATA_DIR / "remaining_fixtures.json"

    # Return cache if fresh enough
    if cache_path.exists():
        age_h = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_h < 12:
            try:
                raw = json.loads(cache_path.read_text())
                today = date.today()
                return [
                    {**f, "date": date.fromisoformat(f["date"])}
                    for f in raw
                    if date.fromisoformat(f["date"]) >= today
                ]
            except Exception:
                pass

    today = date.today()
    end   = date.fromisoformat(season_end_date)
    all_fixtures: list[dict] = []

    d = today
    while d <= end:
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
            f"?dates={d.strftime('%Y%m%d')}"
        )
        data = _fetch_json_with_backoff(url, timeout=10, retries=2)
        if data is None:
            d += timedelta(days=1)
            continue

        for event in data.get("events", []):
            try:
                comps  = event["competitions"][0]
                status = comps["status"]["type"]["name"]
                # Only keep scheduled / unplayed matches
                if status not in ("STATUS_SCHEDULED", "STATUS_IN_PROGRESS"):
                    d += timedelta(days=0)  # no-op, just continue
                    continue

                home_name = away_name = None
                for comp in comps["competitors"]:
                    raw_name = comp["team"]["displayName"]
                    name = _resolve_team_name(raw_name)
                    if comp["homeAway"] == "home":
                        home_name = name
                    else:
                        away_name = name

                if home_name and away_name:
                    all_fixtures.append({
                        "home":     home_name,
                        "away":     away_name,
                        "date":     d.isoformat(),
                        "time_utc": event["date"],
                    })
            except (KeyError, IndexError):
                continue

        d += timedelta(days=1)

    # Persist cache (dates stored as ISO strings)
    try:
        DATA_DIR.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(all_fixtures))
    except Exception:
        pass

    return [
        {**f, "date": date.fromisoformat(f["date"])}
        for f in all_fixtures
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Team motivation state — captures what each team has to play for at a point
# in the season. The empirical Mar-Apr collapse (0/7 across 2024-25 + 2025-26)
# suggests fixed-form models miss the regime change in late season — top teams
# push for wins, dead-rubber mid-tablers play loose, relegation-threatened sides
# play with desperation. This feature lets the model see those states.
# ─────────────────────────────────────────────────────────────────────────────

_MOTIVATION_STATES = ["title_race", "europe_race", "survival",
                      "dead_rubber", "mid_active", "unknown"]


def _classify_motivation(team: str, table: dict, season_progress: float) -> str:
    """Categorise a team's motivation given the league snapshot at match time.

    table: {team: {pos, pts, gd, played}}; season_progress in [0,1].
    """
    if team not in table or season_progress < 0.10:
        return "unknown"
    t   = table[team]
    pos = t["pos"]; pts = t["pts"]
    n_teams = len(table)

    pts_sorted = sorted([v["pts"] for v in table.values()], reverse=True)
    leader_pts = pts_sorted[0]
    pts_at_7   = pts_sorted[6]  if n_teams >= 7  else 0
    pts_at_17  = pts_sorted[16] if n_teams >= 17 else 0

    if pos <= 2 and (leader_pts - pts) <= 6 and season_progress > 0.50:
        return "title_race"
    if 3 <= pos <= 7 and abs(pts - pts_at_7) <= 6 and season_progress > 0.30:
        return "europe_race"
    if pos >= 14 and (pts - pts_at_17) <= 6 and season_progress > 0.30:
        return "survival"
    if (8 <= pos <= 13
        and (pts_at_7  - pts) > 10
        and (pts - pts_at_17) > 10
        and season_progress > 0.60):
        return "dead_rubber"
    return "mid_active"


def add_motivation_features(df_features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add team-motivation features to df_features.

    For each match, computes both teams' league position, games remaining, and
    motivation category given the table snapshot at kickoff (no look-ahead —
    only matches strictly *before* the current date in the same season are used).

    New columns:
      h_pos_at_match, a_pos_at_match
      h_games_remaining, a_games_remaining
      h_mot_<state>, a_mot_<state>       one-hot 0/1 for each motivation category
    """
    out = df_features.copy()
    out["h_pos_at_match"]    = -1
    out["a_pos_at_match"]    = -1
    out["h_games_remaining"] = -1
    out["a_games_remaining"] = -1
    out["h_motivation"]      = "unknown"
    out["a_motivation"]      = "unknown"

    df_sorted = df.sort_values(["Season", "Date"]).reset_index(drop=True)
    # df_features and df share Date+HomeTeam by construction — index by (Date, HomeTeam)
    out_idx_by_dh = {(r["Date"], r["HomeTeam"]): i for i, r in out.iterrows()}

    for season, season_df in df_sorted.groupby("Season"):
        running: dict[str, dict] = {}
        for _, match in season_df.sort_values("Date").iterrows():
            home, away, dt = match["HomeTeam"], match["AwayTeam"], match["Date"]

            # Snapshot table BEFORE this match
            table = {}
            if running:
                rows = [{"team": tm, "pts": s["pts"], "gd": s["gf"]-s["ga"],
                         "gf": s["gf"], "played": s["played"]}
                        for tm, s in running.items()]
                tbl_df = pd.DataFrame(rows).sort_values(
                    ["pts","gd","gf"], ascending=False).reset_index(drop=True)
                for pos, row in tbl_df.iterrows():
                    table[row["team"]] = {"pos": pos+1, "pts": row["pts"],
                                          "gd": row["gd"], "played": row["played"]}

            n_teams = len(table) if table else 20
            total_games = (n_teams - 1) * 2  # 38 for full PL
            home_played = table.get(home, {}).get("played", 0)
            away_played = table.get(away, {}).get("played", 0)
            season_progress = (sum(s["played"] for s in table.values()) /
                               max(1, len(table) * total_games)) if table else 0

            home_state = _classify_motivation(home, table, season_progress)
            away_state = _classify_motivation(away, table, season_progress)

            idx = out_idx_by_dh.get((dt, home))
            if idx is not None:
                out.at[idx, "h_pos_at_match"]    = table.get(home, {}).get("pos", -1)
                out.at[idx, "a_pos_at_match"]    = table.get(away, {}).get("pos", -1)
                out.at[idx, "h_games_remaining"] = total_games - home_played
                out.at[idx, "a_games_remaining"] = total_games - away_played
                out.at[idx, "h_motivation"]      = home_state
                out.at[idx, "a_motivation"]      = away_state

            # Update running totals with the result (post-snapshot, no leak)
            try:
                hg, ag = int(match["FTHG"]), int(match["FTAG"])
            except (ValueError, TypeError):
                continue
            ftr = match["FTR"]
            for tm, gf, ga, pts_won in [
                (home, hg, ag, 3 if ftr=="H" else (1 if ftr=="D" else 0)),
                (away, ag, hg, 3 if ftr=="A" else (1 if ftr=="D" else 0)),
            ]:
                if tm not in running:
                    running[tm] = {"played": 0, "pts": 0, "gf": 0, "ga": 0}
                running[tm]["played"] += 1
                running[tm]["pts"]    += pts_won
                running[tm]["gf"]     += gf
                running[tm]["ga"]     += ga

    # One-hot for XGB
    for state in _MOTIVATION_STATES:
        out[f"h_mot_{state}"] = (out["h_motivation"] == state).astype(int)
        out[f"a_mot_{state}"] = (out["a_motivation"] == state).astype(int)

    return out
