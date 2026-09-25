"""Model Check: is the live draw model still honest, and where is its edge?

Reads the out-of-sample prediction pack the September validation scripts build
(`data/diagnostics/exposure_cap_pack.pkl`, from scripts/validate_exposure_cap.py):
per season, the full live ensemble's RAW probabilities for every match, fitted
only on data before that season, plus the football-data frame with Pinnacle
closing odds. Nothing here refits a model or touches a portfolio; it only
summarises predictions that already exist.

The page used to backtest two intermediate models on the last 50 matches, so
its reliability bins held two to nineteen matches each. These summaries pool
four-plus seasons and put a confidence interval on every bin.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PACK_PATH = Path("data/diagnostics/exposure_cap_pack.pkl")
DRAW_BINS = [0.0, 0.22, 0.26, 0.30, 0.34, 1.01]
BIN_LABELS = ["under 22%", "22 to 26%", "26 to 30%", "30 to 34%", "34% and up"]
LONG_RUN_DRAW_RATE = 0.22   # EPL 2021-26, the figure the calibrator notes use


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """90% Wilson interval for a proportion (robust at small n and near 0/1)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def load_pack(path: Path = PACK_PATH) -> tuple[pd.DataFrame, float] | tuple[None, None]:
    """One frame of every out-of-sample prediction, joined to Pinnacle closes.

    Returns (frame, pack_mtime) or (None, None) when the pack is missing.
    Columns: Date, Season, Home, Away, p_h, p_d, p_a, drew, psd (closing
    Pinnacle draw price, NaN where missing), mkt_d (margin-free close).
    """
    if not path.exists():
        return None, None
    pack = pickle.loads(path.read_bytes())
    rows = []
    for season, (dfs, _ftr, bt, _cal) in pack.items():
        if bt is None or len(bt) == 0:
            continue
        b = bt[["Date", "Home", "Away", "_dc_h", "_dc_d", "_dc_a", "_act_d"]].copy()
        b["Season"] = season
        cols = [c for c in ("PSH", "PSD", "PSA") if c in dfs.columns]
        if len(cols) == 3:
            m = dfs[["Date", "HomeTeam", "AwayTeam", *cols]].rename(
                columns={"HomeTeam": "Home", "AwayTeam": "Away"})
            b = b.merge(m, on=["Date", "Home", "Away"], how="left")
        rows.append(b)
    if not rows:
        return None, None
    f = pd.concat(rows, ignore_index=True).rename(
        columns={"_dc_h": "p_h", "_dc_d": "p_d", "_dc_a": "p_a", "_act_d": "drew",
                 "PSD": "psd"})
    if {"PSH", "psd", "PSA"} <= set(f.columns):
        inv = 1 / f[["PSH", "psd", "PSA"]]
        f["mkt_d"] = inv["psd"] / inv.sum(axis=1)
    else:
        f["psd"], f["mkt_d"] = np.nan, np.nan
    f = f.sort_values("Date").reset_index(drop=True)
    return f, path.stat().st_mtime


def reliability(f: pd.DataFrame, col: str = "p_d") -> pd.DataFrame:
    """Draw reliability by raw-probability bin, with 90% Wilson intervals."""
    g = f.assign(bin=pd.cut(f[col], DRAW_BINS, labels=BIN_LABELS, right=False))
    out = []
    for lbl, grp in g.groupby("bin", observed=True):
        n, k = len(grp), int(grp["drew"].sum())
        lo, hi = wilson(k, n)
        out.append({"bin": str(lbl), "n": n, "predicted": float(grp[col].mean()),
                    "actual": k / n if n else float("nan"), "lo": lo, "hi": hi,
                    "market": float(grp["mkt_d"].mean()) if grp["mkt_d"].notna().any() else float("nan")})
    return pd.DataFrame(out)


def edge_by_bucket(f: pd.DataFrame) -> pd.DataFrame:
    """Flat £1 on every draw in each raw bucket, settled at the Pinnacle close.

    The close is a harder price than the one bet at, so this understates what
    live betting earns; its job is to show WHERE the edge sits, not how big.
    """
    g = f[f["psd"].notna()].assign(
        bin=lambda x: pd.cut(x["p_d"], DRAW_BINS, labels=BIN_LABELS, right=False))
    out = []
    for lbl, grp in g.groupby("bin", observed=True):
        n = len(grp)
        pnl = float(np.where(grp["drew"] == 1, grp["psd"] - 1, -1).sum())
        out.append({"bin": str(lbl), "n": n, "roi": pnl / n if n else float("nan"),
                    "model": float(grp["p_d"].mean()), "market": float(grp["mkt_d"].mean()),
                    "actual": float(grp["drew"].mean())})
    return pd.DataFrame(out)


def season_table(f: pd.DataFrame) -> pd.DataFrame:
    """Per season: matches, draw rate, model mean, and draw Brier (model v market)."""
    out = []
    for s, g in f.groupby("Season"):
        bm = float(((g["p_d"] - g["drew"]) ** 2).mean())
        has_mkt = g["mkt_d"].notna()
        bk = float(((g.loc[has_mkt, "mkt_d"] - g.loc[has_mkt, "drew"]) ** 2).mean()) if has_mkt.any() else float("nan")
        out.append({"season": s, "n": len(g), "draw_rate": float(g["drew"].mean()),
                    "model_mean": float(g["p_d"].mean()), "brier_model": bm,
                    "brier_market": bk})
    return pd.DataFrame(out)


def rolling_draw_rate(f: pd.DataFrame, window: int = 76) -> pd.DataFrame:
    """Draw rate over the last `window` matches (76 = two rounds of 38)."""
    s = f[["Date", "drew", "p_d"]].copy()
    s["actual"] = s["drew"].rolling(window, min_periods=window // 2).mean()
    s["model"] = s["p_d"].rolling(window, min_periods=window // 2).mean()
    return s.dropna(subset=["actual"])
