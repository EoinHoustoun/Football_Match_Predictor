"""Shared fixtures for F_PRED tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable (models.py / portfolio.py sit at repo root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def toy_dc_ratings() -> dict:
    """A hand-crafted Dixon-Coles ratings dict for 4 teams.

    Team A: best attack, worst defence
    Team D: worst attack, best defence
    Home advantage ~1.3x (exp(0.26) ≈ 1.3)
    """
    teams = ["A", "B", "C", "D"]
    return {
        "teams": teams,
        "team_idx": {t: i for i, t in enumerate(teams)},
        "attacks":  {"A": 0.40, "B": 0.10, "C": -0.10, "D": -0.40},
        "defenses": {"A": 0.20, "B": 0.05, "C": -0.05, "D": -0.20},
        "home_adv": 0.26,
        "rho": -0.10,
        "converged": True,
    }


@pytest.fixture
def toy_current_table() -> pd.DataFrame:
    """Synthetic current league table for 4 teams, partway through the season."""
    return pd.DataFrame({
        "Team":    ["A", "B", "C", "D"],
        "Pts":     [40, 35, 20, 10],
        "GF":      [30, 25, 15, 10],
        "GA":      [10, 15, 20, 25],
        "GD":      [20, 10, -5, -15],
        "Played":  [15, 15, 15, 15],
    })


@pytest.fixture
def toy_remaining_fixtures() -> list[dict]:
    """Round-robin over 4 teams — each pair plays once more."""
    teams = ["A", "B", "C", "D"]
    fixtures = []
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i != j:
                fixtures.append({"home": h, "away": a})
    return fixtures


@pytest.fixture
def synthetic_match_history() -> pd.DataFrame:
    """~200 matches between 4 teams, deterministic via a fixed RNG.

    Goals sampled from Poisson(1.3) to give realistic-ish counts.
    Dates span the last year so the DC exponential-decay weighting has signal.
    """
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D"]
    rows = []
    start = pd.Timestamp("2025-05-01")
    for k in range(200):
        h, a = rng.choice(teams, size=2, replace=False)
        hg = int(rng.poisson(1.5))
        ag = int(rng.poisson(1.1))
        rows.append({
            "Date":      start + pd.Timedelta(days=k),
            "HomeTeam":  h,
            "AwayTeam":  a,
            "FTHG":      hg,
            "FTAG":      ag,
            "xg_h":      float(hg) + rng.normal(0, 0.2),
            "xg_a":      float(ag) + rng.normal(0, 0.2),
        })
    return pd.DataFrame(rows)
