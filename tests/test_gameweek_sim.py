"""Gameweek mode in ev_backtest_simulate, and the pending-stake cap.

The day-settled simulator compounds Saturday's winnings into Sunday's stakes
and never holds a pending bet, so it cannot test a cap on pending stake. The
gameweek mode stakes a whole round from its opening bankroll and settles at the
end, as live does. These pin down the grouping, both cap bases, and that the
default path is untouched.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import portfolio as pf

ROOT = Path(__file__).resolve().parent.parent


def _frames(fixtures):
    """fixtures: (date, home, away, draw_prob_pct, draw_odds, result)."""
    bt, df = [], []
    for date, home, away, p_d, odds, res in fixtures:
        d = pd.Timestamp(date)
        p_h = (100 - p_d) / 2
        actual = {"D": "Draw", "H": "Home Win", "A": "Away Win"}[res]
        bt.append({"Date": d, "Home": home, "Away": away, "Actual": actual,
                   "DC_H": p_h, "DC_D": p_d, "DC_A": p_h})
        hg, ag = {"D": (1, 1), "H": (2, 0), "A": (0, 2)}[res]
        df.append({"Date": d, "HomeTeam": home, "AwayTeam": away,
                   "FTHG": hg, "FTAG": ag,
                   "MaxH": 3.0, "MaxD": odds, "MaxA": 3.0,
                   "PSH": 3.0, "PSD": odds, "PSA": 3.0})
    return pd.DataFrame(bt), pd.DataFrame(df)


KW = dict(min_ev_pct=10.0, kelly_frac=1.0, max_stake_pct=0.25,
          initial_bankroll=10_000.0, allowed_markets={"D"},
          odds_source="Max", detect_source="PS")

# One weekend (Sat + Sun), then a midweek round, then the next weekend.
FIXTURES = [
    ("2025-09-13", "A", "B", 40, 4.0, "D"),   # Sat
    ("2025-09-13", "C", "D", 40, 4.0, "H"),   # Sat
    ("2025-09-14", "E", "F", 40, 4.0, "H"),   # Sun
    ("2025-09-16", "G", "H", 40, 4.0, "D"),   # Tue (midweek)
    ("2025-09-20", "I", "J", 40, 4.0, "H"),   # next Sat
]


def test_gameweek_blocks_split_weekend_midweek_and_next_weekend():
    dates = pd.Series(pd.to_datetime(
        ["2025-09-12", "2025-09-13", "2025-09-14", "2025-09-15",   # Fri-Mon
         "2025-09-16", "2025-09-17",                               # Tue-Wed
         "2025-09-20"]))                                           # Sat
    assert pf._gameweek_blocks(dates).tolist() == [0, 0, 0, 0, 1, 1, 2]


def test_gameweek_mode_stakes_the_weekend_from_one_bankroll():
    bt, df = _frames(FIXTURES)
    log, _ = pf.ev_backtest_simulate(bt, df, gameweek_mode=True, **KW)
    weekend = log[log["Date"] <= pd.Timestamp("2025-09-14")]
    # Kelly at p=0.40, odds 4.0 is 20%. Each stake comes off the cash left,
    # none from settled winnings: 2000, then 1600, then 1280.
    assert weekend["Stake"].tolist() == [2000.0, 1600.0, 1280.0]


def test_opening_cap_limits_pending_stake_to_half_the_opening_bankroll():
    bt, df = _frames(FIXTURES[:3] + [
        ("2025-09-14", "K", "L", 40, 4.0, "H"),
        ("2025-09-14", "M", "N", 40, 4.0, "H"),
    ])
    log, summary = pf.ev_backtest_simulate(
        bt, df, gameweek_mode=True, exposure_cap_pct=0.30,
        exposure_cap_basis="opening", **KW)
    assert log["Stake"].sum() == pytest.approx(3000.0)   # 30% of 10,000
    assert summary["skipped_exposure_cap"] >= 1


def test_gameweek_cap_grows_with_the_bankroll_and_opening_cap_does_not():
    # Weekend 1 is a big win; weekend 2 has five qualifying bets.
    wk1 = [("2025-09-13", "A", "B", 40, 10.0, "D")]
    wk2 = [("2025-09-20", f"H{i}", f"A{i}", 40, 4.0, "H") for i in range(5)]
    bt, df = _frames(wk1 + wk2)
    common = dict(gameweek_mode=True, exposure_cap_pct=0.30, **KW)

    log_open, _ = pf.ev_backtest_simulate(bt, df, exposure_cap_basis="opening", **common)
    log_gw, _ = pf.ev_backtest_simulate(bt, df, exposure_cap_basis="gameweek", **common)

    start_wk2 = log_gw.iloc[0]["Bankroll"]
    wk2_open = log_open[log_open["Date"] == pd.Timestamp("2025-09-20")]["Stake"].sum()
    wk2_gw = log_gw[log_gw["Date"] == pd.Timestamp("2025-09-20")]["Stake"].sum()
    assert wk2_open == pytest.approx(3000.0)
    assert wk2_gw == pytest.approx(0.30 * start_wk2, abs=0.05)
    assert wk2_gw > wk2_open


def test_caps_refuse_to_run_without_gameweek_mode():
    bt, df = _frames(FIXTURES)
    with pytest.raises(ValueError):
        pf.ev_backtest_simulate(bt, df, exposure_cap_pct=0.5, **KW)


def test_default_day_settled_path_matches_the_committed_simulator(tmp_path):
    """Nothing validated before today may move: the default path must produce
    the identical bet log to portfolio.py as it was before gameweek mode.
    Pinned to that commit, since HEAD moves on."""
    src = subprocess.run(["git", "show", "eefb4d3:portfolio.py"], cwd=ROOT,
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip("no git history")
    old_file = tmp_path / "portfolio_head.py"
    old_file.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("portfolio_head", old_file)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

    bt, df = _frames(FIXTURES)
    new_log, new_sum = pf.ev_backtest_simulate(bt, df, **KW)
    old_log, old_sum = old.ev_backtest_simulate(bt, df, **KW)
    pd.testing.assert_frame_equal(new_log, old_log)
    new_sum.pop("skipped_exposure_cap")
    assert new_sum == old_sum
