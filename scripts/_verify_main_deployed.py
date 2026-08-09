"""Verify the newly deployed Main aggressive config reproduces the £322k
52w + multi-season figures from the sweep. Reads settings from data/portfolio.json
and runs ev_backtest_simulate over (a) 52w continuous (b) per-season for 4 seasons.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                 # noqa: E402
import portfolio as pf                             # noqa: E402


def _max_drawdown_pct(log: pd.DataFrame, init: float) -> float:
    if log.empty: return 0.0
    s = pd.concat([pd.Series([init]), log["Bankroll"].astype(float)],
                  ignore_index=True)
    return float(((s.cummax() - s) / s.cummax()).max() * 100)


def _season_label(d: pd.Timestamp) -> str:
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def main() -> int:
    settings = json.load(open(ROOT / "data" / "portfolio.json"))["settings"]
    print("DEPLOYED MAIN CONFIG:")
    for k in ("min_prob", "main_banned_dows", "main_banned_months",
              "main_elo_gap_min", "main_elo_gap_max",
              "main_min_team_elo", "main_max_team_elo", "main_max_ev_pct",
              "main_settings_label"):
        print(f"  {k}: {settings.get(k)}")
    print()

    cfg = dict(
        min_ev_pct=float(settings.get("min_ev", 0.40)) * 100,
        kelly_frac=float(settings.get("kelly_fraction", 1.0)),
        max_stake_pct=float(settings.get("max_stake_pct", 0.33)),
        initial_bankroll=10_000.0,
        allowed_markets=set(settings.get("auto_markets", ["D", "under25"])),
        min_prob=float(settings.get("min_prob", 0.30)),
        enable_simultaneous_correction=bool(settings.get("use_simultaneous_kelly", True)),
        skip_late_season=bool(settings.get("skip_late_season", True)),
        skip_home_title_race=bool(settings.get("skip_home_title_race", False)),
        odds_source="Max",
        detect_source="PS",
        market_gates=settings.get("market_gates"),
        banned_dows=set(settings.get("main_banned_dows", [])) or None,
        banned_months=set(settings.get("main_banned_months", [])) or None,
        max_ev_pct=settings.get("main_max_ev_pct"),
        min_team_elo=settings.get("main_min_team_elo"),
        max_team_elo=settings.get("main_max_team_elo"),
        elo_gap_min=settings.get("main_elo_gap_min"),
        elo_gap_max=settings.get("main_elo_gap_max"),
    )

    df = load_data()
    df = df.copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    df_features = add_rolling_features(df)

    print(f"[{datetime.now():%H:%M:%S}] 52w continuous backtest…")
    bt52 = backtest_models(df, df_features, test_weeks=52)
    log52, sum52 = pf.ev_backtest_simulate(bt52, df, df_features=df_features, **cfg)
    if "error" in sum52:
        print(f"ERROR: {sum52['error']}")
    else:
        dd = _max_drawdown_pct(log52, 10_000.0)
        print(f"  52w:  £{sum52['final']:>10,.0f}  "
              f"ROI {sum52['roi']:>+6.1f}%  n={sum52['n_bets']:>3}  "
              f"win {sum52['win_rate']:>4.1f}%  DD {dd:.1f}%")

    print(f"\n[{datetime.now():%H:%M:%S}] Per-season validation…")
    seasons = sorted(df["SeasonLbl"].unique())[1:]
    finals = []
    for season in seasons:
        season_end = df[df["SeasonLbl"] == season]["Date"].max()
        df_slice  = df[df["Date"] <= season_end].copy()
        ftr_slice = df_features[df_features["Date"] <= season_end].copy()
        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty: continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty: continue
        log_, sum_ = pf.ev_backtest_simulate(bt_season, df_slice,
                                              df_features=ftr_slice, **cfg)
        if "error" in sum_:
            print(f"  {season}: ERROR")
            continue
        dd_s = _max_drawdown_pct(log_, 10_000.0)
        finals.append(sum_["final"])
        print(f"  {season}:  £{sum_['final']:>10,.0f}  "
              f"ROI {sum_['roi']:>+6.1f}%  n={sum_['n_bets']:>3}  "
              f"win {sum_['win_rate']:>4.1f}%  DD {dd_s:.1f}%")

    if finals:
        med = sorted(finals)[len(finals) // 2]
        print(f"\n  Mean: £{sum(finals)/len(finals):,.0f}")
        print(f"  Median: £{med:,.0f}")
        print(f"  Worst: £{min(finals):,.0f}")
        print(f"  Best:  £{max(finals):,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
