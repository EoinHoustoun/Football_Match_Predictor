"""Validate the deployed Main config on non-EPL leagues.

If the gates (DOWmf + gMin100 + evMax100 + mp32 + skip Mar-Apr) produce
positive results on La Liga, Bundesliga, Serie A, and Championship —
not just EPL — then we know the edge is STRUCTURAL (the model + filters
exploit a general inefficiency), not EPL-specific noise.

If they fail elsewhere, the EPL backtest gains are partly luck/overfit.

This is the cheapest, most informative validation we can run.
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import add_rolling_features            # noqa: E402
from data_multi_league import load_multi_league   # noqa: E402
from models import backtest_models                # noqa: E402
import portfolio as pf                            # noqa: E402

LEAGUES = {
    "E0":  "EPL",
    "E1":  "Championship",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
}


def _max_drawdown_pct(log: pd.DataFrame, init: float) -> float:
    if log.empty: return 0.0
    s = pd.concat([pd.Series([init]), log["Bankroll"].astype(float)],
                  ignore_index=True)
    return float(((s.cummax() - s) / s.cummax()).max() * 100)


def main() -> int:
    settings = json.load(open(ROOT / "data" / "portfolio.json"))["settings"]
    print("DEPLOYED MAIN CONFIG (used as-is across leagues):")
    for k in ("min_prob", "min_ev", "main_banned_dows",
              "main_elo_gap_min", "main_max_ev_pct"):
        print(f"  {k}: {settings.get(k)}")
    print()

    df_all = load_multi_league()
    print(f"Loaded {len(df_all):,} matches across "
          f"{df_all['League'].nunique()} leagues.")
    print()

    cfg = dict(
        min_ev_pct=float(settings.get("min_ev", 0.40)) * 100,
        kelly_frac=float(settings.get("kelly_fraction", 1.0)),
        max_stake_pct=float(settings.get("max_stake_pct", 0.33)),
        initial_bankroll=10_000.0,
        allowed_markets={"D"},  # Restrict to draw market only (most odds available)
        min_prob=float(settings.get("min_prob", 0.32)),
        skip_late_season=bool(settings.get("skip_late_season", True)),
        skip_home_title_race=False,
        odds_source="Max",
        detect_source="PS",
        enable_simultaneous_correction=True,
        market_gates=None,
        banned_dows=set(settings.get("main_banned_dows", [])) or None,
        max_ev_pct=settings.get("main_max_ev_pct"),
        elo_gap_min=settings.get("main_elo_gap_min"),
        # Disable max_team_elo/min_team_elo if not set
        min_team_elo=settings.get("main_min_team_elo"),
        max_team_elo=settings.get("main_max_team_elo"),
        elo_gap_max=settings.get("main_elo_gap_max"),
    )

    results: list[dict] = []
    for code, name in LEAGUES.items():
        print(f"== {name} ({code}) ==")
        lg_df = df_all[df_all["League"] == code].copy()
        lg_df = lg_df.sort_values("Date").reset_index(drop=True)
        # Ensure Season label column for backtest_models compatibility
        # (it expects a Season column — already in our multi-league CSV)
        # Drop any rows missing required cols
        lg_df = lg_df.dropna(subset=["FTHG", "FTAG", "FTR", "HomeTeam", "AwayTeam"])
        # Add Result column (H/D/A → 0/1/2) — what train_xgb needs
        lg_df["Result"] = lg_df["FTR"].map({"H": 0, "D": 1, "A": 2})
        print(f"  matches: {len(lg_df):,}  "
              f"span: {lg_df['Date'].min().date()} → {lg_df['Date'].max().date()}")

        try:
            ftr = add_rolling_features(lg_df)
        except Exception as e:
            print(f"  [SKIP] add_rolling_features failed: {e}")
            continue
        try:
            bt = backtest_models(lg_df, ftr, test_weeks=52)
        except Exception as e:
            print(f"  [SKIP] backtest_models failed: {e}")
            continue
        if bt.empty:
            print(f"  [SKIP] backtest_models returned empty")
            continue

        try:
            log, sum_ = pf.ev_backtest_simulate(
                bt, lg_df, df_features=ftr, **cfg,
            )
        except Exception as e:
            print(f"  [SKIP] ev_backtest_simulate failed: {e}")
            continue
        if "error" in sum_:
            print(f"  [SKIP] sim error: {sum_['error']}")
            continue

        dd = _max_drawdown_pct(log, cfg["initial_bankroll"])
        print(f"  → 52w £{sum_['final']:>9,.0f}  "
              f"ROI {sum_['roi']:>+6.1f}%  "
              f"n={sum_['n_bets']:>3}  "
              f"win {sum_['win_rate']:>4.1f}%  DD {dd:.1f}%")
        results.append({
            "league":   code,
            "name":     name,
            "matches":  int(len(lg_df)),
            "bt_rows":  int(len(bt)),
            "final":    float(sum_["final"]),
            "roi":      float(sum_["roi"]),
            "n_bets":   int(sum_["n_bets"]),
            "win_rate": float(sum_["win_rate"]),
            "dd":       round(dd, 2),
        })
        print()

    print("=" * 80)
    print("CROSS-LEAGUE COMPARISON  (deployed Main config, £10k start)")
    print("=" * 80)
    print(f"{'league':<14}  {'matches':>7}  {'final':>10}  {'ROI':>7}  "
          f"{'n':>4}  {'win%':>5}  {'DD%':>5}  {'verdict':<12}")
    print("-" * 80)
    for r in results:
        verdict = ("strong" if r["final"] > 30_000
                   else "decent" if r["final"] > 15_000
                   else "neutral" if r["final"] > 9_000
                   else "FAILS")
        print(f"{r['name']:<14}  {r['matches']:>7,}  "
              f"£{r['final']:>9,.0f}  {r['roi']:>+6.1f}%  "
              f"{r['n_bets']:>4}  {r['win_rate']:>4.1f}%  {r['dd']:>4.1f}%  "
              f"{verdict:<12}")

    # Summary verdict
    profitable = [r for r in results if r["final"] > 10_000]
    if len(profitable) >= 4:
        print("\n✅ STRUCTURAL EDGE — gates work in 4+/5 leagues. Real signal.")
    elif len(profitable) >= 3:
        print("\n🟡 PARTIAL TRANSFER — gates work in 3/5 leagues. Edge is plausible but not universal.")
    else:
        print("\n⚠ EPL-SPECIFIC — gates don't transfer. Be wary of overfit risk.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
