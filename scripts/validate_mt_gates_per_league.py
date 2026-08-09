"""Cross-league validation for Mock Two's deployed config (min_team_elo=1500).

If Mock Two's ELO floor filter transfers to other leagues, it's a structural
edge (cuts weak teams everywhere). If not, it's also EPL-specific and we
know both portfolios are EPL-specialised.

Uses backtest_models_v2 + ev_backtest_simulate_v2 to match Mock Two's
research-track stack (K-N γ-inflation + uncertainty Kelly).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import add_rolling_features            # noqa: E402
from data_multi_league import load_multi_league   # noqa: E402
from models import backtest_models_v2             # noqa: E402
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
    settings = json.load(open(ROOT / "data" / "portfolio_two.json"))["settings"]
    print("DEPLOYED MOCK TWO CONFIG:")
    for k in ("min_prob", "min_ev", "v2_min_team_elo",
              "v2_elo_gap_min", "v2_elo_gap_max", "v2_max_ev_pct",
              "v2_banned_dows"):
        print(f"  {k}: {settings.get(k)}")
    print()

    df_all = load_multi_league()
    print(f"Loaded {len(df_all):,} matches across "
          f"{df_all['League'].nunique()} leagues.\n")

    cfg = dict(
        min_ev_pct=float(settings.get("min_ev", 0.40)) * 100,
        base_kelly_frac=float(settings.get("kelly_fraction", 1.0)),
        max_stake_pct=float(settings.get("max_stake_pct", 0.33)),
        initial_bankroll=10_000.0,
        allowed_markets={"D"},  # Draws only for fair cross-league
        min_prob=float(settings.get("min_prob", 0.30)),
        enable_simultaneous_correction=True,
        skip_late_season=bool(settings.get("skip_late_season", True)),
        skip_home_title_race=False,
        odds_source="Max",
        detect_source="PS",
        market_gates=None,
        banned_dows=set(settings.get("v2_banned_dows", [])) or None,
        banned_months=set(settings.get("v2_banned_months", [])) or None,
        max_ev_pct=settings.get("v2_max_ev_pct"),
        min_team_elo=settings.get("v2_min_team_elo"),
        max_team_elo=settings.get("v2_max_team_elo"),
        elo_gap_min=settings.get("v2_elo_gap_min"),
        elo_gap_max=settings.get("v2_elo_gap_max"),
    )

    results: list[dict] = []
    for code, name in LEAGUES.items():
        print(f"== {name} ({code}) ==")
        lg_df = df_all[df_all["League"] == code].copy()
        lg_df = lg_df.sort_values("Date").reset_index(drop=True)
        lg_df = lg_df.dropna(subset=["FTHG", "FTAG", "FTR", "HomeTeam", "AwayTeam"])
        lg_df["Result"] = lg_df["FTR"].map({"H": 0, "D": 1, "A": 2})
        print(f"  matches: {len(lg_df):,}  "
              f"span: {lg_df['Date'].min().date()} → {lg_df['Date'].max().date()}")

        try:
            ftr = add_rolling_features(lg_df)
        except Exception as e:
            print(f"  [SKIP] add_rolling_features failed: {e}")
            continue
        try:
            bt = backtest_models_v2(lg_df, ftr, test_weeks=52)
        except Exception as e:
            print(f"  [SKIP] backtest_models_v2 failed: {e}")
            continue
        if bt.empty:
            print(f"  [SKIP] empty bt")
            continue

        try:
            log, sum_ = pf.ev_backtest_simulate_v2(
                bt, lg_df, df_features=ftr, **cfg,
            )
        except Exception as e:
            print(f"  [SKIP] sim failed: {e}")
            continue
        if "error" in sum_:
            print(f"  [SKIP] sim error: {sum_['error']}")
            continue

        dd = _max_drawdown_pct(log, cfg["initial_bankroll"])
        print(f"  → 52w £{sum_['final']:>9,.0f}  "
              f"ROI {sum_['roi']:>+6.1f}%  "
              f"n={sum_['n_bets']:>3}  "
              f"win {sum_['win_rate']:>4.1f}%  DD {dd:.1f}%\n")
        results.append({
            "league": code, "name": name,
            "matches": int(len(lg_df)),
            "final":   float(sum_["final"]),
            "roi":     float(sum_["roi"]),
            "n_bets":  int(sum_["n_bets"]),
            "win_rate": float(sum_["win_rate"]),
            "dd":      round(dd, 2),
        })

    print("=" * 90)
    print("MOCK TWO CROSS-LEAGUE COMPARISON  (min_team_elo=1500 + mp=0.30)")
    print("=" * 90)
    print(f"{'league':<14}  {'matches':>7}  {'final':>10}  {'ROI':>7}  "
          f"{'n':>4}  {'win%':>5}  {'DD%':>5}  {'verdict':<12}")
    print("-" * 90)
    for r in results:
        verdict = ("strong" if r["final"] > 20_000
                   else "decent" if r["final"] > 12_000
                   else "neutral" if r["final"] > 8_000
                   else "FAILS")
        print(f"{r['name']:<14}  {r['matches']:>7,}  "
              f"£{r['final']:>9,.0f}  {r['roi']:>+6.1f}%  "
              f"{r['n_bets']:>4}  {r['win_rate']:>4.1f}%  {r['dd']:>4.1f}%  "
              f"{verdict:<12}")

    profitable = [r for r in results if r["final"] > 10_000]
    print()
    if len(profitable) >= 4:
        print("✅ STRUCTURAL EDGE — Mock Two transfers to 4+/5 leagues.")
    elif len(profitable) >= 3:
        print("🟡 PARTIAL — Mock Two works in 3/5 leagues.")
    elif len(profitable) >= 2:
        print("🟡 LIMITED — Mock Two transfers to 2 leagues only.")
    else:
        print("⚠ EPL-SPECIFIC — Mock Two's edge is also EPL-fitted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
