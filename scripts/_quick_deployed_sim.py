"""Quick sim of the *currently deployed* Mock Two config across:
  (a) the standard 52-week window (the headline number)
  (b) each prior season (the honest robustness number)

Reads settings live from data/portfolio_two.json so it always reflects
whatever Phase 4 / multi-season pruning we last applied.
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
from portfolio import compute_per_bin_variance     # noqa: E402


def _max_drawdown_pct(log: pd.DataFrame, init: float) -> float:
    if log.empty: return 0.0
    s = pd.concat([pd.Series([init]), log["Bankroll"].astype(float)],
                  ignore_index=True)
    return float(((s.cummax() - s) / s.cummax()).max() * 100)


def _season_label(d: pd.Timestamp) -> str:
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def _config_from_settings(s: dict) -> dict:
    """Translate live portfolio settings → ev_backtest_simulate_v2 kwargs."""
    return dict(
        min_ev_pct=float(s.get("min_ev", 0.40)) * 100,
        base_kelly_frac=float(s.get("kelly_fraction", 1.0)),
        max_stake_pct=float(s.get("max_stake_pct", 0.33)),
        initial_bankroll=10_000.0,
        allowed_markets=set(s.get("auto_markets", ["D", "under25"])),
        min_prob=float(s.get("min_prob", 0.32)),
        enable_simultaneous_correction=bool(s.get("use_simultaneous_kelly", True)),
        skip_late_season=bool(s.get("skip_late_season", True)),
        skip_home_title_race=bool(s.get("skip_home_title_race", False)),
        odds_source="Max",
        detect_source="PS",
        market_gates=s.get("market_gates"),
        # Phase 4 validated filters
        banned_dows=set(s.get("v2_banned_dows", [])) or None,
        banned_months=set(s.get("v2_banned_months", [])) or None,
        max_ev_pct=float(s["v2_max_ev_pct"]) if s.get("v2_max_ev_pct") else None,
    )


def main() -> int:
    settings = json.load(open(ROOT / "data" / "portfolio_two.json"))["settings"]
    cfg = _config_from_settings(settings)

    print(f"DEPLOYED CONFIG ({settings.get('v2_settings_label', '—')}):")
    for k in ("min_prob", "v2_banned_dows", "v2_banned_months", "v2_max_ev_pct",
              "skip_late_season", "kelly_fraction", "max_stake_pct"):
        print(f"  {k}: {settings.get(k)}")
    print()

    df = load_data()
    df_features = add_rolling_features(df)

    # ── 52-week headline ─────────────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] 52-week window…")
    bt52 = backtest_models(df, df_features, test_weeks=52)
    bv52 = {m: compute_per_bin_variance(bt52, m) for m in ("H", "D", "A")}
    log52, sum52 = pf.ev_backtest_simulate_v2(bt52, df, bin_variances=bv52, **cfg)
    if "error" not in sum52:
        dd52 = _max_drawdown_pct(log52, 10_000.0)
        print(f"  52-week:  £{sum52['final']:>10,.0f}  "
              f"ROI {sum52['roi']:>+6.1f}%  n={sum52['n_bets']:>3}  "
              f"win {sum52['win_rate']:>4.1f}%  DD {dd52:>5.1f}%")
    else:
        print(f"  52-week ERROR: {sum52['error']}")

    # ── Per-season ───────────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Per-season validation…")
    df = df.copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    seasons = sorted(df["SeasonLbl"].unique())[1:]  # skip first (no prior)

    season_finals: list[tuple[str, float, float, int]] = []
    for season in seasons:
        season_end = df[df["SeasonLbl"] == season]["Date"].max()
        df_slice  = df[df["Date"] <= season_end].copy()
        ftr_slice = df_features[df_features["Date"] <= season_end].copy()
        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty:
            continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty:
            continue
        bv = {m: compute_per_bin_variance(bt_season, m) for m in ("H", "D", "A")}
        log_, sum_ = pf.ev_backtest_simulate_v2(bt_season, df_slice,
                                                 bin_variances=bv, **cfg)
        if "error" in sum_:
            print(f"  {season}: ERROR {sum_['error']}")
            continue
        dd = _max_drawdown_pct(log_, 10_000.0)
        season_finals.append((season, sum_["final"], dd, sum_["n_bets"]))
        print(f"  {season}:  £{sum_['final']:>10,.0f}  "
              f"ROI {sum_['roi']:>+6.1f}%  n={sum_['n_bets']:>3}  "
              f"win {sum_['win_rate']:>4.1f}%  DD {dd:>5.1f}%")

    if season_finals:
        finals = [f for _, f, _, _ in season_finals]
        mean = sum(finals) / len(finals)
        median = sorted(finals)[len(finals) // 2]
        worst = min(finals); best = max(finals)
        print(f"\n  Mean across {len(season_finals)} seasons:    £{mean:>10,.0f}")
        print(f"  Median:                         £{median:>10,.0f}")
        print(f"  Worst season:                   £{worst:>10,.0f}")
        print(f"  Best season:                    £{best:>10,.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
