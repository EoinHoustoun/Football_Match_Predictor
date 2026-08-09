"""Re-validate the deployed 2026-27 Main config before auto-bet goes live.

The season was relaunched with the Main settings marked PROVISIONAL and
auto-bet switched off, pending exactly this. Two things changed underneath the
config since it was last validated, and both need to be answered honestly:

  1. **The no-history gate is new.** `min_team_matches=6` blocks any fixture
     where either side has too little top-flight history. It fires every season,
     not just this one: three teams come up each August and each is blocked for
     its first six matches. So it is not a 2026-27-only setting and it can be
     measured across the whole dataset.
  2. **Draws only.** `auto_markets` is now `["D"]`, where the config was
     validated on `["D", "under25"]`.

What this script does NOT do is search for a better config. The live settings
were chosen by a 14-fold walk-forward and confirmed by random search across 60
variants; re-running a search here would just re-select on the same data. This
asks the narrower and more useful question: **does the deployed config still
hold up once the gate is on and U2.5 is gone, and is it still sitting in a
region that works rather than on a spike?**

Method, per season, for every season with a prior season to train on:
  - `backtest_models` trains on everything before the season and tests on it.
  - The isotonic calibrator is fit strictly BEFORE the season starts. Fitting it
    on the test window is what produced the £100k artifact.
  - Every variant is run on the identical per-season pack, so differences are
    the config and nothing else.

Reported: per-season profit, the WORST season (survival is the constraint, not
the mean), bet counts, max drawdown, and median CLV against the Pinnacle close
as the leading indicator.

Output: stdout + data/diagnostics/validate_2026_27_<ts>.json
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import load_data, add_rolling_features  # noqa: E402
from models import backtest_models                # noqa: E402
import portfolio as pf                            # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)

INITIAL = 10_000.0

_MARKET_CODE = {"Draw": "D", "Home Win": "H", "Away Win": "A",
                "Over 2.5": "over25", "Under 2.5": "under25"}


def _season_label(d: pd.Timestamp) -> str:
    """August onward belongs to the season starting that year."""
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def live_main_config() -> dict:
    """Read the deployed Main settings and map them onto simulator arguments.

    Read, never hardcode: `data/portfolio.json` is the truth for what will
    actually run, and a hardcoded copy here would silently validate something
    else the next time the settings change.
    """
    s = pf.load_portfolio()["settings"]
    return dict(
        min_ev_pct=float(s.get("min_ev", 0.40)) * 100.0,
        min_prob=float(s.get("min_prob", 0.21)),
        kelly_frac=float(s.get("kelly_fraction", 1.0)),
        max_stake_pct=float(s.get("max_stake_pct", 0.25)),
        allowed_markets=set(s.get("auto_markets", ["D"])),
        skip_late_season=bool(s.get("skip_late_season", True)),
        skip_home_title_race=bool(s.get("skip_home_title_race", False)),
        banned_dows=set(s.get("main_banned_dows", []) or []),
        banned_months=set(s.get("main_banned_months", []) or []),
        min_team_elo=s.get("main_min_team_elo"),
        max_team_elo=s.get("main_max_team_elo"),
        elo_gap_min=s.get("main_elo_gap_min"),
        elo_gap_max=s.get("main_elo_gap_max"),
        max_ev_pct=s.get("main_max_ev_pct"),
        market_gates=s.get("market_gates"),
        min_team_matches=s.get("min_team_matches"),
    )


def _max_drawdown_pct(log: pd.DataFrame) -> float:
    if log.empty:
        return 0.0
    series = pd.concat([pd.Series([INITIAL]), log["Bankroll"].astype(float)],
                       ignore_index=True)
    peaks = series.cummax()
    return float(((peaks - series) / peaks).max() * 100)


def _median_clv(log: pd.DataFrame, df: pd.DataFrame) -> float | None:
    """Median CLV of the simulated bets against the Pinnacle close.

    CLV moves before P&L does, so a config that profits while giving up value to
    the close is on borrowed time. Bets whose match has no closing price simply
    drop out rather than counting as zero.
    """
    if log.empty:
        return None
    values = []
    for _, r in log.iterrows():
        try:
            home, away = str(r["Match"]).split(" vs ", 1)
        except ValueError:
            continue
        market = _MARKET_CODE.get(str(r["Market"]))
        if market is None:
            continue
        close = pf.extract_closing_odds(home, away, r["Date"], market, df)
        clv = pf.compute_clv(float(r["Odds"]), close) if close else None
        if clv is not None:
            values.append(clv)
    return round(statistics.median(values) * 100, 2) if values else None


def build_season_pack(df: pd.DataFrame, df_features: pd.DataFrame) -> dict:
    """Per-season backtest frame plus a calibrator fit only on earlier data."""
    pack: dict[str, tuple] = {}
    for season in sorted(df["SeasonLbl"].unique())[1:]:
        rows = df[df["SeasonLbl"] == season]
        start, end = rows["Date"].min(), rows["Date"].max()
        df_slice = df[df["Date"] <= end].copy()
        ftr_slice = df_features[df_features["Date"] <= end].copy()

        bt = backtest_models(df_slice, ftr_slice, test_weeks=40)
        if bt.empty:
            continue
        bt_season = bt[bt["Date"].apply(_season_label) == season].copy()
        if bt_season.empty:
            continue

        # Honest calibration: strictly before the test window opens.
        pre = df[df["Date"] < start].copy()
        pre_ftr = df_features[df_features["Date"] < start].copy()
        cal: dict = {}
        if len(pre) >= 200:
            try:
                cal = pf.fit_calibrators_from_backtest(
                    backtest_models(pre, pre_ftr, test_weeks=10))
            except Exception:
                cal = {}

        pack[season] = (df_slice, ftr_slice, bt_season, cal)
        print(f"  {season}: {len(bt_season):>4} matches  "
              f"calibrated markets: {sorted(cal) if cal else 'none'}")
    return pack


def _top_club(log: pd.DataFrame) -> tuple[str | None, int]:
    """The club appearing in the most bets, and how many. Concentration is the
    thing the headline profit hides."""
    if log.empty:
        return None, 0
    counts: dict[str, int] = {}
    for match in log["Match"]:
        for club in str(match).split(" vs "):
            counts[club] = counts.get(club, 0) + 1
    club = max(counts, key=counts.get)
    return club, counts[club]


def run_variant(pack: dict, overrides: dict, base: dict) -> dict:
    """One config across every season. Worst season is the headline."""
    cfg = {**base, **overrides}
    per_season: dict[str, dict] = {}
    for season, (df_slice, ftr_slice, bt_season, cal) in pack.items():
        log, summary = pf.ev_backtest_simulate(
            bt_season, df_slice, df_features=ftr_slice,
            initial_bankroll=INITIAL,
            enable_simultaneous_correction=True,
            odds_source="Max", detect_source="PS",
            calibrators=cal, **cfg,
        )
        if "error" in summary:
            per_season[season] = {"error": summary["error"], "profit": 0.0,
                                  "n_bets": 0}
            continue
        top_club, top_n = _top_club(log)
        per_season[season] = {
            "profit":   summary["profit"],
            "top_club": top_club,
            "top_club_share": (round(100 * top_n / len(log), 0)
                               if len(log) else None),
            "roi":      summary["roi"],
            "n_bets":   summary["n_bets"],
            "win_rate": summary["win_rate"],
            "max_dd":   round(_max_drawdown_pct(log), 1),
            "blocked_no_history": summary.get("skipped_no_history", 0),
            "median_clv": _median_clv(log, df_slice),
        }

    profits = [v["profit"] for v in per_season.values()]
    clvs = [v["median_clv"] for v in per_season.values()
            if v.get("median_clv") is not None]
    return {
        "per_season":   per_season,
        "worst_season": min(profits) if profits else 0.0,
        "median_season": statistics.median(profits) if profits else 0.0,
        "total_profit": sum(profits),
        "total_bets":   sum(v["n_bets"] for v in per_season.values()),
        "losing_seasons": sum(1 for p in profits if p < 0),
        "median_clv":   round(statistics.median(clvs), 2) if clvs else None,
    }


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now():%H:%M:%S}] Loading data…")
    df = load_data().copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    df_features = add_rolling_features(df)

    live = live_main_config()
    label = pf.load_portfolio()["settings"].get("main_settings_label", "")
    print(f"\nDeployed config: {label}")
    for k in ("min_ev_pct", "min_prob", "kelly_frac", "max_stake_pct",
              "allowed_markets", "min_team_elo", "banned_dows",
              "min_team_matches"):
        print(f"  {k:18s} {live[k]}")

    print(f"\n[{datetime.now():%H:%M:%S}] Building per-season packs "
          f"(honest calibrators)…")
    pack = build_season_pack(df, df_features)
    if not pack:
        print("No testable seasons.")
        return 1

    # The gate is the thing under test, so it is a variant axis, not a constant.
    gate = live["min_team_matches"]
    variants = [
        ("deployed (gate on)",       {}),
        ("deployed, gate OFF",       {"min_team_matches": None}),
        # Robustness: is the config in a region that works, or on a spike?
        ("EV gate 30%",              {"min_ev_pct": 30.0}),
        ("EV gate 50%",              {"min_ev_pct": 50.0}),
        ("min_prob 0.18",            {"min_prob": 0.18}),
        ("min_prob 0.25",            {"min_prob": 0.25}),
        ("Elo floor 1450",           {"min_team_elo": 1450}),
        ("Elo floor 1550",           {"min_team_elo": 1550}),
        ("half Kelly",               {"kelly_frac": 0.5}),
        # Does a stricter gate buy anything? Tightening-only, so it is fair game.
        ("gate 10 matches",          {"min_team_matches": 10}),
        # Club-exposure cap. The value is chosen a priori — no club should carry
        # more than roughly a third of a season, and this config places 6 to 14
        # bets a season — NOT by taking whichever number backtests best. The
        # neighbours are printed to show the cost of the control, not to pick a
        # winner from them.
        ("club cap 5 (a priori)",    {"max_bets_per_club": 5}),
        ("club cap 3",               {"max_bets_per_club": 3}),
        ("club cap 7",               {"max_bets_per_club": 7}),
        ("club cap 5 + half Kelly",  {"max_bets_per_club": 5, "kelly_frac": 0.5}),
    ]

    print(f"\n[{datetime.now():%H:%M:%S}] Running {len(variants)} variants × "
          f"{len(pack)} seasons…\n")
    seasons = list(pack)
    header = "  ".join(f"{s:>10}" for s in seasons)
    print(f"{'variant':24s}  {header}  {'WORST':>10} {'bets':>5} "
          f"{'CLV':>7} {'1club':>5}")
    print("-" * (26 + len(header) + 26))

    results: dict[str, dict] = {}
    for name, overrides in variants:
        r = run_variant(pack, overrides, live)
        results[name] = r
        cells = "  ".join(
            f"{r['per_season'][s]['profit']:>+10,.0f}" for s in seasons)
        clv = f"{r['median_clv']:>+6.2f}%" if r["median_clv"] is not None else "     n/a"
        shares = [v["top_club_share"] for v in r["per_season"].values()
                  if v.get("top_club_share") is not None]
        conc = f"{max(shares):>3.0f}%" if shares else " n/a"
        print(f"{name:24s}  {cells}  {r['worst_season']:>+10,.0f} "
              f"{r['total_bets']:>5} {clv} {conc}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    on, off = results["deployed (gate on)"], results["deployed, gate OFF"]
    blocked = sum(v.get("blocked_no_history", 0)
                  for v in on["per_season"].values())
    print(f"\nGate effect across {len(seasons)} seasons "
          f"({blocked} candidate bets blocked):")
    print(f"  worst season   {off['worst_season']:>+12,.0f}  →  "
          f"{on['worst_season']:>+12,.0f}")
    print(f"  median season  {off['median_season']:>+12,.0f}  →  "
          f"{on['median_season']:>+12,.0f}")
    print(f"  losing seasons {off['losing_seasons']:>12}  →  "
          f"{on['losing_seasons']:>12}")

    neighbours = [n for n, _ in variants
                  if n not in ("deployed (gate on)", "deployed, gate OFF")]
    healthy = sum(1 for n in neighbours if results[n]["worst_season"] >= 0)
    print(f"\nRobustness: {healthy}/{len(neighbours)} neighbouring configs also "
          f"avoid a losing season.")
    print("A config that only works on its own settings is a spike, not an edge.")

    path = OUT_DIR / f"validate_2026_27_{ts}.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deployed_label": label,
        "deployed_config": {k: sorted(v) if isinstance(v, set) else v
                            for k, v in live.items()},
        "gate_setting": gate,
        "seasons": seasons,
        "results": results,
    }, indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
