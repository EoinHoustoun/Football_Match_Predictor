"""How a binding pending-stake cap should be shared out, and whether the cap
swallows the Kelly A/B.

GW5 2026-27 (18-20 Sep) put the full 50% cap on both lines and lost 6 of 7 on
Main, 7 of 7 on Mock Two. Two things were visible in the real bet history:

  1. Live truncates: bets are placed best-EV first and whoever arrives last
     gets the headroom (Mock Two's Brentford v Chelsea got £390 of a £2k ask;
     Main's first-placed Man City v Sunderland at 6.4 got £7,298).
  2. With 6-7 qualifying bets the cap, not the Kelly fraction, sets the total
     exposure, so Main (1.0 Kelly) and Mock Two (0.45 Kelly) may be staking
     near-identically and the A/B measures nothing.

This runs the gameweek simulator on the same honest per-season pack as
validate_exposure_cap.py, live Main config (read from data/portfolio.json),
no pending-count limit (removed 16 Sep), and compares:

  truncate vs pro-rata allocation at the live 50%-of-gameweek cap
  each at 1.0 Kelly (Main) and 0.45 Kelly (Mock Two's live fraction)
  neighbours at 40% and 30% so a winner is a region, not a spike

Headline per the validation rules: WORST season and deepest drawdown, not the
mean, plus how many gameweeks the cap actually bound.

Usage:
  python3 scripts/validate_exposure_cap.py build   # once, if the pack is missing
  python3 scripts/validate_cap_allocation.py
"""
from __future__ import annotations

import json
import pickle
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import portfolio as pf  # noqa: E402
import validate_2026_27 as v  # noqa: E402

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"


def run_variant(pack: dict, overrides: dict, base: dict) -> dict:
    """validate_2026_27.run_variant, plus the cap-binding count the summary
    now carries and the compounded multiple across seasons."""
    cfg = {**base, **overrides}
    per_season: dict[str, dict] = {}
    for season, (df_slice, ftr_slice, bt_season, cal) in pack.items():
        log, summary = pf.ev_backtest_simulate(
            bt_season, df_slice, df_features=ftr_slice,
            initial_bankroll=v.INITIAL,
            enable_simultaneous_correction=True,
            odds_source="Max", detect_source="PS",
            calibrators=cal, **cfg,
        )
        if "error" in summary:
            print(f"  {season}: {summary['error']}")
            per_season[season] = {"error": summary["error"], "profit": 0.0,
                                  "n_bets": 0, "max_dd": 0.0, "capped_blocks": 0,
                                  "skipped_cap": 0}
            continue
        per_season[season] = {
            "profit":        summary["profit"],
            "roi":           summary["roi"],
            "n_bets":        summary["n_bets"],
            "win_rate":      summary["win_rate"],
            "max_dd":        round(v._max_drawdown_pct(log), 1),
            "median_clv":    v._median_clv(log, df_slice),
            "capped_blocks": summary.get("capped_blocks", 0),
            "skipped_cap":   summary.get("skipped_exposure_cap", 0),
        }
    profits = [s["profit"] for s in per_season.values()]
    final = v.INITIAL
    for s in per_season.values():
        final *= 1 + s["profit"] / v.INITIAL
    return {
        "per_season":       per_season,
        "worst_season":     min(profits) if profits else 0.0,
        "median_season":    statistics.median(profits) if profits else 0.0,
        "total_profit":     sum(profits),
        "total_bets":       sum(s["n_bets"] for s in per_season.values()),
        "losing_seasons":   sum(1 for p in profits if p < 0),
        "worst_drawdown":   max(s["max_dd"] for s in per_season.values()),
        "capped_blocks":    sum(s["capped_blocks"] for s in per_season.values()),
        "skipped_cap":      sum(s["skipped_cap"] for s in per_season.values()),
        "compounded_final": round(final, 2),
    }


def main() -> int:
    if not PACK.exists():
        print("Pack missing: run  python3 scripts/validate_exposure_cap.py build  first")
        return 1
    pack = pickle.loads(PACK.read_bytes())
    live = v.live_main_config()
    live["max_pending_bets"] = None   # the five-bet limit was removed 16 Sep 2026

    gw = {"gameweek_mode": True, "exposure_cap_basis": "gameweek"}
    variants = []
    for kelly in (1.0, 0.45):
        k = {"kelly_frac": kelly}
        tag = f"k={kelly:.2f}"
        variants.append((f"{tag} no cap", {**gw, **k}))
        for pct in (0.50, 0.40, 0.30):
            for mode in ("truncate", "prorata"):
                variants.append((f"{tag} {int(pct*100)}% {mode}",
                                 {**gw, **k, "exposure_cap_pct": pct,
                                  "exposure_cap_mode": mode}))

    seasons = list(pack)
    out = {}
    hdr = (f"{'variant':22s} " + " ".join(f"{s:>9}" for s in seasons)
           + f" {'WORST':>8} {'maxDD':>6} {'bets':>5} {'capGW':>5} {'final £':>9}")
    print(hdr)
    for name, ov in variants:
        r = run_variant(pack, ov, live)
        out[name] = r
        print(f"{name:22s} "
              + " ".join(f"{r['per_season'][s]['profit']:>+9,.0f}" for s in seasons)
              + f" {r['worst_season']:>+8,.0f} {r['worst_drawdown']:>5.1f}%"
              + f" {r['total_bets']:>5} {r['capped_blocks']:>5} {r['compounded_final']:>9,.0f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"cap_allocation_{ts}.json"
    path.write_text(json.dumps({"config": {k: sorted(x) if isinstance(x, set) else x
                                           for k, x in live.items()},
                                "seasons": seasons, "results": out},
                               indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
