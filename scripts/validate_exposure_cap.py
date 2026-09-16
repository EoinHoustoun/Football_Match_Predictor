"""Pending-stake cap: 50% of the OPENING bankroll (live today) vs 50% of the
bankroll at the start of each GAMEWEEK (proposed 2026-09-16).

The standard simulator settles each match day before sizing the next, so it
never holds a weekend of pending bets and cannot see a cap on them at all.
Live does: it places the whole gameweek from one bankroll and caps the stake
riding on unsettled bets. This compares, on the identical honest per-season
pack used by validate_2026_27.py:

  day-settled (the old validation view)   no cap, Saturday compounds into Sunday
  gameweek, no cap                        live sizing, nothing capping pending
  gameweek, 50% of opening bankroll       what runs live today
  gameweek, 50% of gameweek bankroll      the proposal
  gameweek, 30% / 40% of gameweek         neighbours, to see if it is a spike

Headline per the validation rules: WORST season and deepest drawdown, not the
mean. Config is read from data/portfolio.json (Main), never hardcoded.

Usage:
  python3 scripts/validate_exposure_cap.py build   # train + pickle season pack
  python3 scripts/validate_exposure_cap.py run     # run variants on the pack
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import portfolio as pf  # noqa: E402
import validate_2026_27 as v  # noqa: E402

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"


def build() -> None:
    from data import load_data, add_rolling_features
    df = load_data().copy()
    df["SeasonLbl"] = df["Date"].apply(v._season_label)
    df_features = add_rolling_features(df)
    pack = v.build_season_pack(df, df_features)
    PACK.write_bytes(pickle.dumps(pack))
    print(f"Pack written: {PACK} ({len(pack)} seasons)")


def run() -> None:
    pack = pickle.loads(PACK.read_bytes())
    live = v.live_main_config()
    variants = [
        ("day-settled, no cap (old view)", {}),
        ("gameweek, no cap", {"gameweek_mode": True, "max_pending_bets": 5}),
        ("gameweek, 50% of opening (LIVE)", {"gameweek_mode": True, "max_pending_bets": 5,
                                             "exposure_cap_pct": 0.50,
                                             "exposure_cap_basis": "opening"}),
        ("gameweek, 50% of GW bankroll", {"gameweek_mode": True, "max_pending_bets": 5,
                                          "exposure_cap_pct": 0.50,
                                          "exposure_cap_basis": "gameweek"}),
        ("gameweek, 40% of GW bankroll", {"gameweek_mode": True, "max_pending_bets": 5,
                                          "exposure_cap_pct": 0.40,
                                          "exposure_cap_basis": "gameweek"}),
        ("gameweek, 30% of GW bankroll", {"gameweek_mode": True, "max_pending_bets": 5,
                                          "exposure_cap_pct": 0.30,
                                          "exposure_cap_basis": "gameweek"}),
    ]
    seasons = list(pack)
    out = {}
    print(f"{'variant':34s} " + " ".join(f"{s:>10}" for s in seasons)
          + f" {'WORST':>9} {'maxDD':>6} {'bets':>5} {'final £':>10}")
    for name, ov in variants:
        r = v.run_variant(pack, ov, live)
        dds = [s.get("max_dd", 0) for s in r["per_season"].values()]
        # Compounded across seasons, like the live line that rolls its bankroll.
        final = v.INITIAL
        for s in seasons:
            final *= 1 + r["per_season"][s]["profit"] / v.INITIAL
        r["worst_drawdown"] = max(dds) if dds else 0
        r["compounded_final"] = round(final, 2)
        out[name] = r
        print(f"{name:34s} " + " ".join(f"{r['per_season'][s]['profit']:>+10,.0f}" for s in seasons)
              + f" {r['worst_season']:>+9,.0f} {max(dds):>5.1f}% {r['total_bets']:>5} {final:>10,.0f}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"exposure_cap_{ts}.json"
    path.write_text(json.dumps({"config": {k: sorted(x) if isinstance(x, set) else x
                                           for k, x in live.items()},
                                "seasons": seasons, "results": out}, indent=2, default=str))
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
