"""Put both portfolios back on the configuration that ran the 2025-26 season.

Eoin's call, 2026-08-09. The 2026-27 relaunch had layered several filters on top
of the config that actually produced +£8,251.75, and the effect was a line that
would barely trade: at a 40% EV gate the backtest places about nine bets a
season, against the twenty-nine that were really placed last year.

Read from `data/portfolio.backup.20260609_234308.preNextSeasonDefaults.json`,
the snapshot taken immediately before the 2026-27 defaults were applied, so this
restores what was measured rather than what anyone remembers.

What comes back:
  - EV gate 40% -> 23.6%, the level that was live all last season
  - Kelly 0.5 -> 1.0
  - Elo floor 1500 -> off (it was added in June 2026, it never ran last season)
  - club-exposure cap -> off
  - no-history gate -> off, so promoted sides can be staked
  - markets D -> D + Under 2.5

What deliberately stays:
  - **The promoted-side prior.** This is pricing, not a gate. Without it an
    unrated team reads as league average and the Arsenal v Coventry draw prices
    at 31.7% against a market 10.9%, which is what sized a £4,562 stake. With it
    that fixture prices at 17.9%. Removing the gate is Eoin's decision; removing
    the prior would just be a bug.
  - Monday ban, late-season skip, isotonic calibration, simultaneous-bet
    correction, the 0.45 draw cap, per-market Under 2.5 gates. All of these were
    live last season too.

Mock Two gets the same treatment on its own keys, so the A/B compares the two
models rather than two risk policies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE = DATA / "portfolio.backup.20260609_234308.preNextSeasonDefaults.json"

LABEL = ("2026-27 — restored to the 2025-26 live config (EV 23.6%, full Kelly, "
         "no Elo floor, no club cap, promoted sides bettable, D + U2.5). "
         "Promoted-side prior retained for pricing.")


def restore(path: Path, elo_key: str, label_key: str, source_key: str,
            reference: dict) -> dict:
    portfolio = json.loads(path.read_text())
    settings = portfolio["settings"]

    settings["min_ev"] = reference["min_ev"]                       # 0.236
    settings["auto_bet_threshold"] = reference["auto_bet_threshold"]
    settings["kelly_fraction"] = reference["kelly_fraction"]       # 1.0
    settings["min_prob"] = reference["min_prob"]
    settings["auto_markets"] = list(reference["auto_markets"])     # D + under25
    settings["market_gates"] = reference["market_gates"]

    # Filters that did not exist last season.
    settings[elo_key] = None
    settings["max_bets_per_club"] = None
    settings["min_team_matches"] = None

    settings[label_key] = LABEL
    settings[source_key] = "restore_2025_26_20260809"
    settings["restored_on"] = "2026-08-09"

    # Never touched here: bets, bankroll, and the auto-bet switch itself.
    for key in ("bets", "bankroll", "initial_bankroll"):
        assert key in portfolio, key
    path.write_text(json.dumps(portfolio, indent=2))
    return settings


def main() -> int:
    reference = json.loads(REFERENCE.read_text())["settings"]

    main_s = restore(DATA / "portfolio.json", "main_min_team_elo",
                     "main_settings_label", "main_settings_source", reference)
    mt_s = restore(DATA / "portfolio_two.json", "v2_min_team_elo",
                   "v2_settings_label", "v2_settings_source", reference)

    shown = ("min_ev", "auto_bet_threshold", "kelly_fraction", "min_prob",
             "auto_markets", "max_bets_per_club", "min_team_matches",
             "skip_late_season", "auto_bet_enabled")
    for name, s, elo_key in (("MAIN", main_s, "main_min_team_elo"),
                             ("MOCK TWO", mt_s, "v2_min_team_elo")):
        print(f"\n{name}")
        for k in shown:
            print(f"  {k:20s} {s.get(k)}")
        print(f"  {'elo floor':20s} {s.get(elo_key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
