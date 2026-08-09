"""Mock Two must detect at the sharp price or not bet at all.

The walk-forward winner for Mock Two was **F_PS_Max**: measure EV against the
Pinnacle price, place at the best price on the panel. Median CLV +2.68%, 77.7%
of bets beating the close across 14 folds. That result depends entirely on the
EV gate being measured against Pinnacle.

The live path did not enforce it. When The Odds API returned no Pinnacle quote
(`regions=uk,eu` does not guarantee one for every fixture) `detect_odds` fell
back to the placement odds. Best-of-panel is by construction the longest price
available, so EV computed against it is inflated — the gate is measured against
the very price the edge is supposed to be found *against*. That does not
reproduce F_PS_Max, it reproduces F_Max_Max, which was not the validated
workflow.

So: when a portfolio declares a separate detect source, a candidate without a
sharp price is skipped, not repriced. Fewer bets is the correct outcome.
"""
from __future__ import annotations

import portfolio as pf


def _portfolio(**settings) -> dict:
    p = {
        "season": "2026-27",
        "initial_bankroll": 10_000.0,
        "bankroll": 10_000.0,
        "bets": [],
        "settings": {
            "auto_bet_enabled": True,
            "auto_bet_threshold": 0.40,
            "auto_markets": ["D"],
            "min_prob": 0.21,
            "min_ev": 0.40,
            "kelly_fraction": 1.0,
            "max_stake_pct": 0.33,
            "use_calibrated_probs": False,
            "use_uncertainty_kelly": False,
            "use_simultaneous_kelly": False,
            "skip_late_season": False,
            "min_team_matches": None,
            "max_bets_per_club": None,
            "detect_source": "PS",
        },
    }
    p["settings"].update(settings)
    return p


def _candidate(detect_odds=None) -> dict:
    """Best price 9.20 against a 30% model probability: +176% EV at the placement
    price, and still +110% at a sharp 7.00."""
    c = {
        "home": "Everton", "away": "Fulham", "date": "2026-09-12",
        "market": "D", "selection": "Draw",
        "odds": 9.20, "model_prob": 0.30, "ev": 1.76,
        "home_elo": 1600, "away_elo": 1600,
    }
    if detect_odds is not None:
        c["detect_odds"] = detect_odds
    return c


def test_a_sharp_price_is_used_for_the_ev_gate():
    p = _portfolio()

    placed = pf.auto_place_value_bets_v2(p, [_candidate(detect_odds=7.00)], 0.40)

    assert len(placed) == 1
    assert placed[0]["odds"] == 9.20            # placed at the best price
    assert placed[0]["detect_odds"] == 7.00     # detected at the sharp price


def test_a_missing_sharp_price_skips_rather_than_falling_back():
    p = _portfolio()

    placed = pf.auto_place_value_bets_v2(p, [_candidate(detect_odds=None)], 0.40)

    assert placed == []


def test_the_skip_is_recorded():
    p = _portfolio()
    skip_log: list[dict] = []

    pf.auto_place_value_bets_v2(p, [_candidate()], 0.40, skip_log=skip_log)

    assert any(e.get("reason") == "no_sharp_price" for e in skip_log)


def test_an_unusable_sharp_price_also_skips():
    """A bookmaker returning 1.0 or junk is absence, not a price."""
    p = _portfolio()

    assert pf.auto_place_value_bets_v2(p, [_candidate(detect_odds=1.0)], 0.40) == []


def test_no_detect_source_keeps_the_old_fallback():
    """Main declares no separate detect source and must be unaffected."""
    p = _portfolio()
    del p["settings"]["detect_source"]

    placed = pf.auto_place_value_bets_v2(p, [_candidate(detect_odds=None)], 0.40)

    assert len(placed) == 1


def test_detect_source_matching_the_placement_source_keeps_the_fallback():
    """Detecting and placing at the same book is single-source mode, where
    there is no sharp price to be missing."""
    p = _portfolio(detect_source="Max", odds_source="Max")

    placed = pf.auto_place_value_bets_v2(p, [_candidate(detect_odds=None)], 0.40)

    assert len(placed) == 1
