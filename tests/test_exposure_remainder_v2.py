"""Mock Two must not turn a sub-penny exposure remainder into a £0.00 bet."""
from __future__ import annotations

from portfolio import auto_place_value_bets_v2


def test_auto_bet_v2_skips_sub_penny_exposure_remainder():
    pending = [
        ("Chelsea", "Hull", 3965.67),
        ("Sunderland", "Arsenal", 4265.28),
        ("Liverpool", "Fulham", 894.92),
    ]
    port = {
        "initial_bankroll": 18_251.75, "bankroll": 9_125.88,
        "bets": [{"home": h, "away": a, "market": "D", "status": "pending",
                  "stake": s} for h, a, s in pending],
        "settings": {
            "kelly_fraction": 1.0, "max_stake_pct": 0.25,
            "auto_markets": ["D"], "min_prob": 0.17,
            "use_calibrated_probs": False, "use_uncertainty_kelly": False,
        },
    }
    cand = [{
        "home": "Coventry", "away": "Brighton", "date": "2026-09-13",
        "market": "D", "selection": "Draw",
        "model_prob": 0.3597, "odds": 3.9, "ev": 0.3597 * 3.9 - 1,
    }]
    placed = auto_place_value_bets_v2(port, cand, threshold=0.23)
    assert placed == []
    assert all(b["stake"] > 0 for b in port["bets"])
