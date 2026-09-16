"""Live pending-stake cap follows the current bankroll, not the opening one.

Before 16 Sep 2026 both auto-bet paths capped pending stake at 50% of the
season's opening bankroll. After Main grew from £18,252 to £53,344 that let
only 17% of the bankroll ride, and Spurs v Villa was silently squeezed out.
The cap is now 50% of the bankroll before pending stakes come off (cash plus
what is already staked), which stays fixed through a gameweek because nothing
settles until the matches are played. Backtested in
scripts/validate_exposure_cap.py: no worse in any losing season.
"""
from __future__ import annotations

import pytest

from portfolio import auto_place_value_bets, auto_place_value_bets_v2

_SPURS_VILLA = [{
    "home": "Tottenham", "away": "Aston Villa", "date": "2026-09-19",
    "market": "D", "selection": "Draw",
    "model_prob": 0.36, "odds": 3.76, "ev": 0.36 * 3.76 - 1,
}]


def _main_16_sep(extra_settings=None):
    """Main on 16 Sep: £44,218.61 cash, £9,125.88 pending, opened at £18,251.75."""
    return {
        "initial_bankroll": 18_251.75, "bankroll": 44_218.61,
        "bets": [
            {"home": "Man City", "away": "Sunderland", "market": "D",
             "status": "pending", "stake": 7298.49},
            {"home": "Brighton", "away": "Arsenal", "market": "D",
             "status": "pending", "stake": 1827.39},
        ],
        "settings": {"kelly_fraction": 1.0, "max_stake_pct": 0.25,
                     "auto_markets": ["D"], "min_prob": 0.21,
                     "use_calibrated_probs": False,
                     "use_uncertainty_kelly": False,
                     **(extra_settings or {})},
    }


@pytest.mark.parametrize("place", [auto_place_value_bets, auto_place_value_bets_v2])
def test_cap_room_comes_from_the_current_bankroll(place):
    port = _main_16_sep()
    placed = place(port, _SPURS_VILLA, threshold=0.23)
    assert len(placed) == 1
    pending = sum(b["stake"] for b in port["bets"] if b["status"] == "pending")
    # 50% of (44,218.61 + 9,125.88) = 26,672.25; the old cap was 9,125.88.
    assert 9_125.88 < pending <= 26_672.25


@pytest.mark.parametrize("place", [auto_place_value_bets, auto_place_value_bets_v2])
def test_cap_still_binds_at_half_the_bankroll(place):
    port = _main_16_sep()
    port["bankroll"] = 9_125.88          # cash now equals what is pending
    placed = place(port, _SPURS_VILLA, threshold=0.23)
    assert placed == []                  # (9,125.88 + 9,125.88) / 2 is already staked


def _four_weekend_draws():
    return [{"home": f"H{i}", "away": f"A{i}", "date": "2026-09-19",
             "market": "D", "selection": "Draw",
             "model_prob": 0.36, "odds": 4.0, "ev": 0.36 * 4.0 - 1}
            for i in range(4)]


@pytest.mark.parametrize("place", [auto_place_value_bets, auto_place_value_bets_v2])
def test_a_count_limit_binds_on_the_running_count(place, monkeypatch):
    """When a count limit is set it binds mid-run: a run that began on three
    pending bets used to add three more and finish on six (Mock Two, 16 Sep
    2026). The live limit is off; this pins the mechanism for anyone who sets
    it again."""
    import portfolio
    monkeypatch.setattr(portfolio, "MAX_PENDING_SINGLES", 5)
    port = _main_16_sep()
    port["bets"].append({"home": "Everton", "away": "Ipswich", "market": "D",
                         "status": "pending", "stake": 100.0})
    placed = place(port, _four_weekend_draws(), threshold=0.23)
    pending = [b for b in port["bets"]
               if b["status"] == "pending" and b.get("type") != "acca"]
    assert len(pending) == 5
    assert len(placed) == 2


@pytest.mark.parametrize("place", [auto_place_value_bets, auto_place_value_bets_v2])
def test_no_count_limit_live_only_money_caps_apply(place):
    """Live has no pending-bet count limit: six pending singles do not stop a
    seventh when the stake cap still has room."""
    port = _main_16_sep()
    for i in range(4):
        port["bets"].append({"home": f"X{i}", "away": f"Y{i}", "market": "D",
                             "status": "pending", "stake": 10.0})
    placed = place(port, _four_weekend_draws(), threshold=0.23)
    assert len(placed) >= 1
