"""Per-line exposure cap share and pro-rata allocation (11 Oct 2026).

The live cap was hardcoded at 50% of cash + pending and filled first-come, so
the last bet placed in a run got whatever room was left. Eoin approved 35% with
pro-rata allocation on both lines; the share and the mode are now settings and
default to the old behaviour (0.50, truncate) when absent.
"""
from __future__ import annotations

import pytest

import portfolio
from portfolio import (_prorata_factor, auto_place_value_bets,
                       auto_place_value_bets_v2)

PLACERS = [auto_place_value_bets, auto_place_value_bets_v2]


def _draws(n: int, prob: float = 0.40, odds: float = 4.0):
    return [{"home": f"H{i}", "away": f"A{i}", "date": "2026-10-17",
             "market": "D", "selection": "Draw",
             "model_prob": prob, "odds": odds, "ev": prob * odds - 1}
            for i in range(n)]


def _port(cash: float = 10_000.0, pending: float = 0.0, **settings):
    bets = ([{"home": "X", "away": "Y", "market": "D", "status": "pending",
              "stake": pending}] if pending else [])
    return {"initial_bankroll": cash, "bankroll": cash, "bets": bets,
            "settings": {"kelly_fraction": 1.0, "max_stake_pct": 0.25,
                         "auto_markets": ["D"], "min_prob": 0.21,
                         "use_calibrated_probs": False,
                         "use_uncertainty_kelly": False,
                         "use_simultaneous_kelly": False, **settings}}


def _new_stakes(port, placed):
    return [b["stake"] for b in placed]


def test_prorata_factor_is_one_when_everything_fits():
    assert _prorata_factor(10_000, [0.1, 0.1], 5_000) == 1.0


def test_prorata_factor_fills_the_headroom_exactly():
    # Sequential sizing off falling cash: 2,000 then 1,600 = 3,600 wanted.
    f = _prorata_factor(10_000, [0.2, 0.2], 1_800)
    first = 10_000 * 0.2 * f
    second = (10_000 - first) * 0.2 * f
    assert 0 < f < 1
    assert first + second == pytest.approx(1_800, abs=0.01)


def test_prorata_factor_ignores_zero_kelly_and_no_headroom():
    assert _prorata_factor(10_000, [0.0, 0.0], 100) == 1.0
    assert _prorata_factor(10_000, [0.2], 0.0) == 1.0


@pytest.mark.parametrize("place", PLACERS)
def test_defaults_keep_the_old_50pct_truncate_cap(place):
    port = _port()
    place(port, _draws(3), threshold=0.10)
    pending = sum(b["stake"] for b in port["bets"] if b["status"] == "pending")
    assert pending <= 0.50 * 10_000 + 0.01


@pytest.mark.parametrize("place", PLACERS)
def test_35pct_cap_binds_on_cash_plus_pending(place):
    port = _port(cash=10_000.0, pending=0.0, max_exposure_pct=0.35)
    place(port, _draws(3), threshold=0.10)
    pending = sum(b["stake"] for b in port["bets"] if b["status"] == "pending")
    assert pending <= 0.35 * 10_000 + 0.01
    assert pending > 0.30 * 10_000          # the cap, not Kelly, set the total


@pytest.mark.parametrize("place", PLACERS)
def test_prorata_shares_the_room_instead_of_starving_the_last_bet(place):
    trunc = _port(max_exposure_pct=0.35)
    pro = _port(max_exposure_pct=0.35, exposure_cap_mode="prorata")
    t = _new_stakes(trunc, place(trunc, _draws(3), threshold=0.10))
    p = _new_stakes(pro, place(pro, _draws(3), threshold=0.10))
    # Truncate fills the cap with the first two and starves the third;
    # pro-rata places all three at the same share of their Kelly asks (equal
    # asks off falling cash give a geometric run), and the total fits the cap.
    assert len(t) == 2
    assert len(p) == 3
    assert p[1] / p[0] == pytest.approx(p[2] / p[1], rel=1e-3)
    assert sum(p) == pytest.approx(0.35 * 10_000, abs=0.05)


@pytest.mark.parametrize("place", PLACERS)
def test_prorata_respects_room_already_used_by_earlier_runs(place):
    port = _port(cash=6_500.0, pending=3_000.0, max_exposure_pct=0.35,
                 exposure_cap_mode="prorata")
    placed = place(port, _draws(3), threshold=0.10)
    pending = sum(b["stake"] for b in port["bets"] if b["status"] == "pending")
    assert pending <= 0.35 * 9_500 + 0.05
    assert all(b["stake"] > 0 for b in placed)


@pytest.mark.parametrize("place", PLACERS)
def test_prorata_changes_nothing_when_the_cap_does_not_bind(place):
    small = _draws(1, prob=0.30, odds=3.6)
    a = _port(max_exposure_pct=0.35)
    b = _port(max_exposure_pct=0.35, exposure_cap_mode="prorata")
    sa = _new_stakes(a, place(a, small, threshold=0.0))
    sb = _new_stakes(b, place(b, small, threshold=0.0))
    assert sa == sb


def test_default_constant_is_the_pre_change_value():
    assert portfolio.DEFAULT_MAX_EXPOSURE_PCT == 0.50
