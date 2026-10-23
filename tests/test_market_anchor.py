"""Mock Two market anchor · the properties that make it safe to trust.

The anchor may reorder clubs toward the market; it may NOT move the league's
level, exceed the per-club cap, touch clubs the odds table does not know, or
do anything at all at beta 0.
"""
import math
from statistics import mean

import pytest

from research_market_anchor import (MAX_SHIFT_DC, MAX_SHIFT_ELO,
                                    anchored_dc_ratings, anchored_entry_elo,
                                    market_strength_z)

# A tiny six-club league with a market that disagrees about two clubs:
# the odds say OverRated is weaker than carried Elo thinks, Upstart stronger.
ODDS = {
    "Giant": "500/1", "Solid": "20/1", "Mid": "8/1",
    "Upstart": "6/1", "OverRated": "5/2", "Doomed": "1/3",
}
CARRIED = {
    "Giant": 1650.0, "Solid": 1560.0, "Mid": 1500.0,
    "OverRated": 1520.0, "Upstart": 1400.0, "Doomed": 1370.0,
}


def test_strength_z_is_standardised_and_ordered():
    z = market_strength_z(ODDS)
    assert abs(mean(z.values())) < 1e-9
    assert z["Giant"] > z["Solid"] > z["Mid"] > z["Upstart"] > z["OverRated"] > z["Doomed"]


def test_beta_zero_is_identity():
    assert anchored_entry_elo(CARRIED, beta=0.0, odds=ODDS) == CARRIED


def test_anchor_moves_toward_market_and_preserves_league_mean():
    out = anchored_entry_elo(CARRIED, beta=0.5, odds=ODDS)
    # The market's disagreements move in the market's direction…
    assert out["OverRated"] < CARRIED["OverRated"]
    assert out["Upstart"] > CARRIED["Upstart"]
    # …and the league level does not move.
    assert abs(mean(out.values()) - mean(CARRIED.values())) < 1e-6


def test_per_club_shift_is_capped():
    # beta 1 with an extreme market view must still respect the cap.
    extreme = dict(ODDS, OverRated="1/50")
    out = anchored_entry_elo(CARRIED, beta=1.0, odds=extreme)
    shifts = {t: abs(out[t] - CARRIED[t]) for t in CARRIED}
    # Mean re-centring can add at most the mean of the capped shifts on top.
    assert max(shifts.values()) <= MAX_SHIFT_ELO + MAX_SHIFT_ELO


def test_unknown_clubs_are_left_alone():
    carried = dict(CARRIED, Mystery=1480.0)      # not in the odds table
    out = anchored_entry_elo(carried, beta=0.5, odds=ODDS)
    # Mystery gets no market shift · only the shared re-centring never applies
    # to it because it is outside the anchored set.
    assert out["Mystery"] == 1480.0


def test_dc_anchor_splits_shift_between_attack_and_defense():
    dc = {"attacks": {t: (r - 1500.0) / 400.0 for t, r in CARRIED.items()},
          "defenses": {t: 0.0 for t in CARRIED},
          "home_adv": 0.25}
    out = anchored_dc_ratings(dc, beta=0.5, odds=ODDS)
    for t in CARRIED:
        d_att = out["attacks"][t] - dc["attacks"][t]
        d_def = out["defenses"][t] - dc["defenses"][t]
        # Equal and opposite: a stronger club scores more AND concedes less.
        assert math.isclose(d_att, -d_def, abs_tol=1e-9)
        assert abs(d_att) <= MAX_SHIFT_DC          # half the net cap each side
    # Net strength moves the right way for the market's two disagreements.
    net = lambda r, t: r["attacks"][t] - r["defenses"][t]
    assert net(out, "OverRated") < net(dc, "OverRated")
    assert net(out, "Upstart") > net(dc, "Upstart")
    # Untouched keys survive.
    assert out["home_adv"] == dc["home_adv"]


def test_degenerate_odds_tables_do_nothing():
    assert market_strength_z({"A": "1/2"}) == {}
    assert anchored_entry_elo(CARRIED, beta=0.5, odds={"A": "1/2"}) == CARRIED
