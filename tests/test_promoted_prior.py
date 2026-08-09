"""Seeding promoted teams with an honest prior instead of league average.

`predict_dixon_coles` falls back to `attacks.get(team, 0.0)`, so a team with no
rating is modelled as exactly league average. For a promoted side that is not
merely missing, it is optimistic: it hands a newly promoted team the attack and
defence of a mid-table Premier League club.

The size of that error, measured on the real opening fixture: Arsenal v
Coventry prices at 33.6% for the draw as league average, against 12.0% under
the promoted prior and roughly 10.9% implied by the market. At a best price of
9.2 that difference is the gap between a +209% EV signal and a +10% one, which
is the difference between a £4,562 full-Kelly stake and no bet.

The prior itself comes from `scripts/promoted_team_prior.py`: fifteen promoted
teams, 2021-22 to 2025-26. Leave-one-out testing there found that no
Championship signal beats the pooled mean, so the prior is deliberately flat.
"""
from __future__ import annotations

import pytest

from models import PROMOTED_PRIOR, predict_dixon_coles, seed_promoted_teams


def _ratings() -> dict:
    return {
        "teams": ["Arsenal", "Everton"],
        "team_idx": {"Arsenal": 0, "Everton": 1},
        "attacks":  {"Arsenal": 0.05, "Everton": -0.30},
        "defenses": {"Arsenal": -0.18, "Everton": 0.40},
        "home_adv": 0.26,
        "rho": -0.10,
        "converged": True,
    }


# ── Seeding ───────────────────────────────────────────────────────────────────

def test_missing_team_is_seeded_with_the_promoted_prior():
    # An unpriced side still gets the pooled value; a side the relegation market
    # prices is now ordered against the 60% base rate instead (see
    # tests/test_market_promoted.py), so this asserts the fallback path.
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Everton", "Unlisted FC"])
    assert seeded["attacks"]["Unlisted FC"] == PROMOTED_PRIOR["attack"]
    assert seeded["defenses"]["Unlisted FC"] == PROMOTED_PRIOR["defense"]


def test_seeding_records_which_teams_it_invented():
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Coventry", "Hull"])
    assert sorted(seeded["seeded_teams"]) == ["Coventry", "Hull"]


def test_established_teams_are_untouched():
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Everton", "Coventry"])
    assert seeded["attacks"]["Arsenal"] == 0.05
    assert seeded["defenses"]["Everton"] == 0.40


def test_seeding_does_not_mutate_the_original_ratings():
    original = _ratings()
    seed_promoted_teams(original, ["Arsenal", "Coventry"])
    assert "Coventry" not in original["attacks"]
    assert "seeded_teams" not in original


def test_seeded_team_joins_the_team_list():
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Coventry"])
    assert "Coventry" in seeded["teams"]
    assert seeded["team_idx"]["Coventry"] == len(seeded["teams"]) - 1


def test_nothing_to_seed_leaves_an_empty_marker():
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Everton"])
    assert seeded["seeded_teams"] == []


# ── The prior itself ──────────────────────────────────────────────────────────

def test_prior_is_worse_than_league_average_in_both_phases():
    """Lower attack is worse; higher defence is worse (Arsenal -0.18 is best)."""
    assert PROMOTED_PRIOR["attack"] < 0.0
    assert PROMOTED_PRIOR["defense"] > 0.0


def test_prior_carries_its_own_uncertainty_and_provenance():
    for key in ("attack_sd", "defense_sd", "n_teams", "fitted_on", "source"):
        assert key in PROMOTED_PRIOR
    assert PROMOTED_PRIOR["n_teams"] == 15


# ── Effect on the number that actually matters ────────────────────────────────

def test_seeding_cuts_the_draw_probability_against_a_strong_side():
    ratings = _ratings()
    as_average = predict_dixon_coles("Arsenal", "Coventry", ratings)["draw"]
    seeded     = seed_promoted_teams(ratings, ["Arsenal", "Coventry"])
    as_promoted = predict_dixon_coles("Arsenal", "Coventry", seeded)["draw"]

    assert as_promoted < as_average
    # The distortion is large, not marginal — this is the whole point.
    assert as_average - as_promoted > 0.10


@pytest.mark.parametrize("market", ["home_win", "draw", "away_win"])
def test_seeded_predictions_stay_a_valid_distribution(market):
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Coventry"])
    p = predict_dixon_coles("Arsenal", "Coventry", seeded)
    assert 0.0 <= p[market] <= 1.0
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 0.02


# ── Resolving a promoted side's name ──────────────────────────────────────────
#
# ESPN calls them "Coventry City" and "Hull City"; football-data calls them
# "Coventry" and "Hull". Neither appears in any Premier League CSV, so the
# top-flight name list cannot resolve them and the seeded ratings would land
# under names nothing else in the pipeline uses.

import data as fpred_data


@pytest.fixture
def name_lists(monkeypatch):
    monkeypatch.setattr(fpred_data, "KNOWN_FD_TEAMS",
                        {"Arsenal", "Man United", "Leeds", "Sheffield United"})
    monkeypatch.setattr(fpred_data, "KNOWN_SECOND_TIER_TEAMS",
                        {"Coventry", "Hull", "Millwall", "Sheffield Weds"})


def test_promoted_side_resolves_from_the_second_tier_list(name_lists):
    assert fpred_data._resolve_team_name("Coventry City") == "Coventry"
    assert fpred_data._resolve_team_name("Hull City") == "Hull"


def test_top_flight_names_still_win(name_lists):
    assert fpred_data._resolve_team_name("Leeds United") == "Leeds"
    assert fpred_data._resolve_team_name("Arsenal") == "Arsenal"


def test_a_similar_second_tier_name_does_not_hijack_a_top_flight_one(name_lists):
    """Sheffield Weds must not capture Sheffield United."""
    assert fpred_data._resolve_team_name("Sheffield United") == "Sheffield United"


def test_an_unknown_name_is_returned_unchanged(name_lists):
    assert fpred_data._resolve_team_name("Real Madrid") == "Real Madrid"


def test_missing_championship_file_is_a_silent_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(fpred_data, "DATA_DIR", tmp_path)
    assert fpred_data.load_promoted_team_names("2025-26") == set()
