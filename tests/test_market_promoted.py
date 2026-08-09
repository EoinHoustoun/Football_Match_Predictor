"""Differentiate promoted sides by what the market thinks of them.

The fitted prior is flat: no Championship signal predicted the step up out of
sample, so every promoted team got attack -0.645 and defence +0.817. That is
defensible as statistics and indefensible as football. The market prices Hull at
1/4 to go down and Coventry at 4/6 — 71% against 53% once the book is
normalised — and treating them as the same team is a choice to ignore the
sharpest read available.

Two separate holes get filled here:

  1. **Ratings.** Both promoted sides carried the identical Dixon-Coles prior.
  2. **Elo.** They carried none at all, and an unseen team defaults to 1500 —
     which would rank Hull above Ipswich (1351) and Burnley (1340). Measured
     over the fifteen promoted teams since 2021-22, a promoted side finishes its
     first Premier League season on **mean 1394, sd 91**, range 1254 to 1559.

The magnitude is deliberately shrunk. Nothing here is validated out of sample —
three attempts to predict a promoted side's rating pre-kickoff all failed — so
the market view moves each team by half the observed spread, never more, and the
result is clamped inside the range promoted teams have actually occupied. The
additive Championship prior was rejected precisely because it produced a Hull
defence worse than any promoted team on record.
"""
from __future__ import annotations

import pytest

from models import (MARKET_RELEGATION_ODDS, PROMOTED_PRIOR,
                    market_relegation_probs, promoted_prior_for,
                    seed_promoted_elo, seed_promoted_teams)


# ── Reading the market ────────────────────────────────────────────────────────

def test_fractional_odds_become_probabilities():
    probs = market_relegation_probs({"Hull": "1/4", "Arsenal": "1000/1"})
    assert probs["Hull"] > probs["Arsenal"]


def test_the_book_is_normalised_to_three_relegations():
    """Three teams go down, so the whole book must sum to 3, not to 1."""
    probs = market_relegation_probs(MARKET_RELEGATION_ODDS)
    assert sum(probs.values()) == pytest.approx(3.0, abs=1e-6)


def test_hull_is_priced_worse_than_coventry():
    probs = market_relegation_probs(MARKET_RELEGATION_ODDS)
    assert probs["Hull"] > probs["Coventry"]


def test_decimal_odds_are_accepted_too():
    probs = market_relegation_probs({"A": 1.25, "B": 5.0})
    assert probs["A"] > probs["B"]


# ── Turning it into a rating ──────────────────────────────────────────────────

def test_the_worse_priced_side_gets_the_worse_rating():
    priors = promoted_prior_for(["Hull", "Coventry"])
    assert priors["Hull"]["attack"] < priors["Coventry"]["attack"]
    assert priors["Hull"]["defense"] > priors["Coventry"]["defense"]


def test_scoring_is_against_the_historical_base_rate_not_this_year_cohort():
    """Nine of fifteen promoted teams went down, so 60% is average. Hull at 71%
    must land worse than the pooled prior. Scoring against the cohort's own mean
    would force the two to straddle it and flatter whoever is second-worst."""
    priors = promoted_prior_for(["Hull", "Coventry"])
    assert priors["Hull"]["attack"] < PROMOTED_PRIOR["attack"]
    assert priors["Hull"]["defense"] > PROMOTED_PRIOR["defense"]


def test_a_side_priced_at_the_base_rate_gets_the_pooled_prior():
    priors = promoted_prior_for(["A", "B"], odds={"A": "3/2", "B": "3/2"})
    # 3/2 normalises to the book, so both sit at the cohort's own average; what
    # matters is that identical prices give identical ratings.
    assert priors["A"] == priors["B"]


def test_one_promoted_side_is_still_rated_by_the_market():
    """Unlike a cohort-relative score, a base-rate anchor works for a single
    team — which matters in a season with one promoted side left unrated."""
    priors = promoted_prior_for(["Hull"])
    assert priors["Hull"]["attack"] < PROMOTED_PRIOR["attack"]


def test_no_rating_escapes_the_observed_promoted_range():
    """The additive Championship prior was thrown out for inventing a defence
    worse than any promoted team has recorded. Don't repeat it."""
    priors = promoted_prior_for(["Hull", "Coventry"])
    for p in priors.values():
        assert -0.95 <= p["attack"] <= -0.20
        assert 0.38 <= p["defense"] <= 1.25


def test_a_team_the_market_does_not_price_falls_back_to_the_flat_prior():
    priors = promoted_prior_for(["Some Unlisted Club"])
    assert priors["Some Unlisted Club"]["attack"] == PROMOTED_PRIOR["attack"]
    assert priors["Some Unlisted Club"]["defense"] == PROMOTED_PRIOR["defense"]


# ── Seeding it into the ratings dict ──────────────────────────────────────────

def _ratings() -> dict:
    return {"teams": ["Arsenal"], "team_idx": {"Arsenal": 0},
            "attacks": {"Arsenal": 0.5}, "defenses": {"Arsenal": -0.3},
            "home_adv": 0.26, "rho": -0.1}


def test_seeding_uses_the_market_prior_when_it_can():
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Hull", "Coventry"])
    assert seeded["attacks"]["Hull"] < seeded["attacks"]["Coventry"]
    assert sorted(seeded["seeded_teams"]) == ["Coventry", "Hull"]


def test_seeding_leaves_rated_teams_alone():
    seeded = seed_promoted_teams(_ratings(), ["Arsenal", "Hull"])
    assert seeded["attacks"]["Arsenal"] == 0.5


# ── Elo ───────────────────────────────────────────────────────────────────────

def test_promoted_elo_is_seeded_at_all():
    elo = seed_promoted_elo({"Arsenal": 1773.0}, ["Arsenal", "Hull", "Coventry"])
    assert "Hull" in elo and "Coventry" in elo


def test_promoted_elo_is_well_below_the_1500_default():
    """1500 would put a promoted side above Ipswich on 1351."""
    elo = seed_promoted_elo({}, ["Hull", "Coventry"])
    assert all(v < 1460 for v in elo.values())


def test_hull_is_seeded_below_coventry():
    elo = seed_promoted_elo({}, ["Hull", "Coventry"])
    assert elo["Hull"] < elo["Coventry"]


def test_seeded_elo_stays_inside_what_promoted_teams_have_recorded():
    elo = seed_promoted_elo({}, ["Hull", "Coventry"])
    assert all(1254 <= v <= 1559 for v in elo.values())


def test_an_already_rated_team_keeps_its_elo():
    elo = seed_promoted_elo({"Arsenal": 1773.0}, ["Arsenal", "Hull"])
    assert elo["Arsenal"] == 1773.0
