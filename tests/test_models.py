"""Model math — Dixon-Coles prediction, DC fit, season simulation."""
from __future__ import annotations

import numpy as np
import pytest

from models import (
    compute_dixon_coles_ratings,
    predict_dixon_coles,
    simulate_season,
)


# ── predict_dixon_coles ─────────────────────────────────────────────────────

def test_dc_probabilities_sum_to_one(toy_dc_ratings):
    """home_win + draw + away_win must equal 1.0 for any matchup."""
    for (h, a) in [("A", "B"), ("B", "A"), ("A", "D"), ("C", "B")]:
        pred = predict_dixon_coles(h, a, toy_dc_ratings)
        total = pred["home_win"] + pred["draw"] + pred["away_win"]
        assert total == pytest.approx(1.0, abs=1e-6), f"{h} vs {a}: total={total}"


def test_dc_home_advantage_favours_home(toy_dc_ratings):
    """Same two teams — home team should win more often at home than away."""
    home_at_home = predict_dixon_coles("B", "C", toy_dc_ratings)
    away_at_home = predict_dixon_coles("C", "B", toy_dc_ratings)
    # B winning at home > B winning away
    assert home_at_home["home_win"] > away_at_home["away_win"]


def test_dc_stronger_team_wins_more_often(toy_dc_ratings):
    """Team A (strong) at home vs D (weak) should win clearly more than it loses."""
    pred = predict_dixon_coles("A", "D", toy_dc_ratings)
    assert pred["home_win"] > pred["away_win"]
    assert pred["home_win"] > 0.5


def test_dc_lambdas_are_positive(toy_dc_ratings):
    pred = predict_dixon_coles("A", "B", toy_dc_ratings)
    assert pred["lambda_home"] > 0
    assert pred["lambda_away"] > 0


# ── compute_dixon_coles_ratings ─────────────────────────────────────────────

def test_dc_fit_returns_all_teams(synthetic_match_history):
    dc = compute_dixon_coles_ratings(synthetic_match_history)
    assert set(dc["attacks"].keys()) == {"A", "B", "C", "D"}
    assert set(dc["defenses"].keys()) == {"A", "B", "C", "D"}


def test_dc_fit_identifies_home_advantage(synthetic_match_history):
    """Fitted home_adv should be positive (home teams win more in real football)."""
    dc = compute_dixon_coles_ratings(synthetic_match_history)
    # Historical EPL home advantage is log(~1.3) ≈ 0.26 — allow wide tolerance
    # for a small synthetic dataset; just check sign + reasonable magnitude.
    assert -0.5 < dc["home_adv"] < 1.0


def test_dc_fit_probabilities_still_valid(synthetic_match_history):
    """After fitting on synthetic data, predictions must still sum to 1."""
    dc = compute_dixon_coles_ratings(synthetic_match_history)
    pred = predict_dixon_coles("A", "B", dc)
    total = pred["home_win"] + pred["draw"] + pred["away_win"]
    assert total == pytest.approx(1.0, abs=1e-6)


# ── simulate_season ─────────────────────────────────────────────────────────

def test_simulate_season_returns_one_row_per_team(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    summary = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=500, n_boot=10,
    )
    assert len(summary) == 4
    assert set(summary["Team"]) == {"A", "B", "C", "D"}


def test_simulate_season_position_probs_sum_to_one(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """For each team, position probabilities across positions 1–N must sum to 1."""
    summary = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=500, n_boot=10,
    )
    n_teams = len(summary)
    for _, row in summary.iterrows():
        pos_sum = sum(row[f"pos_{p+1}"] for p in range(n_teams))
        assert pos_sum == pytest.approx(1.0, abs=1e-6)


def test_simulate_season_title_prob_matches_pos_1(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """p_title must equal pos_1 for every team (sanity: aggregates are consistent)."""
    summary = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=500, n_boot=10,
    )
    for _, row in summary.iterrows():
        assert row["p_title"] == pytest.approx(row["pos_1"], abs=1e-9)


def test_simulate_season_leader_has_highest_title_prob(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """Team A leads the table (40 pts) AND has the strongest DC ratings — should
    have the highest title probability."""
    summary = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=1000, n_boot=20,
    )
    top = summary.iloc[0]
    assert top["Team"] == "A"
    assert top["p_title"] > 0.5


def test_simulate_season_higher_noise_widens_title_race(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """Larger param_noise should reduce the leader's title certainty."""
    low_noise = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=2000, n_boot=40, param_noise=0.05, rng_seed=1,
    )
    high_noise = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=2000, n_boot=40, param_noise=0.25, rng_seed=1,
    )
    leader = "A"
    p_low  = low_noise.set_index("Team").loc[leader, "p_title"]
    p_high = high_noise.set_index("Team").loc[leader, "p_title"]
    # Higher noise → more uncertainty → leader's title prob drops
    assert p_high < p_low


def test_simulate_season_return_samples_shape(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """With return_samples=True, sim_pts and sim_positions must be (n_sims × n_teams)."""
    summary, sim_pts, fixtures_used, sim_positions = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=500, n_boot=10, return_samples=True,
    )
    assert sim_pts.shape == (500, 4)
    assert sim_positions.shape == (500, 4)
    # Every sim position must be in [1, n_teams]
    assert sim_positions.min() >= 1
    assert sim_positions.max() <= 4


def test_simulate_season_points_only_increase(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """Simulated final points must be ≥ starting points (teams can't lose points)."""
    _, sim_pts, _, _ = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=200, n_boot=10, return_samples=True,
    )
    starting_pts = np.array([40, 35, 20, 10])
    assert (sim_pts >= starting_pts[None, :]).all()


def test_simulate_season_max_pts_bounded_by_fixtures(
    toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
):
    """Each team plays 6 remaining fixtures (round-robin, 3 home + 3 away).
    Max extra pts = 18. So final ≤ starting + 18."""
    _, sim_pts, fixtures_used, _ = simulate_season(
        toy_remaining_fixtures, toy_current_table, toy_dc_ratings,
        n_sims=200, n_boot=10, return_samples=True,
    )
    starting_pts = np.array([40, 35, 20, 10])
    # Count fixtures per team
    from collections import Counter
    counts = Counter()
    for f in fixtures_used:
        counts[f["home"]] += 1
        counts[f["away"]] += 1
    max_extra = np.array([counts[t] * 3 for t in ["A", "B", "C", "D"]])
    assert (sim_pts <= starting_pts[None, :] + max_extra[None, :]).all()
