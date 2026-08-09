"""Tests for research-track additions:
  - Karlis-Ntzoufras diagonal-inflation Dixon-Coles variant
  - Baker-McHale uncertainty-shrunk Kelly
  - Simultaneous-bet Kelly correction
  - CLV (closing-line value) computation + closing-odds extraction
  - Per-bin Var(p̂) helpers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models as md
import portfolio as pf


# ─────────────────────────────────────────────────────────────────────────────
# Karlis-Ntzoufras D-C variant
# ─────────────────────────────────────────────────────────────────────────────

def _toy_kn_ratings(gamma: float = 0.0) -> dict:
    return {
        "teams":         ["A", "B"],
        "team_idx":      {"A": 0, "B": 1},
        "attacks":       {"A": 0.0, "B": 0.0},
        "defenses":      {"A": 0.0, "B": 0.0},
        "home_adv":      0.25,
        "rho":           -0.10,
        "gamma":         gamma,
        "model_variant": "dixon-coles-kn",
        "converged":     True,
    }


def test_kn_probabilities_sum_to_one():
    pred = md.predict_dixon_coles_kn("A", "B", _toy_kn_ratings(gamma=0.10))
    total = pred["home_win"] + pred["draw"] + pred["away_win"]
    assert abs(total - 1.0) < 1e-9


def test_kn_gamma_zero_matches_standard_dc():
    """γ=0 must reduce exactly to standard Dixon-Coles."""
    r = _toy_kn_ratings(gamma=0.0)
    kn_pred = md.predict_dixon_coles_kn("A", "B", r)
    dc_pred = md.predict_dixon_coles("A", "B", r)
    assert abs(kn_pred["draw"]     - dc_pred["draw"])     < 1e-9
    assert abs(kn_pred["home_win"] - dc_pred["home_win"]) < 1e-9
    assert abs(kn_pred["away_win"] - dc_pred["away_win"]) < 1e-9


def test_kn_positive_gamma_lifts_draw():
    p_zero = md.predict_dixon_coles_kn("A", "B", _toy_kn_ratings(gamma=0.0))["draw"]
    p_pos  = md.predict_dixon_coles_kn("A", "B", _toy_kn_ratings(gamma=0.15))["draw"]
    assert p_pos > p_zero


def test_kn_negative_gamma_reduces_draw():
    p_zero = md.predict_dixon_coles_kn("A", "B", _toy_kn_ratings(gamma=0.0))["draw"]
    p_neg  = md.predict_dixon_coles_kn("A", "B", _toy_kn_ratings(gamma=-0.15))["draw"]
    assert p_neg < p_zero


def test_kn_fit_converges_on_synthetic(synthetic_match_history):
    fit = md.compute_dixon_coles_kn_ratings(synthetic_match_history, decay_weeks=14)
    assert fit["converged"]
    # γ is bounded by ±0.20
    assert -0.20 <= fit["gamma"] <= 0.20
    assert "gamma" in fit
    assert fit.get("model_variant") == "dixon-coles-kn"


# ─────────────────────────────────────────────────────────────────────────────
# Baker-McHale uncertainty-shrunk Kelly
# ─────────────────────────────────────────────────────────────────────────────

def test_uncertainty_kelly_zero_var_matches_flat_kelly():
    """With Var(p̂)=0 (perfect knowledge), shrinkage should be 1.0."""
    flat = pf.kelly_stake_amount(0.30, 4.0, 10000.0, fraction=0.5)
    stake, shrink = pf.kelly_stake_uncertainty_adjusted(
        0.30, 4.0, 10000.0, var_p=0.0, fraction=0.5,
    )
    assert shrink == 1.0
    assert stake == flat


def test_uncertainty_kelly_at_max_binomial_var_returns_zero():
    """When Var(p̂) reaches the binomial max p(1−p), shrinkage → 0, stake → 0."""
    p = 0.30
    var_max = p * (1.0 - p)
    stake, shrink = pf.kelly_stake_uncertainty_adjusted(
        p, 4.0, 10000.0, var_p=var_max, fraction=0.5,
    )
    assert shrink == 0.0
    assert stake == 0.0


def test_uncertainty_kelly_intermediate_var_shrinks_partially():
    p = 0.30
    half_max = p * (1.0 - p) * 0.5
    stake, shrink = pf.kelly_stake_uncertainty_adjusted(
        p, 4.0, 10000.0, var_p=half_max, fraction=0.5,
    )
    flat = pf.kelly_stake_amount(p, 4.0, 10000.0, fraction=0.5)
    assert 0 < shrink < 1
    assert 0 < stake < flat
    # Shrinkage of 0.5 means stake = 0.5 × flat half-Kelly
    assert abs(stake - flat * 0.5) < 0.01


def test_uncertainty_kelly_no_edge_returns_zero():
    # Fair odds at p=0.5 → no edge regardless of variance
    stake, _ = pf.kelly_stake_uncertainty_adjusted(0.50, 2.0, 10000.0, var_p=0.0)
    assert stake == 0.0


def test_uncertainty_kelly_invalid_odds_returns_zero():
    stake, shrink = pf.kelly_stake_uncertainty_adjusted(0.50, 1.0, 10000.0, var_p=0.0)
    assert stake == 0.0
    assert shrink == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Simultaneous-bet Kelly correction
# ─────────────────────────────────────────────────────────────────────────────

def test_simultaneous_correction_empty_returns_empty():
    assert pf.simultaneous_kelly_correction([]) == []


def test_simultaneous_correction_single_bet_unchanged():
    assert pf.simultaneous_kelly_correction([0.05]) == [0.05]


def test_simultaneous_correction_reduces_three_concurrent():
    out = pf.simultaneous_kelly_correction([0.05, 0.05, 0.05])
    expected = 0.05 * (1 - 0.05) ** 2  # 0.04512 5
    assert all(abs(v - expected) < 1e-6 for v in out)


def test_simultaneous_correction_collapses_when_one_full_kelly():
    """If one bet would consume 100% of bankroll, others must be 0."""
    out = pf.simultaneous_kelly_correction([1.0, 0.05])
    assert out[1] == 0.0
    # First bet untouched by the others (they have small kelly)
    assert abs(out[0] - 1.0 * (1 - 0.05)) < 1e-6


def test_simultaneous_correction_clamps_negative_inputs():
    """Negative or >1 kelly values should be safely clamped."""
    out = pf.simultaneous_kelly_correction([-0.01, 0.05])
    # Negative clamped to 0 — its multiplier (1-0) = 1, so other unchanged
    assert abs(out[1] - 0.05) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# CLV computation + closing-odds extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_clv_positive_when_taken_better():
    # Took 5.0, market closed at 4.0 → 25% better than the close
    clv = pf.compute_clv(5.0, 4.0)
    assert clv == round(5.0 / 4.0 - 1.0, 4)
    assert clv > 0


def test_compute_clv_negative_when_taken_worse():
    clv = pf.compute_clv(3.0, 4.0)
    assert clv is not None and clv < 0


def test_compute_clv_zero_at_equal_prices():
    assert pf.compute_clv(3.5, 3.5) == 0.0


def test_compute_clv_invalid_returns_none():
    assert pf.compute_clv(0, 3.5)   is None
    assert pf.compute_clv(3.5, 0)   is None
    assert pf.compute_clv(1.0, 3.5) is None
    assert pf.compute_clv(3.5, 1.0) is None
    assert pf.compute_clv(None, 3.5) is None  # type: ignore[arg-type]


def test_extract_closing_odds_prefers_pinnacle():
    df = pd.DataFrame({
        "Date":     pd.to_datetime(["2026-04-20"]),
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "PSH":      [2.50],
        "PSD":      [3.40],
        "PSA":      [2.80],
        "AvgD":     [3.50],
    })
    assert pf.extract_closing_odds("A", "B", "2026-04-20", "D", df) == 3.40


def test_extract_closing_odds_falls_back_when_pinnacle_missing():
    df = pd.DataFrame({
        "Date":     pd.to_datetime(["2026-04-20"]),
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "PSD":      [np.nan],
        "AvgD":     [3.55],
        "B365D":    [3.60],
    })
    # Should pick AvgD before B365D
    assert pf.extract_closing_odds("A", "B", "2026-04-20", "D", df) == 3.55


def test_extract_closing_odds_returns_none_when_no_match():
    df = pd.DataFrame({
        "Date":     pd.to_datetime(["2026-04-20"]),
        "HomeTeam": ["X"],
        "AwayTeam": ["Y"],
        "PSD":      [3.40],
    })
    assert pf.extract_closing_odds("A", "B", "2026-04-20", "D", df) is None


def test_extract_closing_odds_skips_invalid_values():
    df = pd.DataFrame({
        "Date":     pd.to_datetime(["2026-04-20"]),
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "PSD":      [1.0],   # invalid (≤ 1.0)
        "AvgD":     [3.55],
    })
    assert pf.extract_closing_odds("A", "B", "2026-04-20", "D", df) == 3.55


def test_backfill_clv_does_not_mutate_stake_or_status():
    df = pd.DataFrame({
        "Date":     pd.to_datetime(["2026-04-20"]),
        "HomeTeam": ["A"], "AwayTeam": ["B"],
        "PSD":      [3.50],
    })
    p = {
        "initial_bankroll": 10000.0,
        "bankroll": 9100.0,
        "bets": [{
            "id": "x", "home": "A", "away": "B", "date": "2026-04-20",
            "market": "D", "selection": "Draw",
            "model_prob": 0.30, "odds": 4.00, "ev": 0.20, "stake": 200.0,
            "status": "won", "profit": 600.0,
            "placed_at": "2026-04-19T10:00:00", "settled_at": "2026-04-20T17:00:00",
        }],
        "settings": {},
    }
    n = pf.backfill_clv_for_settled_bets(p, df)
    bet = p["bets"][0]
    assert n == 1
    assert bet["stake"]  == 200.0
    assert bet["odds"]   == 4.00
    assert bet["status"] == "won"
    assert bet["profit"] == 600.0
    # New CLV fields added
    assert bet["closing_odds"] == 3.50
    assert bet["clv"] == round(4.00 / 3.50 - 1.0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-bin variance helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_per_bin_variance_empty_returns_empty():
    assert pf.compute_per_bin_variance(pd.DataFrame(), "D") == []


def test_compute_per_bin_variance_missing_columns_returns_empty():
    assert pf.compute_per_bin_variance(pd.DataFrame({"unrelated": [1, 2]}), "D") == []


def test_compute_per_bin_variance_real_shape():
    bt = pd.DataFrame({
        "_dc_d":  [0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.50],
        "_act_d": [   0,    0,    0,    1,    1,    0,    1],
    })
    bins = pf.compute_per_bin_variance(bt, "D", n_bins=4)
    assert len(bins) == 4
    # At least one bin should have observations
    assert any(b["n"] > 0 for b in bins)
    # All present variances are non-negative
    assert all(b["var"] >= 0 for b in bins if b["var"] is not None)


def test_lookup_bin_variance_finds_correct_bin():
    bins = [{"lo": 0.0, "hi": 0.5, "n": 10, "freq": 0.4, "var": 0.024},
            {"lo": 0.5, "hi": 1.0, "n": 5,  "freq": 0.6, "var": 0.048}]
    assert pf.lookup_bin_variance(bins, 0.30) == 0.024
    assert pf.lookup_bin_variance(bins, 0.70) == 0.048


def test_lookup_bin_variance_uses_fallback_when_empty():
    assert pf.lookup_bin_variance([], 0.30, fallback=0.05) == 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Mock Portfolio Two persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_two_loads_with_default_settings_when_file_missing(tmp_path, monkeypatch):
    # Point MOCK2_PORTFOLIO_FILE at a non-existent path
    fake = tmp_path / "portfolio_two_missing.json"
    monkeypatch.setattr(pf, "MOCK2_PORTFOLIO_FILE", fake)
    p = pf.load_portfolio_two()
    assert p["initial_bankroll"] == 10000.0
    assert p["bankroll"] == 10000.0
    assert p["bets"] == []
    s = p["settings"]
    assert s["use_kn_model"]            is True
    assert s["use_uncertainty_kelly"]   is True
    assert s["use_simultaneous_kelly"]  is True
    assert s["model_variant"]           == "dixon-coles-kn"


def test_mock_two_save_then_load_roundtrips(tmp_path, monkeypatch):
    fake = tmp_path / "portfolio_two_roundtrip.json"
    monkeypatch.setattr(pf, "MOCK2_PORTFOLIO_FILE", fake)
    p = pf.load_portfolio_two()
    p["bets"].append({"id": "abc", "stake": 100.0, "status": "pending"})
    pf.save_portfolio_two(p)
    p2 = pf.load_portfolio_two()
    assert len(p2["bets"]) == 1
    assert p2["bets"][0]["id"] == "abc"
