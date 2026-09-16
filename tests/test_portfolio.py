"""Portfolio math — EV, Kelly sizing, implied probability."""
from __future__ import annotations

import pytest

from portfolio import compute_ev, implied_prob, kelly_stake_amount


# ── compute_ev ───────────────────────────────────────────────────────────────

def test_compute_ev_zero_edge_at_fair_odds():
    """50% prob at 2.0 odds → zero edge (fair bet)."""
    assert compute_ev(0.5, 2.0) == pytest.approx(0.0)


def test_compute_ev_positive_edge():
    """60% prob at 2.0 odds → +20% EV."""
    assert compute_ev(0.6, 2.0) == pytest.approx(0.2)


def test_compute_ev_negative_edge():
    """30% prob at 2.0 odds → −40% EV."""
    assert compute_ev(0.3, 2.0) == pytest.approx(-0.4)


# ── implied_prob ─────────────────────────────────────────────────────────────

def test_implied_prob_standard():
    assert implied_prob(2.0) == pytest.approx(0.5)
    assert implied_prob(4.0) == pytest.approx(0.25)


def test_implied_prob_invalid_odds_returns_zero():
    """Odds ≤ 1.0 are degenerate — returns 0 rather than raising."""
    assert implied_prob(1.0) == 0.0
    assert implied_prob(0.5) == 0.0


# ── kelly_stake_amount ───────────────────────────────────────────────────────

def test_kelly_zero_stake_when_no_edge():
    """At fair odds, full Kelly = 0 → stake = 0."""
    stake = kelly_stake_amount(0.5, 2.0, bankroll=10_000, fraction=0.5)
    assert stake == 0.0


def test_kelly_zero_stake_when_negative_edge():
    """Under-priced bet (model < bookmaker implied) must stake nothing."""
    stake = kelly_stake_amount(0.4, 2.0, bankroll=10_000, fraction=0.5)
    assert stake == 0.0


def test_kelly_positive_stake_when_edge_exists():
    """60% prob at 2.0 odds, full Kelly = 0.2, half-Kelly = 0.1 → £1,000."""
    stake = kelly_stake_amount(0.6, 2.0, bankroll=10_000, fraction=0.5, max_pct=0.5)
    # Full Kelly f* = (p·b − q)/b = (0.6·1 − 0.4)/1 = 0.2
    # Half Kelly = 0.1 → £1,000 on £10k
    assert stake == pytest.approx(1000.0)


def test_kelly_respects_max_pct_cap():
    """Massive edge must still respect the bankroll cap."""
    # 90% prob at 10.0 odds → huge edge
    stake = kelly_stake_amount(0.9, 10.0, bankroll=10_000, fraction=1.0, max_pct=0.05)
    # Cap at 5% of bankroll = £500
    assert stake == pytest.approx(500.0)


def test_kelly_scales_linearly_with_bankroll():
    """Stake should double when bankroll doubles (same edge, same fraction)."""
    s1 = kelly_stake_amount(0.6, 2.0, bankroll=10_000, fraction=0.5, max_pct=0.5)
    s2 = kelly_stake_amount(0.6, 2.0, bankroll=20_000, fraction=0.5, max_pct=0.5)
    assert s2 == pytest.approx(2 * s1)


def test_kelly_rejects_invalid_odds():
    """Odds ≤ 1.0 make b ≤ 0 — return 0 defensively."""
    assert kelly_stake_amount(0.7, 1.0, bankroll=10_000) == 0.0
    assert kelly_stake_amount(0.7, 0.5, bankroll=10_000) == 0.0


# ── Isotonic calibration ─────────────────────────────────────────────────────

def test_calibrate_prob_identity_when_no_calibrator():
    """No calibrator for this market → return raw prob unchanged."""
    from portfolio import calibrate_prob
    assert calibrate_prob(0.37, "D", {}) == 0.37
    assert calibrate_prob(0.62, "X_unknown", {"D": object()}) == 0.62


def test_fit_and_apply_calibrator_monotonic():
    """Calibrator fitted on biased data should correct systematic over-confidence."""
    import numpy as np
    import pandas as pd
    from portfolio import calibrate_prob, fit_calibrators_from_backtest

    rng = np.random.default_rng(0)
    n = 500
    # Predicted probs uniform in [0.1, 0.9]
    preds = rng.uniform(0.1, 0.9, size=n)
    # Actual hits: systematic over-confidence — actual rate = 0.8 × predicted
    actual_rate = preds * 0.8
    acts = (rng.uniform(size=n) < actual_rate).astype(float)

    bt = pd.DataFrame({
        "_dc_h": rng.uniform(0.1, 0.9, n), "_act_h": rng.integers(0, 2, n),
        "_dc_d": preds,                     "_act_d": acts,
        "_dc_a": rng.uniform(0.1, 0.9, n), "_act_a": rng.integers(0, 2, n),
    })
    cals = fit_calibrators_from_backtest(bt)

    assert "D" in cals
    # Over-confident 0.7 model probability should calibrate downward
    calibrated = calibrate_prob(0.7, "D", cals)
    assert calibrated < 0.7
    # Monotonicity: calibrate(higher raw) >= calibrate(lower raw)
    assert calibrate_prob(0.8, "D", cals) >= calibrate_prob(0.4, "D", cals)


def test_fit_calibrators_skips_tiny_data():
    """With <20 samples the fit is unstable → skip that market."""
    import pandas as pd
    from portfolio import fit_calibrators_from_backtest
    tiny = pd.DataFrame({
        "_dc_h": [0.5] * 5, "_act_h": [1, 0, 1, 0, 1],
        "_dc_d": [0.3] * 5, "_act_d": [0, 1, 0, 0, 0],
        "_dc_a": [0.2] * 5, "_act_a": [0, 0, 0, 1, 0],
    })
    cals = fit_calibrators_from_backtest(tiny)
    assert cals == {}


def test_fit_calibrators_handles_empty_df():
    import pandas as pd
    from portfolio import fit_calibrators_from_backtest
    assert fit_calibrators_from_backtest(pd.DataFrame()) == {}


# ── Auto-bet prob gate ───────────────────────────────────────────────────────

def test_auto_bet_respects_min_prob_gate():
    """A high-EV long-shot (prob below min_prob) must be rejected."""
    from portfolio import auto_place_value_bets
    port = {
        "initial_bankroll": 10_000.0, "bankroll": 10_000.0,
        "bets": [],
        "settings": {
            "kelly_fraction": 0.5, "max_stake_pct": 0.10,
            "auto_markets": ["D"], "min_prob": 0.30,
            "use_calibrated_probs": False,
        },
    }
    # 15% prob at 10.0 odds = +50% EV (huge) — but below 30% prob gate
    longshot = [{
        "home": "A", "away": "B", "date": "2026-05-01",
        "market": "D", "selection": "Draw",
        "model_prob": 0.15, "odds": 10.0, "ev": 0.5,
    }]
    placed = auto_place_value_bets(port, longshot, threshold=0.05)
    assert placed == []


def test_auto_bet_accepts_when_prob_above_gate():
    """A +EV pick above the prob gate and in an allowed market should be placed."""
    from portfolio import auto_place_value_bets
    port = {
        "initial_bankroll": 10_000.0, "bankroll": 10_000.0,
        "bets": [],
        "settings": {
            "kelly_fraction": 0.5, "max_stake_pct": 0.10,
            "auto_markets": ["D"], "min_prob": 0.22,
            "use_calibrated_probs": False,
        },
    }
    cand = [{
        "home": "A", "away": "B", "date": "2026-05-01",
        "market": "D", "selection": "Draw",
        "model_prob": 0.35, "odds": 3.2, "ev": 0.35 * 3.2 - 1,   # +12%
    }]
    placed = auto_place_value_bets(port, cand, threshold=0.05)
    assert len(placed) == 1
    assert placed[0]["stake"] > 0


# ── Exposure cap: no zero-stake bets from a sub-penny remainder ──────────────

def _capped_portfolio():
    """Main's real state before the 09 Sep 2026 run: three pending draws that
    sum to £9,125.87 against a cap of £9,125.875 (50% of £18,251.75), leaving
    half a penny of float headroom."""
    pending = [
        ("Chelsea", "Hull", 3965.67),
        ("Sunderland", "Arsenal", 4265.28),
        ("Liverpool", "Fulham", 894.92),
    ]
    return {
        "initial_bankroll": 18_251.75, "bankroll": 20_841.72,
        "bets": [{"home": h, "away": a, "market": "D", "status": "pending",
                  "stake": s} for h, a, s in pending],
        "settings": {
            "kelly_fraction": 1.0, "max_stake_pct": 0.25,
            "auto_markets": ["D"], "min_prob": 0.17,
            "use_calibrated_probs": False,
        },
    }


_COVENTRY_BRIGHTON = [{
    "home": "Coventry", "away": "Brighton", "date": "2026-09-13",
    "market": "D", "selection": "Draw",
    "model_prob": 0.3597, "odds": 3.9, "ev": 0.3597 * 3.9 - 1,
}]


def test_auto_bet_skips_sub_penny_exposure_remainder():
    """Half a penny of cap headroom must not become a £0.00 bet."""
    from portfolio import auto_place_value_bets
    port = _capped_portfolio()
    placed = auto_place_value_bets(port, _COVENTRY_BRIGHTON, threshold=0.23)
    assert placed == []
    assert all(b["stake"] > 0 for b in port["bets"])
