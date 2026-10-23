"""Mock Two research · preseason market anchor for season-entry ratings.

The problem: carried-over ratings know nothing a summer did — a new manager,
a spent transfer window, a gutted squad. The preseason market prices all of
it. Rather than model spend or managers ourselves (weak signal, tiny sample,
and the market subsumes both), anchor each club's season-entry rating part of
the way toward what the market implies.

The instrument is the full-league relegation table already hand-captured in
`models.MARKET_RELEGATION_ODDS` — the same market, capture ritual and
staleness tracking the promoted-side mechanism trusts. Its resolution is best
at the bottom of the league and coarse at the top (50/1 Spurs vs 66/1 Villa
separates poorly), so the map is deliberately rank-affine rather than
anything cleverer: market order is trusted further than market magnitude,
exactly the MARKET_SHRINKAGE philosophy.

Validation status (2026-08-16, docs/WORKFLOW.md): a one-fixture-odds proxy of
this anchor improved early-season Brier in 3 of 4 walk-forward seasons at
beta 0.5, by a small margin. The season-market version cannot be backtested
(no archived preseason odds), so it runs as a Mock Two PAPER LINE for
2026-27: `scripts/mock2_market_anchor_sim.py` freezes both the anchored and
unanchored season simulations to disk in August, and the pair is scored
against the real table in May.

RESEARCH TRACK ONLY. Nothing in the live app imports this module.
"""
from __future__ import annotations

import math
from statistics import mean, stdev

from models import market_relegation_probs

# How far toward the market an entry rating moves. 0 = pure carryover,
# 1 = pure market. 0.5 mirrors MARKET_SHRINKAGE and the walk-forward study's
# least-bad beta. A Mock Two knob, not a live constant.
DEFAULT_BETA: float = 0.5

# A summer is not allowed to move a club more than this many Elo points.
# The largest single-season Elo change in the five loaded seasons is ~130;
# a PRIOR should not claim more than a season's worth of evidence.
MAX_SHIFT_ELO: float = 120.0

# Same cap in Dixon-Coles net-strength units (att − def, log-goals scale).
MAX_SHIFT_DC: float = 0.35


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def market_strength_z(odds: dict | None = None) -> dict[str, float]:
    """Standardised full-league strength score from the relegation market.

    −logit(P(relegation)) rises with quality; standardising across the twenty
    clubs makes it unit-free so it can be mapped onto any rating scale by an
    affine transform. Returns {} if the odds table is unusable.
    """
    probs = market_relegation_probs(odds)
    if len(probs) < 4:
        return {}
    # The 3.0-normalised book can push an extreme favourite past 1.0 ·
    # dropping it would remove exactly the club the market is loudest about,
    # so clamp into logit's domain instead.
    scores = {t: -_logit(min(max(p, 0.005), 0.985)) for t, p in probs.items()}
    mu = mean(scores.values())
    sd = stdev(scores.values())
    if sd <= 0:
        return {}
    return {t: (s - mu) / sd for t, s in scores.items()}


def _affine_anchor(carried: dict[str, float], z: dict[str, float],
                   beta: float, cap: float) -> dict[str, float]:
    """Move each club toward the market's implied position on ITS OWN scale.

    The market z-scores are mapped onto the carried distribution's mean and
    spread, so the anchor reorders clubs without inflating or deflating the
    scale, then the league mean is restored exactly · both Elo and DC are
    zero-sum-ish scales and a prior must not shift the whole league.
    """
    both = [t for t in carried if t in z]
    if len(both) < 4 or beta <= 0:
        return dict(carried)
    mu = mean(carried[t] for t in both)
    sd = stdev(carried[t] for t in both)
    out = dict(carried)
    for t in both:
        implied = mu + sd * z[t]
        shift = beta * (implied - carried[t])
        out[t] = carried[t] + max(-cap, min(cap, shift))
    # Restore the mean over the anchored set so the league level is untouched.
    drift = mean(out[t] for t in both) - mu
    for t in both:
        out[t] -= drift
    return out


def anchored_entry_elo(carried_elo: dict[str, float],
                       beta: float = DEFAULT_BETA,
                       odds: dict | None = None) -> dict[str, float]:
    """Season-entry Elo with the summer market blended in.

    Only clubs present in BOTH the carried ratings and the odds table move ·
    promoted sides carry no rating here and keep their existing entry
    mechanism (`entry_offsets` on the Elo series), which this composes with.
    """
    return _affine_anchor(carried_elo, market_strength_z(odds), beta,
                          MAX_SHIFT_ELO)


def anchored_dc_ratings(dc_ratings: dict, beta: float = DEFAULT_BETA,
                        odds: dict | None = None) -> dict:
    """Dixon-Coles ratings with the summer market blended in.

    The market speaks one number per club; DC carries two. The shift lands on
    net strength (attack − defense, the log-goal-difference scale) and is
    split half to attack, half against defense, so a market upgrade makes a
    club both score more and concede less rather than only one of them.
    """
    att, dfn = dict(dc_ratings["attacks"]), dict(dc_ratings["defenses"])
    strength = {t: att[t] - dfn[t] for t in att if t in dfn}
    anchored = _affine_anchor(strength, market_strength_z(odds), beta,
                              MAX_SHIFT_DC)
    out = dict(dc_ratings)
    out["attacks"], out["defenses"] = dict(att), dict(dfn)
    for t, s_new in anchored.items():
        d = s_new - strength[t]
        out["attacks"][t] = att[t] + d / 2.0
        out["defenses"][t] = dfn[t] - d / 2.0
    return out
