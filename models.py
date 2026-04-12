"""
Prediction models: Poisson goal model + Dixon-Coles + XGBoost form classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from xgboost import XGBClassifier, XGBRFClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Shared matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_matrix(lam_h: float, lam_a: float, matrix: np.ndarray, max_goals: int) -> dict:
    matrix = np.maximum(matrix, 0.0)
    matrix /= matrix.sum()

    hw = float(np.tril(matrix, -1).sum())
    d  = float(np.diag(matrix).sum())
    aw = float(np.triu(matrix, 1).sum())

    best = np.unravel_index(matrix.argmax(), matrix.shape)
    flat = sorted(
        [(matrix[i, j], i, j)
         for i in range(max_goals + 1) for j in range(max_goals + 1)],
        reverse=True,
    )
    top5 = [(int(i), int(j), round(p * 100, 1)) for p, i, j in flat[:5]]

    h_lo, h_hi = int(poisson.ppf(0.10, lam_h)), int(poisson.ppf(0.90, lam_h))
    a_lo, a_hi = int(poisson.ppf(0.10, lam_a)), int(poisson.ppf(0.90, lam_a))

    # Goals grid for vectorised market calculations
    gh = np.arange(max_goals + 1)[:, None]
    ga = np.arange(max_goals + 1)[None, :]
    total = gh + ga

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "home_win":    hw,
        "draw":        d,
        "away_win":    aw,
        "most_likely": (int(best[0]), int(best[1])),
        "score_matrix": matrix,
        "home_pmf":    matrix.sum(axis=1),
        "away_pmf":    matrix.sum(axis=0),
        "top5":        top5,
        "home_ci":     (h_lo, h_hi),
        "away_ci":     (a_lo, a_hi),
        # Additional markets derived directly from the score matrix
        "over_15":     float(matrix[total > 1].sum()),
        "over_25":     float(matrix[total > 2].sum()),
        "over_35":     float(matrix[total > 3].sum()),
        "btts":        float(matrix[1:, 1:].sum()),
        "home_cs":     float(matrix[:, 0].sum()),   # home clean sheet
        "away_cs":     float(matrix[0, :].sum()),   # away clean sheet
    }


# ─────────────────────────────────────────────────────────────────────────────
# Poisson model (multiplicative ratings, time-decayed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_poisson_ratings(df: pd.DataFrame, decay_weeks: float = 14.0) -> dict:
    """
    Compute per-team attack/defence ratings using exponential time decay.
    Uses xG when available for more stable estimates.
    """
    max_date = df["Date"].max()
    df = df.copy()
    df["days_ago"] = (max_date - df["Date"]).dt.days
    df["w"] = np.exp(-df["days_ago"] / (decay_weeks * 7))

    gh = "xg_h" if "xg_h" in df.columns else "FTHG"
    ga = "xg_a" if "xg_a" in df.columns else "FTAG"

    avg_home = float(np.average(df[gh], weights=df["w"]))
    avg_away = float(np.average(df[ga], weights=df["w"]))

    teams = sorted(set(df["HomeTeam"].tolist()) | set(df["AwayTeam"].tolist()))
    attack_home, attack_away   = {}, {}
    defense_home, defense_away = {}, {}

    for team in teams:
        hm = df[df["HomeTeam"] == team]
        aw = df[df["AwayTeam"] == team]
        w_hm, w_aw = hm["w"].sum(), aw["w"].sum()

        attack_home[team]  = ((hm[gh] * hm["w"]).sum() / w_hm / avg_home) if w_hm > 0.01 else 1.0
        attack_away[team]  = ((aw[ga] * aw["w"]).sum() / w_aw / avg_away) if w_aw > 0.01 else 1.0
        defense_home[team] = ((hm[ga] * hm["w"]).sum() / w_hm / avg_away) if w_hm > 0.01 else 1.0
        defense_away[team] = ((aw[gh] * aw["w"]).sum() / w_aw / avg_home) if w_aw > 0.01 else 1.0

    return {
        "attack_home":  attack_home,
        "attack_away":  attack_away,
        "defense_home": defense_home,
        "defense_away": defense_away,
        "avg_home":     avg_home,
        "avg_away":     avg_away,
        "teams":        teams,
    }


def predict_poisson(home_team: str, away_team: str, ratings: dict, max_goals: int = 8) -> dict:
    lam_h = max(
        ratings["attack_home"].get(home_team, 1.0)
        * ratings["defense_away"].get(away_team, 1.0)
        * ratings["avg_home"],
        0.05,
    )
    lam_a = max(
        ratings["attack_away"].get(away_team, 1.0)
        * ratings["defense_home"].get(home_team, 1.0)
        * ratings["avg_away"],
        0.05,
    )
    h_pmf = np.array([poisson.pmf(i, lam_h) for i in range(max_goals + 1)])
    a_pmf = np.array([poisson.pmf(i, lam_a) for i in range(max_goals + 1)])
    matrix = np.outer(h_pmf, a_pmf)
    return _summarise_matrix(lam_h, lam_a, matrix, max_goals)


# ─────────────────────────────────────────────────────────────────────────────
# Dixon-Coles model (MLE with low-score correction + time decay)
# ─────────────────────────────────────────────────────────────────────────────

def _dc_neg_log_lik(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hgoals: np.ndarray,   # actual integer goals — used for tau correction only
    agoals: np.ndarray,
    hxg: np.ndarray,      # xG values — used for Poisson likelihood
    axg: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Vectorised negative log-likelihood for the Dixon-Coles model.

    Hybrid xG approach:
      - Poisson likelihood fitted to xG (removes lucky/unlucky finishing noise)
      - τ correction applied using actual integer goals (discrete low-score fix)

    Parameterisation (log-space for identifiability):
        attack[0]  = 0 (fixed)
        attack[1:] = params[0 : n-1]
        defense    = params[n-1 : 2n-1]
        home_adv   = params[2n-1]
        rho        = params[2n]
    """
    attacks  = np.concatenate([[0.0], params[:n_teams - 1]])
    defenses = params[n_teams - 1: 2 * n_teams - 1]
    home_adv = params[2 * n_teams - 1]
    rho      = params[2 * n_teams]

    lam = np.exp(attacks[home_idx] + defenses[away_idx] + home_adv)
    mu  = np.exp(attacks[away_idx] + defenses[home_idx])

    # Poisson log-probability fitted to xG (gammaln handles continuous values)
    ll = (
        hxg * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(hxg + 1)
        + axg * np.log(np.maximum(mu,  1e-10)) - mu  - gammaln(axg + 1)
    )

    # Dixon-Coles τ correction on actual integer goals
    tau = np.ones(len(hgoals))
    m00 = (hgoals == 0) & (agoals == 0)
    m01 = (hgoals == 0) & (agoals == 1)
    m10 = (hgoals == 1) & (agoals == 0)
    m11 = (hgoals == 1) & (agoals == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10]  * rho
    tau[m11] = 1.0 - rho

    tau = np.maximum(tau, 1e-10)
    ll += np.log(tau)

    return -float(np.sum(weights * ll))


def compute_dixon_coles_ratings(df: pd.DataFrame, decay_weeks: float = 14.0) -> dict:
    """
    Fit Dixon-Coles model via maximum-likelihood with exponential time decay.
    Uses xG for the Poisson likelihood (better quality signal) and actual goals
    for the tau low-score correction.
    """
    max_date = df["Date"].max()
    df = df.copy()
    df["days_ago"] = (max_date - df["Date"]).dt.days
    df["w"] = np.exp(-df["days_ago"] / (decay_weeks * 7))

    teams   = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n       = len(teams)
    t_idx   = {t: i for i, t in enumerate(teams)}

    home_idx = df["HomeTeam"].map(t_idx).values.astype(int)
    away_idx = df["AwayTeam"].map(t_idx).values.astype(int)
    hgoals   = df["FTHG"].values.astype(int)
    agoals   = df["FTAG"].values.astype(int)
    # xG for Poisson likelihood — fall back to actual goals if not present
    hxg = df["xg_h"].values if "xg_h" in df.columns else hgoals.astype(float)
    axg = df["xg_a"].values if "xg_a" in df.columns else agoals.astype(float)
    weights  = df["w"].values

    # Initial params: (n-1) attacks + n defences + home_adv + rho
    n_params = 2 * n + 1
    x0 = np.zeros(n_params)
    x0[2 * n - 1] = 0.25   # home advantage (log-scale ~ 1.28x)
    x0[2 * n]     = -0.1   # rho: small negative (negative goal correlation)

    bounds = [(-3.0, 3.0)] * (2 * n - 1) + [(-1.0, 1.0), (-0.99, 0.99)]

    result = minimize(
        _dc_neg_log_lik,
        x0,
        args=(n, home_idx, away_idx, hgoals, agoals, hxg, axg, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
    )

    params   = result.x
    attacks  = np.concatenate([[0.0], params[:n - 1]])
    defenses = params[n - 1: 2 * n - 1]
    home_adv = float(params[2 * n - 1])
    rho      = float(params[2 * n])

    return {
        "teams":    teams,
        "team_idx": t_idx,
        "attacks":  {t: float(attacks[i])  for i, t in enumerate(teams)},
        "defenses": {t: float(defenses[i]) for i, t in enumerate(teams)},
        "home_adv": home_adv,
        "rho":      rho,
        "converged": result.success,
    }


def predict_dixon_coles(
    home_team: str,
    away_team: str,
    dc_ratings: dict,
    max_goals: int = 8,
) -> dict:
    """Predict scoreline distribution using fitted Dixon-Coles parameters."""
    att = dc_ratings["attacks"]
    dfn = dc_ratings["defenses"]
    h   = dc_ratings["home_adv"]
    rho = dc_ratings["rho"]

    lam_h = max(np.exp(att.get(home_team, 0.0) + dfn.get(away_team, 0.0) + h), 0.05)
    lam_a = max(np.exp(att.get(away_team, 0.0) + dfn.get(home_team, 0.0)),       0.05)

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if   i == 0 and j == 0: p *= max(1.0 - lam_h * lam_a * rho, 1e-10)
            elif i == 0 and j == 1: p *= max(1.0 + lam_h * rho,          1e-10)
            elif i == 1 and j == 0: p *= max(1.0 + lam_a * rho,          1e-10)
            elif i == 1 and j == 1: p *= max(1.0 - rho,                   1e-10)
            matrix[i, j] = p

    return _summarise_matrix(lam_h, lam_a, matrix, max_goals)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost form classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_xgb(df: pd.DataFrame):
    """Train XGBoost multi-class classifier (H/D/A) on rolling form features."""
    df = df.copy()

    feature_cols = [
        "home_roll_gf", "home_roll_ga", "home_roll_pts",
        "away_roll_gf", "away_roll_ga", "away_roll_pts",
    ]
    df["gf_diff"]  = df["home_roll_gf"] - df["away_roll_ga"]
    df["ga_diff"]  = df["away_roll_gf"] - df["home_roll_ga"]
    df["pts_diff"] = df["home_roll_pts"] - df["away_roll_pts"]
    feature_cols  += ["gf_diff", "ga_diff", "pts_diff"]

    if "home_roll_xg" in df.columns and "away_roll_xg" in df.columns:
        df["xg_diff"]  = df["home_roll_xg"]  - df["away_roll_xga"]
        df["xga_diff"] = df["away_roll_xg"]  - df["home_roll_xga"]
        feature_cols  += [
            "home_roll_xg", "home_roll_xga",
            "away_roll_xg", "away_roll_xga",
            "xg_diff", "xga_diff",
        ]

    if "home_days_rest" in df.columns and "away_days_rest" in df.columns:
        df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
        feature_cols   += ["home_days_rest", "away_days_rest", "rest_diff"]

    if "elo_diff" in df.columns:
        feature_cols += ["elo_diff"]

    if "home_venue_gf" in df.columns and "away_venue_gf" in df.columns:
        feature_cols += [
            "home_venue_gf", "home_venue_ga", "home_venue_pts",
            "away_venue_gf", "away_venue_ga", "away_venue_pts",
        ]
        # Team-specific home advantage and road factor
        # home_ha_score: how much better home team performs at home vs their overall average
        #   positive = fortress, zero = same everywhere
        # away_road_score: how much better/worse away team performs on road vs their average
        #   negative = struggles away, near zero = travels well
        df["home_ha_score"]   = df["home_venue_pts"] - df["home_roll_pts"]
        df["away_road_score"] = df["away_venue_pts"] - df["away_roll_pts"]
        # Defensive home/road profiles
        # home_fortress: how many fewer goals home team concedes at home vs average
        #   negative = concedes fewer at home (fortress), positive = leaky at home
        # away_road_def: how many more goals away team concedes on road vs average
        #   positive = porous away, negative = better away defensively (unusual)
        df["home_fortress"]   = df["home_venue_ga"]  - df["home_roll_ga"]
        df["away_road_def"]   = df["away_venue_ga"]  - df["away_roll_ga"]
        feature_cols += [
            "home_ha_score", "away_road_score",
            "home_fortress",  "away_road_def",
        ]

    if "home_draw_rate" in df.columns and "away_draw_rate" in df.columns:
        df["draw_rate_sum"] = df["home_draw_rate"] + df["away_draw_rate"]
        feature_cols += ["home_draw_rate", "away_draw_rate", "draw_rate_sum"]

    train = df.dropna(subset=feature_cols + ["Result"])
    if len(train) < 50:
        return None, feature_cols

    # ── Recency weighting: exponential decay with 180-day half-life ───────
    # Recent matches (this season) get ~4× more weight than a year ago.
    latest_date = pd.to_datetime(train["Date"].max())
    days_ago    = (latest_date - pd.to_datetime(train["Date"])).dt.days
    sample_weight = np.exp(-np.log(2) * days_ago / 180.0)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    )
    model.fit(train[feature_cols].values, train["Result"].values,
              sample_weight=sample_weight.values)
    return model, feature_cols


def predict_xgb(model, feature_cols: list, home_stats: dict, away_stats: dict) -> dict | None:
    """Predict W/D/L probabilities from rolling form stats."""
    if model is None:
        return None

    hgf  = home_stats["avg_gf"];  hga  = home_stats["avg_ga"];  hpts = home_stats["avg_pts"]
    agf  = away_stats["avg_gf"];  aga  = away_stats["avg_ga"];  apts = away_stats["avg_pts"]
    hxg  = home_stats.get("avg_xg",  hgf);  hxga = home_stats.get("avg_xga", hga)
    axg  = away_stats.get("avg_xg",  agf);  axga = away_stats.get("avg_xga", aga)
    h_rest = home_stats.get("days_rest", 7.0)
    a_rest = away_stats.get("days_rest", 7.0)

    h_hvgf  = home_stats.get("home_venue_gf",  hgf)
    h_hvga  = home_stats.get("home_venue_ga",  hga)
    h_hvpts = home_stats.get("home_venue_pts", hpts)
    a_avgf  = away_stats.get("away_venue_gf",  agf)
    a_avga  = away_stats.get("away_venue_ga",  aga)
    a_avpts = away_stats.get("away_venue_pts", apts)
    h_draw  = home_stats.get("draw_rate", 0.27)
    a_draw  = away_stats.get("draw_rate", 0.27)
    h_elo   = home_stats.get("elo", 1500.0)
    a_elo   = away_stats.get("elo", 1500.0)

    # Team-specific home/road advantage scores
    # home_stats has both venue perspectives: home_venue_pts (their home pts) and away_venue_pts (their away pts)
    h_away_venue_pts = home_stats.get("away_venue_pts", hpts)
    h_away_venue_ga  = home_stats.get("away_venue_ga",  hga)
    a_home_venue_pts = away_stats.get("home_venue_pts", apts)
    a_home_venue_ga  = away_stats.get("home_venue_ga",  aga)

    feat_map = {
        "home_roll_gf":    hgf,
        "home_roll_ga":    hga,
        "home_roll_pts":   hpts,
        "away_roll_gf":    agf,
        "away_roll_ga":    aga,
        "away_roll_pts":   apts,
        "gf_diff":         hgf  - aga,
        "ga_diff":         agf  - hga,
        "pts_diff":        hpts - apts,
        "home_roll_xg":    hxg,
        "home_roll_xga":   hxga,
        "away_roll_xg":    axg,
        "away_roll_xga":   axga,
        "xg_diff":         hxg  - axga,
        "xga_diff":        axg  - hxga,
        "home_days_rest":  h_rest,
        "away_days_rest":  a_rest,
        "rest_diff":       h_rest - a_rest,
        "elo_diff":        h_elo  - a_elo,
        "home_venue_gf":   h_hvgf,
        "home_venue_ga":   h_hvga,
        "home_venue_pts":  h_hvpts,
        "away_venue_gf":   a_avgf,
        "away_venue_ga":   a_avga,
        "away_venue_pts":  a_avpts,
        "home_draw_rate":  h_draw,
        "away_draw_rate":  a_draw,
        "draw_rate_sum":   h_draw + a_draw,
        # Team-specific home advantage and road factor
        "home_ha_score":   h_hvpts - hpts,          # how much better home team is at home vs average
        "away_road_score": a_avpts - apts,           # how much better/worse away team is on road vs average
        "home_fortress":   h_hvga  - hga,            # fewer goals conceded at home vs average (neg = fortress)
        "away_road_def":   a_avga  - aga,            # more goals conceded away vs average (pos = leaky away)
    }

    X = np.array([[feat_map[f] for f in feature_cols]])
    probs = model.predict_proba(X)[0]  # [P(H), P(D), P(A)]
    return {"home_win": float(probs[0]), "draw": float(probs[1]), "away_win": float(probs[2])}


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble blending
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(hw: float, d: float, aw: float) -> dict:
    total = hw + d + aw
    return {"home_win": hw / total, "draw": d / total, "away_win": aw / total}


def blend(poisson_pred: dict, xgb_pred: dict | None, alpha: float = 0.80) -> dict:
    """Blend Poisson (alpha) + XGBoost (1-alpha)."""
    if xgb_pred is None:
        return {k: poisson_pred[k] for k in ("home_win", "draw", "away_win")}
    hw = alpha * poisson_pred["home_win"] + (1 - alpha) * xgb_pred["home_win"]
    d  = alpha * poisson_pred["draw"]     + (1 - alpha) * xgb_pred["draw"]
    aw = alpha * poisson_pred["away_win"] + (1 - alpha) * xgb_pred["away_win"]
    return _normalise(hw, d, aw)


def blend_dc(dc_pred: dict, xgb_pred: dict | None, alpha: float = 0.85) -> dict:
    """Blend Dixon-Coles (alpha) + XGBoost (1-alpha). DC is primary."""
    if xgb_pred is None:
        return {k: dc_pred[k] for k in ("home_win", "draw", "away_win")}
    hw = alpha * dc_pred["home_win"] + (1 - alpha) * xgb_pred["home_win"]
    d  = alpha * dc_pred["draw"]     + (1 - alpha) * xgb_pred["draw"]
    aw = alpha * dc_pred["away_win"] + (1 - alpha) * xgb_pred["away_win"]
    return _normalise(hw, d, aw)


# ─────────────────────────────────────────────────────────────────────────────
# Draw specialist models
# ─────────────────────────────────────────────────────────────────────────────

def compute_draw_dc_ratings(df: pd.DataFrame, decay_weeks: float = 14.0,
                            draw_boost: float = 3.0) -> dict:
    """
    Dixon-Coles variant with draw-weighted training samples.
    Upweights draw matches so the MLE optimises harder for draw accuracy.
    Uses tighter rho bounds for better low-score draw calibration.
    """
    max_date = df["Date"].max()
    df = df.copy()
    df["days_ago"] = (max_date - df["Date"]).dt.days
    df["w"] = np.exp(-df["days_ago"] / (decay_weeks * 7))
    # Upweight draws
    df["w_draw"] = df["w"] * np.where(df["FTR"] == "D", draw_boost, 1.0)

    teams   = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n       = len(teams)
    t_idx   = {t: i for i, t in enumerate(teams)}

    home_idx = df["HomeTeam"].map(t_idx).values.astype(int)
    away_idx = df["AwayTeam"].map(t_idx).values.astype(int)
    hgoals   = df["FTHG"].values.astype(int)
    agoals   = df["FTAG"].values.astype(int)
    hxg = df["xg_h"].values if "xg_h" in df.columns else hgoals.astype(float)
    axg = df["xg_a"].values if "xg_a" in df.columns else agoals.astype(float)
    weights  = df["w_draw"].values

    n_params = 2 * n + 1
    x0 = np.zeros(n_params)
    x0[2 * n - 1] = 0.25
    x0[2 * n]     = -0.15  # start rho slightly more negative (draws favour low scores)

    # Tighter rho bounds for draw calibration
    bounds = [(-3.0, 3.0)] * (2 * n - 1) + [(-1.0, 1.0), (-0.5, 0.5)]

    result = minimize(
        _dc_neg_log_lik,
        x0,
        args=(n, home_idx, away_idx, hgoals, agoals, hxg, axg, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
    )

    params   = result.x
    attacks  = np.concatenate([[0.0], params[:n - 1]])
    defenses = params[n - 1: 2 * n - 1]

    return {
        "teams":    teams,
        "team_idx": t_idx,
        "attacks":  {t: float(attacks[i])  for i, t in enumerate(teams)},
        "defenses": {t: float(defenses[i]) for i, t in enumerate(teams)},
        "home_adv": float(params[2 * n - 1]),
        "rho":      float(params[2 * n]),
        "converged": result.success,
    }


def train_draw_xgb(df: pd.DataFrame):
    """
    Binary XGBoost classifier: draw vs not-draw.
    Uses all standard features plus draw-specific ones.
    """
    df = df.copy()

    # Standard features (same as main XGB)
    feature_cols = [
        "home_roll_gf", "home_roll_ga", "home_roll_pts",
        "away_roll_gf", "away_roll_ga", "away_roll_pts",
    ]
    df["gf_diff"]  = df["home_roll_gf"] - df["away_roll_ga"]
    df["ga_diff"]  = df["away_roll_gf"] - df["home_roll_ga"]
    df["pts_diff"] = df["home_roll_pts"] - df["away_roll_pts"]
    feature_cols += ["gf_diff", "ga_diff", "pts_diff"]

    if "home_roll_xg" in df.columns and "away_roll_xg" in df.columns:
        df["xg_diff"]  = df["home_roll_xg"]  - df["away_roll_xga"]
        df["xga_diff"] = df["away_roll_xg"]  - df["home_roll_xga"]
        feature_cols += [
            "home_roll_xg", "home_roll_xga",
            "away_roll_xg", "away_roll_xga",
            "xg_diff", "xga_diff",
        ]

    if "home_days_rest" in df.columns:
        df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
        feature_cols += ["home_days_rest", "away_days_rest", "rest_diff"]

    if "elo_diff" in df.columns:
        feature_cols += ["elo_diff"]

    if "home_venue_gf" in df.columns:
        feature_cols += [
            "home_venue_gf", "home_venue_ga", "home_venue_pts",
            "away_venue_gf", "away_venue_ga", "away_venue_pts",
        ]
        df["home_ha_score"]   = df["home_venue_pts"] - df["home_roll_pts"]
        df["away_road_score"] = df["away_venue_pts"] - df["away_roll_pts"]
        df["home_fortress"]   = df["home_venue_ga"]  - df["home_roll_ga"]
        df["away_road_def"]   = df["away_venue_ga"]  - df["away_roll_ga"]
        feature_cols += ["home_ha_score", "away_road_score", "home_fortress", "away_road_def"]

    if "home_draw_rate" in df.columns:
        df["draw_rate_sum"] = df["home_draw_rate"] + df["away_draw_rate"]
        feature_cols += ["home_draw_rate", "away_draw_rate", "draw_rate_sum"]

    # ── Draw-specific features ────────────────────────────────────────────
    if "xg_convergence" in df.columns:
        feature_cols += ["xg_convergence"]
    if "elo_diff_abs" in df.columns:
        feature_cols += ["elo_diff_abs"]
    if "both_draw_prone" in df.columns:
        feature_cols += ["both_draw_prone"]
    if "home_venue_draw_rate" in df.columns:
        feature_cols += ["home_venue_draw_rate", "away_venue_draw_rate"]

    # Binary target: draw = 1, not draw = 0
    df["is_draw"] = (df["FTR"] == "D").astype(int)

    train = df.dropna(subset=feature_cols + ["is_draw"])
    if len(train) < 50:
        return None, feature_cols

    latest_date = pd.to_datetime(train["Date"].max())
    days_ago    = (latest_date - pd.to_datetime(train["Date"])).dt.days
    sample_weight = np.exp(-np.log(2) * days_ago / 180.0)

    # ~27% draw rate → scale_pos_weight ≈ 2.7
    draw_rate = train["is_draw"].mean()
    spw = (1 - draw_rate) / max(draw_rate, 0.01)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,
        eval_metric="logloss",
        objective="binary:logistic",
        verbosity=0,
        random_state=42,
    )
    model.fit(train[feature_cols].values, train["is_draw"].values,
              sample_weight=sample_weight.values)
    return model, feature_cols


def predict_draw_xgb(model, feature_cols: list, home_stats: dict,
                     away_stats: dict) -> float:
    """Predict P(draw) from binary draw XGB. Returns a single float."""
    if model is None:
        return 0.27  # league average fallback

    hgf  = home_stats["avg_gf"];  hga  = home_stats["avg_ga"];  hpts = home_stats["avg_pts"]
    agf  = away_stats["avg_gf"];  aga  = away_stats["avg_ga"];  apts = away_stats["avg_pts"]
    hxg  = home_stats.get("avg_xg",  hgf);  hxga = home_stats.get("avg_xga", hga)
    axg  = away_stats.get("avg_xg",  agf);  axga = away_stats.get("avg_xga", aga)
    h_rest = home_stats.get("days_rest", 7.0)
    a_rest = away_stats.get("days_rest", 7.0)
    h_draw = home_stats.get("draw_rate", 0.27)
    a_draw = away_stats.get("draw_rate", 0.27)
    h_elo  = home_stats.get("elo", 1500.0)
    a_elo  = away_stats.get("elo", 1500.0)

    feat_map = {
        "home_roll_gf": hgf, "home_roll_ga": hga, "home_roll_pts": hpts,
        "away_roll_gf": agf, "away_roll_ga": aga, "away_roll_pts": apts,
        "gf_diff": hgf - aga, "ga_diff": agf - hga, "pts_diff": hpts - apts,
        "home_roll_xg": hxg, "home_roll_xga": hxga,
        "away_roll_xg": axg, "away_roll_xga": axga,
        "xg_diff": hxg - axga, "xga_diff": axg - hxga,
        "home_days_rest": h_rest, "away_days_rest": a_rest, "rest_diff": h_rest - a_rest,
        "elo_diff": h_elo - a_elo,
        "home_venue_gf": home_stats.get("home_venue_gf", hgf),
        "home_venue_ga": home_stats.get("home_venue_ga", hga),
        "home_venue_pts": home_stats.get("home_venue_pts", hpts),
        "away_venue_gf": away_stats.get("away_venue_gf", agf),
        "away_venue_ga": away_stats.get("away_venue_ga", aga),
        "away_venue_pts": away_stats.get("away_venue_pts", apts),
        "home_ha_score": home_stats.get("home_venue_pts", hpts) - hpts,
        "away_road_score": away_stats.get("away_venue_pts", apts) - apts,
        "home_fortress": home_stats.get("home_venue_ga", hga) - hga,
        "away_road_def": away_stats.get("away_venue_ga", aga) - aga,
        "home_draw_rate": h_draw, "away_draw_rate": a_draw,
        "draw_rate_sum": h_draw + a_draw,
        # Draw-specific
        "xg_convergence": abs(hxg - axg),
        "elo_diff_abs": abs(h_elo - a_elo),
        "both_draw_prone": h_draw * a_draw,
        "home_venue_draw_rate": home_stats.get("home_venue_draw_rate", 0.27),
        "away_venue_draw_rate": away_stats.get("away_venue_draw_rate", 0.27),
    }

    X = np.array([[feat_map.get(f, 0.0) for f in feature_cols]])
    prob = model.predict_proba(X)[0]
    # Binary classifier: prob[1] = P(draw)
    return float(prob[1]) if len(prob) > 1 else float(prob[0])


def blend_draw_specialist(
    main_result: dict,
    draw_dc_ratings: dict,
    draw_xgb_prob: float,
    home_team: str,
    away_team: str,
    specialist_weight: float = 0.35,
) -> dict:
    """
    Blend the draw specialist's probability into the main model output.
    The specialist's draw estimate replaces a portion of the main model's draw.
    H/A probabilities are re-normalised proportionally.
    """
    # Get draw prob from draw-weighted DC
    draw_dc_pred = predict_dixon_coles(home_team, away_team, draw_dc_ratings)
    draw_dc_prob = draw_dc_pred["draw"]

    # Combine the two draw specialists (50/50 DC draw + XGB draw)
    draw_specialist = 0.5 * draw_dc_prob + 0.5 * draw_xgb_prob

    # Blend with main model
    main_draw = main_result["draw"]
    final_draw = (1.0 - specialist_weight) * main_draw + specialist_weight * draw_specialist

    # Re-normalise H/A proportionally
    old_non_draw = main_result["home_win"] + main_result["away_win"]
    remaining = max(1.0 - final_draw, 0.01)
    if old_non_draw > 0:
        scale = remaining / old_non_draw
        final_home = main_result["home_win"] * scale
        final_away = main_result["away_win"] * scale
    else:
        final_home = remaining / 2
        final_away = remaining / 2

    return _normalise(final_home, final_draw, final_away)


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_table(df: pd.DataFrame) -> dict[str, int]:
    """
    Compute league table positions from match data.
    Returns {team: position} where 1 = top of table.
    Tiebreakers: points → goal difference → goals scored.
    """
    rows = []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        ftr = r["FTR"]
        rows.append({"team": h, "pts": 3 if ftr=="H" else (1 if ftr=="D" else 0),
                     "gf": hg, "ga": ag})
        rows.append({"team": a, "pts": 3 if ftr=="A" else (1 if ftr=="D" else 0),
                     "gf": ag, "ga": hg})

    tbl = (
        pd.DataFrame(rows)
        .groupby("team")
        .agg(pts=("pts","sum"), gf=("gf","sum"), ga=("ga","sum"))
        .assign(gd=lambda x: x["gf"] - x["ga"])
        .sort_values(["pts","gd","gf"], ascending=False)
        .reset_index()
    )
    return {row["team"]: pos + 1 for pos, row in tbl.iterrows()}


def backtest_models(
    df: pd.DataFrame,
    df_features: pd.DataFrame,
    test_weeks: int = 10,
) -> pd.DataFrame:
    """
    Train-test split backtest.
    Trains each model on data up to `test_weeks` ago, tests on the remainder.
    Includes a table-position baseline: always predict the higher-placed team wins.
    Returns DataFrame with match-level predictions vs actuals.
    """
    from data import get_current_stats, get_current_elo  # local import to avoid circular dep

    max_date = df["Date"].max()
    cutoff   = max_date - pd.Timedelta(weeks=test_weeks)

    train_df  = df[df["Date"] <= cutoff].copy()
    train_ft  = df_features[df_features["Date"] <= cutoff].copy()
    test_df   = df[df["Date"] > cutoff].copy()

    if len(train_df) < 100 or len(test_df) == 0:
        return pd.DataFrame()

    # Train all models once on the training window
    poisson_r = compute_poisson_ratings(train_df)
    dc_r      = compute_dixon_coles_ratings(train_df)
    dc_draw_r = compute_draw_dc_ratings(train_df)
    xgb_m, feat_cols = train_xgb(train_ft)
    draw_xgb_m, draw_fc = train_draw_xgb(train_ft)
    elo_dict  = get_current_elo(train_df)

    # Pre-compute rolling table snapshot for each unique test match date.
    # For a match on date D we use only that season's completed matches with
    # Date < D — no look-ahead and no contamination from previous seasons
    # (which would inflate the team count beyond 20).
    test_dates = sorted(test_df["Date"].unique())
    table_by_date: dict = {}
    # Build a date→season lookup from the test set
    date_to_season = test_df.drop_duplicates("Date").set_index("Date")["Season"].to_dict()
    for match_date in test_dates:
        season = date_to_season.get(match_date)
        prior = df[(df["Date"] < match_date) & (df["Season"] == season)]
        table_by_date[match_date] = _compute_table(prior) if len(prior) >= 1 else {}

    def _label(d: dict) -> str:
        k = max(d, key=d.get)
        return {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}[k]

    records = []
    for _, match in test_df.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        actual = {"H": "Home Win", "D": "Draw", "A": "Away Win"}.get(match["FTR"], "?")

        hs = get_current_stats(train_df, home, elo_dict=elo_dict)
        as_ = get_current_stats(train_df, away, elo_dict=elo_dict)
        xgb_p = predict_xgb(xgb_m, feat_cols, hs, as_)

        p_pred   = predict_poisson(home, away, poisson_r)
        dc_pred  = predict_dixon_coles(home, away, dc_r)
        p_blend  = blend(p_pred, xgb_p)
        dc_blend = blend_dc(dc_pred, xgb_p)

        # Draw specialist blend
        draw_xgb_prob = predict_draw_xgb(draw_xgb_m, draw_fc, hs, as_)
        final = blend_draw_specialist(dc_blend, dc_draw_r, draw_xgb_prob, home, away)

        # Table baseline: use table as it stood the morning of the match (no look-ahead)
        table    = table_by_date.get(match["Date"], {})
        home_pos = table.get(home, 999)
        away_pos = table.get(away, 999)
        if home_pos <= away_pos:
            table_pred = "Home Win"
        else:
            table_pred = "Away Win"

        records.append({
            "Date":     match["Date"],
            "Home":     home,
            "Away":     away,
            "HomePos":  home_pos,
            "AwayPos":  away_pos,
            "Score":    f"{int(match['FTHG'])}–{int(match['FTAG'])}",
            "Actual":   actual,
            # Table baseline
            "Table_Pred":    table_pred,
            "Table_Correct": table_pred == actual,
            # Poisson + XGB
            "Pois_H":   round(p_blend["home_win"] * 100, 1),
            "Pois_D":   round(p_blend["draw"]     * 100, 1),
            "Pois_A":   round(p_blend["away_win"] * 100, 1),
            "Pois_Pred": _label(p_blend),
            "Pois_Correct": _label(p_blend) == actual,
            # DC + XGB + Draw Specialist
            "DC_H":     round(final["home_win"] * 100, 1),
            "DC_D":     round(final["draw"]     * 100, 1),
            "DC_A":     round(final["away_win"] * 100, 1),
            "DC_O25":   round(dc_pred.get("over_25", 0.5) * 100, 1),
            "DC_Pred":  _label(final),
            "DC_Correct": _label(final) == actual,
            # Probabilities for Brier score
            "_p_h": p_blend["home_win"], "_p_d": p_blend["draw"], "_p_a": p_blend["away_win"],
            "_dc_h": final["home_win"], "_dc_d": final["draw"], "_dc_a": final["away_win"],
            "_act_h": 1.0 if actual == "Home Win" else 0.0,
            "_act_d": 1.0 if actual == "Draw"     else 0.0,
            "_act_a": 1.0 if actual == "Away Win" else 0.0,
        })

    return pd.DataFrame(records)


def compute_backtest_summary(bt: pd.DataFrame) -> dict:
    """Compute accuracy, Brier score, and log-loss from backtest DataFrame."""
    if bt.empty:
        return {}

    def brier(prefix):
        return float(np.mean(
            (bt[f"_{prefix}_h"] - bt["_act_h"]) ** 2 +
            (bt[f"_{prefix}_d"] - bt["_act_d"]) ** 2 +
            (bt[f"_{prefix}_a"] - bt["_act_a"]) ** 2
        ) / 2)  # normalise to [0,1]

    def log_loss(prefix):
        eps = 1e-10
        ph = np.clip(bt[f"_{prefix}_h"].values, eps, 1 - eps)
        pd_ = np.clip(bt[f"_{prefix}_d"].values, eps, 1 - eps)
        pa = np.clip(bt[f"_{prefix}_a"].values, eps, 1 - eps)
        return float(-np.mean(
            bt["_act_h"].values * np.log(ph) +
            bt["_act_d"].values * np.log(pd_) +
            bt["_act_a"].values * np.log(pa)
        ))

    # Random (1/3 each) baseline on the same test data
    rand_brier = float(np.mean(
        (1/3 - bt["_act_h"]) ** 2 + (1/3 - bt["_act_d"]) ** 2 + (1/3 - bt["_act_a"]) ** 2
    ) / 2)

    return {
        "n_matches":       len(bt),
        "pois_accuracy":   round(bt["Pois_Correct"].mean()  * 100, 1),
        "dc_accuracy":     round(bt["DC_Correct"].mean()    * 100, 1),
        "table_accuracy":  round(bt["Table_Correct"].mean() * 100, 1),
        "pois_brier":      round(brier("p"),  4),
        "dc_brier":        round(brier("dc"), 4),
        "pois_logloss":    round(log_loss("p"),  4),
        "dc_logloss":      round(log_loss("dc"), 4),
        "rand_brier":      round(rand_brier, 4),
        "rand_accuracy":   33.3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Season simulator (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_season(
    remaining_fixtures: list[dict],
    current_table: pd.DataFrame,
    dc_ratings: dict,
    n_sims: int = 10_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of the remaining Premier League season.

    Parameters
    ----------
    remaining_fixtures : list of {"home": str, "away": str}
    current_table     : DataFrame with columns [Team, Pts, GF, GA, GD, Played]
    dc_ratings        : fitted Dixon-Coles ratings dict
    n_sims            : number of simulations

    Returns
    -------
    DataFrame indexed by Team with columns:
        mean_pts, mean_pos, p_title, p_top4, p_top6, p_relegated
        and pos_1 … pos_20 (probability of finishing in each position)
    """
    rng = np.random.default_rng(rng_seed)
    teams = list(current_table["Team"])
    n_teams = len(teams)
    t_idx = {t: i for i, t in enumerate(teams)}

    # Pre-compute H/D/A probabilities for every remaining fixture
    fixture_probs = []
    for fix in remaining_fixtures:
        h, a = fix["home"], fix["away"]
        if h not in t_idx or a not in t_idx:
            continue
        pred = predict_dixon_coles(h, a, dc_ratings)
        fixture_probs.append((
            t_idx[h], t_idx[a],
            pred["home_win"], pred["draw"], pred["away_win"],
        ))

    # Starting points, GF, GA for each team
    base_pts = current_table.set_index("Team")["Pts"].reindex(teams).fillna(0).values.astype(float)
    base_gf  = current_table.set_index("Team")["GF"].reindex(teams).fillna(0).values.astype(float)
    base_ga  = current_table.set_index("Team")["GA"].reindex(teams).fillna(0).values.astype(float)

    # Vectorised simulation: shape (n_sims, n_teams)
    sim_pts = np.tile(base_pts, (n_sims, 1))
    sim_gf  = np.tile(base_gf,  (n_sims, 1))
    sim_ga  = np.tile(base_ga,  (n_sims, 1))

    for hi, ai, ph, pd_, pa in fixture_probs:
        outcomes = rng.choice(3, size=n_sims, p=[ph, pd_, pa])
        # 0 = home win, 1 = draw, 2 = away win
        home_win = outcomes == 0
        draw     = outcomes == 1
        away_win = outcomes == 2

        sim_pts[:, hi] += home_win * 3 + draw * 1
        sim_pts[:, ai] += away_win * 3 + draw * 1

        # Approximate scorelines for GD tiebreaker (Poisson-based)
        lam_h = max(np.exp(
            dc_ratings["attacks"].get(teams[hi], 0.0) +
            dc_ratings["defenses"].get(teams[ai], 0.0) +
            dc_ratings["home_adv"]
        ), 0.1)
        lam_a = max(np.exp(
            dc_ratings["attacks"].get(teams[ai], 0.0) +
            dc_ratings["defenses"].get(teams[hi], 0.0)
        ), 0.1)
        hg = rng.poisson(lam_h, n_sims).astype(float)
        ag = rng.poisson(lam_a, n_sims).astype(float)

        sim_gf[:, hi] += hg;  sim_ga[:, hi] += ag
        sim_gf[:, ai] += ag;  sim_ga[:, ai] += hg

    sim_gd = sim_gf - sim_ga

    # Rank teams in each simulation (lower rank = better)
    # Primary: pts desc, secondary: gd desc, tertiary: gf desc
    # Build sort key: negate so argmax gives rank
    sort_key = sim_pts * 1e8 + sim_gd * 1e4 + sim_gf
    # argsort descending → positions
    ranks = np.argsort(-sort_key, axis=1)           # shape (n_sims, n_teams): index of team at each position
    positions = np.argsort(ranks, axis=1) + 1        # shape (n_sims, n_teams): position of each team

    rows = []
    for i, team in enumerate(teams):
        team_pos = positions[:, i]
        team_pts = sim_pts[:, i]
        pos_dist = np.bincount(team_pos, minlength=n_teams + 1)[1:] / n_sims
        rows.append({
            "Team":         team,
            "mean_pts":     float(np.mean(team_pts)),
            "mean_pos":     float(np.mean(team_pos)),
            "p_title":      float((team_pos == 1).mean()),
            "p_top4":       float((team_pos <= 4).mean()),
            "p_top6":       float((team_pos <= 6).mean()),
            "p_relegated":  float((team_pos >= 18).mean()),
            **{f"pos_{p+1}": float(pos_dist[p]) for p in range(n_teams)},
        })

    result = pd.DataFrame(rows).sort_values("mean_pos").reset_index(drop=True)
    return result
