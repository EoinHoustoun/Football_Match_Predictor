"""
Mock betting portfolio for F_PRED — EV-based paper trading.
No real money involved. For analysis and model validation only.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
_LIVE_ODDS_CACHE = DATA_DIR / "live_odds_cache.json"
_API_USAGE_FILE  = DATA_DIR / "odds_api_usage.json"
_ODDS_SNAPSHOT_DIR = DATA_DIR / "odds_snapshots"


def resolve_odds_api_key(settings_key: str = "") -> str:
    """The Odds API key, preferring the ODDS_API_KEY environment variable over
    the plaintext settings value so the key can be kept out of portfolio.json.
    """
    return os.environ.get("ODDS_API_KEY", "").strip() or (settings_key or "").strip()

# ── API usage tracking (free tier = 500 req/month) ───────────────────────────
_API_MONTHLY_CAP = 400   # conservative cap — leaves 100 buffer
_CACHE_HOURS     = 6     # how long to cache before re-fetching


def _load_api_usage() -> dict:
    """Load API usage counter: {"month": "2026-03", "count": 12}"""
    if _API_USAGE_FILE.exists():
        try:
            return json.loads(_API_USAGE_FILE.read_text())
        except Exception:
            pass
    return {"month": "", "count": 0}


def _save_api_usage(usage: dict):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        _API_USAGE_FILE.write_text(json.dumps(usage))
    except Exception:
        pass


def _record_api_call() -> bool:
    """Record an API call. Returns True if under cap, False if blocked."""
    usage = _load_api_usage()
    current_month = datetime.now().strftime("%Y-%m")
    if usage.get("month") != current_month:
        usage = {"month": current_month, "count": 0}
    if usage["count"] >= _API_MONTHLY_CAP:
        return False
    usage["count"] += 1
    _save_api_usage(usage)
    return True


def get_api_usage() -> dict:
    """Return current usage stats for display in the UI."""
    usage = _load_api_usage()
    current_month = datetime.now().strftime("%Y-%m")
    if usage.get("month") != current_month:
        return {"month": current_month, "count": 0, "cap": _API_MONTHLY_CAP, "remaining": _API_MONTHLY_CAP}
    return {
        "month": current_month,
        "count": usage["count"],
        "cap": _API_MONTHLY_CAP,
        "remaining": max(0, _API_MONTHLY_CAP - usage["count"]),
    }

# Markets where the model has demonstrated edge in backtesting
# Draw (+76% ROI), Over 2.5 (+14% ROI), Under 2.5 (+14% ROI)
# Home/Away win bets lose money (-16% to -20% ROI)
PROFITABLE_MARKETS = {"D", "over25", "under25"}

# Defaults = the 2026-27 validated config (2026-06-09 session): the deployed
# v158 core plus two tightening filters that improved the worst season from
# £1k to £10.9k across the 4-season honest validation (ELO floor 1500, EV gate
# 40%) — see data/diagnostics/draw_cap_validation_*.json + sweep_100k_findings.md.
_DEFAULT: dict = {
    "initial_bankroll": 1000.0,
    "bankroll": 1000.0,
    "bets": [],
    "settings": {
        "min_ev": 0.40,
        "max_stake_pct": 0.25,
        "kelly_fraction": 1.0,
        "odds_api_key": "",
        "auto_bet_enabled": False,
        "auto_bet_threshold": 0.40,   # 40% EV gate (4-season validated)
        "auto_markets": ["D", "under25"],
        "min_prob":            0.21,  # reject longshots even if +EV — variance kills them
        "use_calibrated_probs": True, # isotonic calibration (draw capped at 0.45)
        "use_simultaneous_kelly": True,
        "skip_late_season": True,     # Mar-Apr 0/7 across 2 seasons
        "skip_home_title_race": False,
        "market_gates": {"under25": {"min_prob": 0.50, "min_ev": 0.05}},
        "main_banned_dows": ["Mon"],
        "main_min_team_elo": 1500,    # worst-season crash protection
        "min_team_matches": 6,        # promoted sides aren't rated yet
        "max_bets_per_club": 5,       # no club carries a third+ of the season
    },
}


def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            data = json.loads(PORTFOLIO_FILE.read_text())
            # Back-fill missing settings keys
            data.setdefault("settings", {})
            for k, v in _DEFAULT["settings"].items():
                data["settings"].setdefault(k, v)
            return data
        except Exception:
            pass
    return {
        "initial_bankroll": _DEFAULT["initial_bankroll"],
        "bankroll": _DEFAULT["bankroll"],
        "bets": [],
        "settings": dict(_DEFAULT["settings"]),
    }


def save_portfolio(p: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps(p, indent=2, default=str))


# ── Core maths ─────────────────────────────────────────────────────────────────

def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """EV as a fraction: 0.10 = +10%."""
    return model_prob * decimal_odds - 1.0


# ── Isotonic probability calibration ─────────────────────────────────────────

def fit_calibrators_from_backtest(bt_df: pd.DataFrame) -> dict:
    """Fit per-market IsotonicRegression calibrators from backtest predictions.

    Backtest DataFrame must contain columns: _dc_h/_dc_d/_dc_a (predicted probs
    from DC+XGB) and _act_h/_act_d/_act_a (0/1 actual outcomes).

    Returns a dict {market_code: fitted_calibrator}. Missing or empty data ->
    empty dict (callers treat that as identity calibration).
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return {}

    if bt_df is None or bt_df.empty:
        return {}

    calibrators: dict = {}
    for market, pred_col, act_col in [
        ("H", "_dc_h", "_act_h"),
        ("D", "_dc_d", "_act_d"),
        ("A", "_dc_a", "_act_a"),
    ]:
        if pred_col not in bt_df.columns or act_col not in bt_df.columns:
            continue
        preds = bt_df[pred_col].values.astype(float)
        acts  = bt_df[act_col].values.astype(float)
        if len(preds) < 20:    # too few samples for a stable fit
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
        try:
            iso.fit(preds, acts)
            calibrators[market] = iso
        except Exception:
            continue
    return calibrators


# Safety cap on the *calibrated* draw probability. Isotonic fits on small
# windows can map a band of raw probs onto absurd plateaus — observed
# 2026-06-09: a 91-match pre-season window sent every raw draw prob ≥0.40 to
# 0.667, which then sized near-max Kelly stakes off a fantasy edge (see
# data/diagnostics/sweep_100k_findings.md). No EPL fixture has a true draw
# probability anywhere near 2/3; ~0.45 is already beyond anything plausible.
# Module-level so research scripts can A/B it (set to None to disable).
DRAW_PROB_CAP: float | None = 0.45


def calibrate_prob(raw_prob: float, market: str, calibrators: dict | None) -> float:
    """Transform a raw model probability into its calibrated equivalent.

    Returns raw_prob unchanged if no calibrator exists for this market — so
    callers can always use this safely without pre-checking. Calibrated draw
    probabilities are clamped to DRAW_PROB_CAP (small-sample isotonic guard).
    """
    if not calibrators or market not in calibrators:
        return float(raw_prob)
    try:
        p = float(calibrators[market].predict([float(raw_prob)])[0])
    except Exception:
        return float(raw_prob)
    if market == "D" and DRAW_PROB_CAP is not None:
        p = min(p, DRAW_PROB_CAP)
    return p


def kelly_stake_amount(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = 0.5,
    max_pct: float = 0.10,
) -> float:
    """Fractional Kelly stake capped at max_pct of bankroll."""
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    if b <= 0 or model_prob <= 0:
        return 0.0
    full_kelly = max(0.0, (model_prob * b - q) / b)
    capped = min(full_kelly * fraction, max_pct)
    return round(bankroll * capped, 2)


def implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to raw implied probability (includes bookmaker margin)."""
    return 1.0 / decimal_odds if decimal_odds > 1 else 0.0


# ── Bet CRUD ───────────────────────────────────────────────────────────────────

def place_bet(
    p: dict,
    home: str,
    away: str,
    match_date: str,
    market: str,       # "H", "D", "A", "over25", "under25"
    selection: str,    # human-readable
    model_prob: float,
    decimal_odds: float,
    stake: float,
) -> dict:
    ev_val = compute_ev(model_prob, decimal_odds)
    bet = {
        "id": uuid.uuid4().hex[:8],
        "home": home,
        "away": away,
        "date": match_date,
        "market": market,
        "selection": selection,
        "model_prob": round(model_prob, 4),
        "odds": round(decimal_odds, 2),
        "ev": round(ev_val, 4),
        "stake": round(stake, 2),
        "status": "pending",
        "profit": None,
        "placed_at": datetime.now().isoformat(),
        "settled_at": None,
    }
    p["bankroll"] = round(p["bankroll"] - stake, 2)
    p["bets"].append(bet)
    return bet


def remove_pending_bet(p: dict, bet_id: str) -> bool:
    """Cancel a pending bet and refund the stake."""
    for i, bet in enumerate(p["bets"]):
        if bet["id"] == bet_id and bet["status"] == "pending":
            p["bankroll"] = round(p["bankroll"] + bet["stake"], 2)
            p["bets"].pop(i)
            return True
    return False


# ── Auto-settlement ─────────────────────────────────────────────────────────────

def _check_result(home: str, away: str, date_str: str, market: str,
                  df: pd.DataFrame) -> bool | None:
    """Return True/False if result found in df, None if not yet available."""
    try:
        md = pd.to_datetime(date_str).normalize()
    except Exception:
        return None
    rows = df[
        (df["HomeTeam"] == home) & (df["AwayTeam"] == away) & (df["Date"] == md)
    ]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    ftr   = str(row["FTR"])
    total = int(row["FTHG"]) + int(row["FTAG"])
    return {
        "H":       ftr == "H",
        "D":       ftr == "D",
        "A":       ftr == "A",
        "over25":  total > 2,
        "under25": total <= 2,
    }.get(market, False)


def auto_settle(p: dict, df: pd.DataFrame) -> int:
    """
    Settle all pending bets (singles + accas) against completed results in df.
    Returns count of newly settled bets.
    """
    count = 0
    for bet in p["bets"]:
        if bet["status"] != "pending":
            continue

        if bet.get("type") == "acca":
            # All legs must be settled; any None means still pending
            leg_results = [
                _check_result(lg["home"], lg["away"], lg["date"], lg["market"], df)
                for lg in bet["legs"]
            ]
            if None in leg_results:
                continue
            all_won = all(leg_results)
            if all_won:
                ret    = round(bet["stake"] * bet["combined_odds"], 2)
                profit = round(ret - bet["stake"], 2)
                p["bankroll"] = round(p["bankroll"] + ret, 2)
            else:
                profit = round(-bet["stake"], 2)
        else:
            # Single bet
            result = _check_result(bet["home"], bet["away"], bet["date"],
                                   bet["market"], df)
            if result is None:
                continue
            all_won = result
            if all_won:
                ret    = round(bet["stake"] * bet["odds"], 2)
                profit = round(ret - bet["stake"], 2)
                p["bankroll"] = round(p["bankroll"] + ret, 2)
            else:
                profit = round(-bet["stake"], 2)

        bet["status"]      = "won" if all_won else "lost"
        bet["profit"]      = profit
        bet["settled_at"]  = datetime.now().isoformat()
        count += 1
    return count


# ── Accumulators ───────────────────────────────────────────────────────────────

def place_acca(p: dict, legs: list[dict], stake: float) -> dict:
    """
    Place an accumulator bet. Legs: list of {home, away, date, market, selection,
    model_prob, odds}. Combined odds = product of all leg odds.
    """
    combined_prob  = 1.0
    combined_odds  = 1.0
    for lg in legs:
        combined_prob  *= lg["model_prob"]
        combined_odds  *= lg["odds"]
    combined_odds = round(combined_odds, 2)
    ev_val        = compute_ev(combined_prob, combined_odds)

    bet = {
        "id":                   uuid.uuid4().hex[:8],
        "type":                 "acca",
        "legs":                 legs,
        "combined_model_prob":  round(combined_prob, 4),
        "combined_odds":        combined_odds,
        "odds":                 combined_odds,   # alias for stats compatibility
        "ev":                   round(ev_val, 4),
        "stake":                round(stake, 2),
        "status":               "pending",
        "profit":               None,
        "placed_at":            datetime.now().isoformat(),
        "settled_at":           None,
    }
    p["bankroll"] = round(p["bankroll"] - stake, 2)
    p["bets"].append(bet)
    return bet


def build_acca_suggestions(
    candidates: list[dict],
    min_single_ev: float = 0.0,
    max_legs: int = 3,
) -> list[dict]:
    """
    Build the highest-EV accumulator combinations from a list of single-bet candidates.
    One selection per fixture; only positive-EV singles considered.
    Returns list sorted by combined EV descending, each entry has 'legs', 'n_legs',
    'combined_prob', 'combined_odds', 'ev'.
    """
    from itertools import combinations as _combos

    # Best EV selection per fixture
    best: dict[tuple, dict] = {}
    for c in candidates:
        key = (c["home"], c["away"])
        if c["ev"] >= min_single_ev:
            if key not in best or c["ev"] > best[key]["ev"]:
                best[key] = c

    pool = list(best.values())
    results: list[dict] = []

    for n in range(2, min(max_legs + 1, len(pool) + 1)):
        for combo in _combos(pool, n):
            cp = 1.0
            co = 1.0
            for lg in combo:
                cp *= lg["model_prob"]
                co *= lg["odds"]
            results.append({
                "legs":           list(combo),
                "n_legs":         n,
                "combined_prob":  round(cp, 4),
                "combined_odds":  round(co, 2),
                "ev":             round(cp * co - 1.0, 4),
            })

    return sorted(results, key=lambda x: x["ev"], reverse=True)


# ── Stats ───────────────────────────────────────────────────────────────────────

def portfolio_stats(p: dict) -> dict:
    bets = p["bets"]
    settled = [b for b in bets if b["status"] in ("won", "lost")]
    pending  = [b for b in bets if b["status"] == "pending"]
    won      = [b for b in settled if b["status"] == "won"]
    staked   = sum(b["stake"] for b in settled) if settled else 0.0
    profit   = sum(b["profit"] for b in settled) if settled else 0.0
    return {
        "bankroll":    p["bankroll"],
        "initial":     p["initial_bankroll"],
        "profit":      round(profit, 2),
        "roi":         round(profit / staked * 100, 2) if staked > 0 else 0.0,
        "n_settled":   len(settled),
        "n_pending":   len(pending),
        "win_rate":    round(len(won) / len(settled) * 100, 1) if settled else 0.0,
        "staked":      round(staked, 2),
        "avg_odds":    round(float(np.mean([b["odds"] for b in settled])), 2) if settled else 0.0,
        "avg_ev_pct":  round(float(np.mean([b["ev"] for b in settled])) * 100, 1) if settled else 0.0,
        "best_win":    round(max((b["profit"] for b in settled), default=0.0), 2),
        "worst_loss":  round(min((b["profit"] for b in settled), default=0.0), 2),
    }


def bankroll_history(p: dict) -> pd.DataFrame:
    """Build bankroll time series from settled bets."""
    settled = sorted(
        [b for b in p["bets"] if b["status"] in ("won", "lost") and b.get("settled_at")],
        key=lambda b: b["settled_at"],
    )
    running = p["initial_bankroll"]
    rows = [{"idx": 0, "label": "Start", "bankroll": running, "profit": 0.0,
             "match": "—", "selection": "—", "result": "—"}]
    for i, b in enumerate(settled, 1):
        running = round(running + b["profit"], 2)
        if b.get("type") == "acca":
            match_label = " & ".join(
                f"{lg['home']} vs {lg['away']}" for lg in b.get("legs", [])
            )
            sel_label = " + ".join(
                lg.get("selection", "?") for lg in b.get("legs", [])
            )
            date_label = ""
        else:
            match_label = f"{b['home']} vs {b['away']}"
            sel_label = b.get("selection", "?")
            date_label = (b.get("date") or "")[:10]
        rows.append({
            "idx":       i,
            "label":     date_label,
            "match":     match_label,
            "selection": sel_label,
            "profit":    b["profit"],
            "bankroll":  running,
            "result":    b["status"],
        })
    return pd.DataFrame(rows)


# ── Auto-bet ───────────────────────────────────────────────────────────────────

def auto_place_value_bets(p: dict, candidates: list[dict], threshold: float,
                          max_auto_bets: int = 3,
                          calibrators: dict | None = None,
                          match_counts: dict[str, int] | None = None,
                          skip_log: list | None = None) -> list[dict]:
    """
    Automatically place Kelly-sized bets for every candidate with EV >= threshold.
    Skips duplicates (already have a pending bet on that home/away/market).
    Caps total pending exposure at 50% of initial bankroll.
    Caps auto-bets per run at max_auto_bets.

    Filters (all must pass):
    - Market must be in allowed auto_markets
    - Model probability (optionally calibrated) >= min_prob setting
    - EV (recomputed with calibrated prob if calibration enabled) >= threshold

    Each candidate dict: {home, away, date, market, selection, model_prob, odds, ev}
    Returns list of newly placed bets.
    """
    placed = []
    kelly_frac    = float(p["settings"].get("kelly_fraction", 0.5))
    max_stake_pct = float(p["settings"].get("max_stake_pct", 0.10))
    allowed_mkts  = set(p["settings"].get("auto_markets", list(PROFITABLE_MARKETS)))
    min_prob      = float(p["settings"].get("min_prob", 0.22))
    use_cal       = bool(p["settings"].get("use_calibrated_probs", True))
    use_sim       = bool(p["settings"].get("use_simultaneous_kelly", False))
    market_gates  = p["settings"].get("market_gates")

    # ── Optional v2-style filters for Main (defaults all OFF) ──
    main_max_ev      = p["settings"].get("main_max_ev_pct")
    main_max_ev      = float(main_max_ev) if main_max_ev is not None else None
    main_banned_dows   = set(p["settings"].get("main_banned_dows", []))
    main_banned_months = set(p["settings"].get("main_banned_months", []))
    main_min_te = p["settings"].get("main_min_team_elo")
    main_max_te = p["settings"].get("main_max_team_elo")
    main_gap_min = p["settings"].get("main_elo_gap_min")
    main_gap_max = p["settings"].get("main_elo_gap_max")
    main_elo_filter_on = any(v is not None for v in
                             (main_min_te, main_max_te, main_gap_min, main_gap_max))
    min_team_matches = p["settings"].get("min_team_matches")
    max_bets_per_club = p["settings"].get("max_bets_per_club")
    club_counts = (club_bet_counts(p["bets"], since=_season_start(p))
                   if max_bets_per_club else None)

    pending_singles = [b for b in p["bets"] if b["status"] == "pending" and b.get("type") != "acca"]
    pending_stake = sum(b["stake"] for b in p["bets"] if b["status"] == "pending")
    max_exposure  = p["initial_bankroll"] * 0.50

    if len(pending_singles) >= 5:
        return []

    # Pass 1 — enrich each candidate
    enriched = []
    for c in candidates:
        raw_prob = float(c.get("model_prob", 0.0))
        eff_prob = calibrate_prob(raw_prob, c.get("market", ""), calibrators) if use_cal else raw_prob
        _odds_raw = c.get("odds")
        odds = float(_odds_raw) if _odds_raw is not None else 0.0
        # Detect_odds: sharp benchmark for EV gate; falls back to placement odds
        _detect_raw = c.get("detect_odds")
        try:
            detect_odds = float(_detect_raw) if _detect_raw is not None and float(_detect_raw) > 1 else odds
        except (ValueError, TypeError):
            detect_odds = odds
        eff_ev = compute_ev(eff_prob, detect_odds) if detect_odds > 1 else -1.0

        # Compute Kelly fraction (don't multiply by bankroll yet — sim correction
        # may scale this down before placement)
        if odds > 1 and eff_prob > 0:
            b_ = odds - 1.0
            q_ = 1.0 - eff_prob
            full_kelly = max(0.0, (eff_prob * b_ - q_) / b_)
        else:
            full_kelly = 0.0
        kelly_pct = min(full_kelly * kelly_frac, max_stake_pct)

        enriched.append({**c, "_eff_prob": eff_prob, "_eff_ev": eff_ev,
                         "_raw_prob": raw_prob, "_kelly_pct": kelly_pct,
                         "_detect_odds": detect_odds})

    # Pass 2 — apply gates and pick the qualifying set, ranked by EV
    skip_late = bool(p["settings"].get("skip_late_season", False))
    qualifying: list[dict] = []
    for c in sorted(enriched, key=lambda x: x["_eff_ev"], reverse=True):
        if len(qualifying) >= max_auto_bets:
            break
        if not c.get("odds") or c["odds"] <= 1:
            continue
        # NaN guard (parity with ev_backtest_simulate): NaN<threshold is False,
        # so a NaN EV/prob would sail through the gates below. Reject explicitly
        # rather than relying on the kelly_pct<=0 catch further down.
        if pd.isna(c["_eff_ev"]) or pd.isna(c["_eff_prob"]):
            continue
        # No-history gate: promoted sides the model has never rated
        if should_skip_unrated(c["home"], c["away"], match_counts, min_team_matches):
            _log_skip(skip_log, "no_history", c["home"], c["away"],
                      _unrated_detail(c["home"], c["away"], match_counts,
                                      int(min_team_matches)))
            continue
        if skip_late and c.get("date") and _is_late_season(c["date"]):
            continue
        # Main: optional calendar filter (DOW/Month bans)
        if _should_skip_calendar(c.get("date"), main_banned_dows, main_banned_months):
            continue
        # Main: optional ELO-profile filter
        if main_elo_filter_on and should_skip_elo_profile(
                c.get("home_elo"), c.get("away_elo"),
                main_min_te, main_max_te, main_gap_min, main_gap_max,
                require_known_elo=True):
            _log_skip(skip_log, "elo_profile", c["home"], c["away"],
                      f"home_elo={c.get('home_elo')} away_elo={c.get('away_elo')} "
                      f"outside band (floor {main_min_te})")
            continue
        # Main: optional max-EV cap
        if main_max_ev is not None and c["_eff_ev"] > main_max_ev:
            continue
        if c["market"] not in allowed_mkts:
            continue
        if c["_eff_prob"] < _gate(market_gates, c["market"], "min_prob", min_prob):
            continue
        if c["_eff_ev"] < _gate(market_gates, c["market"], "min_ev", threshold):
            continue
        if any(b.get("type") != "acca" and b["status"] == "pending"
               and b["home"] == c["home"] and b["away"] == c["away"]
               and b["market"] == c["market"] for b in p["bets"]):
            continue
        if c["_kelly_pct"] <= 0:
            continue
        # Club-exposure cap: don't let one club carry the season. Counted live
        # so bets queued earlier in this same scan use up the allowance too.
        if should_skip_club_exposure(c["home"], c["away"], club_counts,
                                     max_bets_per_club):
            _log_skip(skip_log, "club_exposure", c["home"], c["away"],
                      f"already {club_counts.get(c['home'], 0)}/"
                      f"{club_counts.get(c['away'], 0)} bets this season "
                      f"(cap {max_bets_per_club})")
            continue
        if club_counts is not None:
            for club in (c["home"], c["away"]):
                club_counts[club] = club_counts.get(club, 0) + 1
        qualifying.append(c)

    # Pass 3 — apply simultaneous-bet correction across the qualifying set.
    # Mirrors Mock Two's stack: bankroll cannot recompound between concurrent
    # settlements, so each stake should be reduced for capital tied up in the
    # other bets being placed in the same scan. Empirically prevented the
    # Saturday-card variance blow-up in WF.
    if use_sim and len(qualifying) > 1:
        corrected = simultaneous_kelly_correction([q["_kelly_pct"] for q in qualifying])
        for q, k_new in zip(qualifying, corrected):
            q["_sim_factor"] = round(k_new / q["_kelly_pct"], 4) if q["_kelly_pct"] > 0 else 1.0
            q["_kelly_pct"]  = k_new
    else:
        for q in qualifying:
            q["_sim_factor"] = 1.0

    # Pass 4 — place bets
    for c in qualifying:
        if pending_stake >= max_exposure:
            break
        stake = round(p["bankroll"] * c["_kelly_pct"], 2)
        if stake <= 0 or stake > p["bankroll"]:
            continue
        stake = min(stake, max_exposure - pending_stake)
        if stake <= 0:
            break

        bet = place_bet(
            p, c["home"], c["away"], c["date"],
            c["market"], c["selection"],
            c["_raw_prob"], c["odds"], stake,
        )
        # Tag with sim factor for auditing parity with v2
        bet["sim_factor"] = c["_sim_factor"]
        if c.get("_detect_odds") and c["_detect_odds"] != c["odds"]:
            bet["detect_odds"] = c["_detect_odds"]
        placed.append(bet)
        pending_stake += stake

    return placed


# ── Historical EV simulation ────────────────────────────────────────────────────

def _is_late_season(d) -> bool:
    """True for matches in March-April — the empirical "run-in" window
    where the model has shown 0/7 wins across 2024-25 and 2025-26 combined.

    May was previously bundled in by implementation, but the empirical
    evidence is Mar-Apr only. May is no longer filtered by skip_late_season.
    """
    return pd.Timestamp(d).month in (3, 4)


def _gate(market_gates: dict | None, market: str, key: str, default):
    """Per-market gate lookup with fallback to the global default.

    market_gates: optional dict {market_code: {gate_key: value}}. Lets callers
    apply different min_prob / min_ev thresholds per market — e.g. a tight 40%
    EV gate on draws (which sit at long odds and need conviction) and a loose
    8% EV gate on Under 2.5 (which trades at lower odds with smaller margins).
    """
    if market_gates and market in market_gates and key in market_gates[market]:
        return market_gates[market][key]
    return default


# Bookmaker odds sources available in football-data CSVs.
#   "B365" : Bet365 kickoff
#   "Max"  : best price across the panel at kickoff (recommended for +EV)
#   "Avg"  : panel average kickoff
#   "PS"   : Pinnacle kickoff (sharp, used for CLV)
# Each maps to the column prefix; O/U keys are None when the source has no
# over/under coverage (PS doesn't, in football-data).
ODDS_SOURCES = {
    "B365": {"H": "B365H", "D": "B365D", "A": "B365A",
             "O25": "B365>2.5", "U25": "B365<2.5"},
    "Max":  {"H": "MaxH",  "D": "MaxD",  "A": "MaxA",
             "O25": "Max>2.5",  "U25": "Max<2.5"},
    "Avg":  {"H": "AvgH",  "D": "AvgD",  "A": "AvgA",
             "O25": "Avg>2.5",  "U25": "Avg<2.5"},
    "PS":   {"H": "PSH",   "D": "PSD",   "A": "PSA",
             "O25": None,       "U25": None},
}


def ev_backtest_simulate(
    bt_df: pd.DataFrame,
    df: pd.DataFrame,
    min_ev_pct: float = 5.0,
    kelly_frac: float = 0.5,
    max_stake_pct: float = 0.10,
    initial_bankroll: float = 1000.0,
    allowed_markets: set | None = None,
    min_prob: float = 0.0,
    skip_late_season: bool = False,
    skip_home_title_race: bool = False,
    odds_source: str = "B365",
    detect_source: str | None = None,
    enable_simultaneous_correction: bool = False,
    market_gates: dict | None = None,
    # Validated filters ported from v2 (defaults preserve old behaviour)
    max_ev_pct: float | None = None,
    banned_dows: set[str] | None = None,
    banned_months: set[str] | None = None,
    min_team_elo: float | None = None,
    max_team_elo: float | None = None,
    elo_gap_min: float | None = None,
    elo_gap_max: float | None = None,
    # No-history gate: both sides must have this many prior top-flight matches
    min_team_matches: int | None = None,
    # Club-exposure cap: most bets one club may carry across the window
    max_bets_per_club: int | None = None,
    df_features: pd.DataFrame | None = None,
    # xG-regression filter: skip when team is luck-driven (actual vs xG diverge)
    xg_overperform_threshold: float | None = None,
    # Calibration — when provided, applies isotonic calibration to model
    # probs before EV gate, matching live auto_place_value_bets behaviour.
    # Caller is responsible for ensuring no train/test leakage.
    calibrators: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate EV-based Kelly betting using full-ensemble (DC + XGB + Draw Specialist)
    model probabilities vs historical odds from chosen bookmaker source.

    bt_df:         output of backtest_models() — must have DC_H, DC_D, DC_A columns.
    df:            full match DataFrame with the chosen odds columns present.
    odds_source:   "B365" | "Max" | "Avg" | "PS" — the price you ACTUALLY PLACE at.
    detect_source: optional separate source used for EV detection only. When set,
                   EV is computed against detect_source (e.g. PS for sharpness),
                   then bets are placed at odds_source (e.g. Max for realism).
                   Defaults to odds_source (single-source mode).
    min_prob:      probability floor (0–1). Skipped if calibrated prob below.
    Returns (bet_log_df, summary_dict).
    """
    if odds_source not in ODDS_SOURCES:
        return pd.DataFrame(), {"error": f"Unknown odds_source '{odds_source}'"}
    if detect_source is None:
        detect_source = odds_source
    if detect_source not in ODDS_SOURCES:
        return pd.DataFrame(), {"error": f"Unknown detect_source '{detect_source}'"}
    place_src  = ODDS_SOURCES[odds_source]
    detect_src = ODDS_SOURCES[detect_source]
    # Backward-compat alias
    src = place_src
    if bt_df.empty:
        return pd.DataFrame(), {"error": "Backtest returned no data"}

    min_ev = min_ev_pct / 100.0
    # Place columns
    h_col, d_col, a_col = place_src["H"], place_src["D"], place_src["A"]
    # Detect columns (may be same as place)
    dh_col, dd_col, da_col = detect_src["H"], detect_src["D"], detect_src["A"]
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
              h_col, d_col, a_col, dh_col, dd_col, da_col]
    needed = list(dict.fromkeys(needed))  # de-dup
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(), {"error": f"{odds_source}/{detect_source} 1X2 odds not found"}

    # Optional O/U columns — both detect and place need them
    o25_col  = place_src.get("O25");  u25_col  = place_src.get("U25")
    do25_col = detect_src.get("O25"); du25_col = detect_src.get("U25")
    has_ou   = (o25_col and u25_col
                and o25_col in df.columns and u25_col in df.columns
                and do25_col and du25_col
                and do25_col in df.columns and du25_col in df.columns
                and "DC_O25" in bt_df.columns)
    ou_cols  = (list({o25_col, u25_col, do25_col, du25_col}) if has_ou else [])

    # Enrich df with motivation state if we need to filter on it
    df_for_odds = df
    if skip_home_title_race:
        from data import add_motivation_features
        df_for_odds = add_motivation_features(df, df)

    cols_to_keep = needed + ou_cols
    if "h_motivation" in df_for_odds.columns:
        cols_to_keep = cols_to_keep + ["h_motivation", "a_motivation"]
    cols_to_keep = list(dict.fromkeys(cols_to_keep))  # de-dup

    odds_df = (
        df_for_odds[cols_to_keep]
        .rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})
        .dropna(subset=[h_col, d_col, a_col])
    )
    merged = bt_df.merge(odds_df, on=["Date", "Home", "Away"], how="inner")

    if merged.empty:
        return pd.DataFrame(), {
            "error": "No test matches have B365 odds — try a longer test window"
        }

    # Attach historical ELO at time of match (for ELO-profile filter)
    elo_filter_active = any(v is not None for v in (
        min_team_elo, max_team_elo, elo_gap_min, elo_gap_max))
    if elo_filter_active and df_features is not None and not df_features.empty:
        elo_cols_present = [c for c in ("home_elo", "away_elo")
                            if c in df_features.columns]
        if len(elo_cols_present) == 2:
            elo_slice = (df_features[["Date", "HomeTeam", "AwayTeam",
                                       "home_elo", "away_elo"]]
                         .rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}))
            merged = merged.merge(
                elo_slice, on=["Date", "Home", "Away"], how="left",
            )

    # Attach rolling xG/goals for xG-regression filter (luck detection)
    xg_filter_active = xg_overperform_threshold is not None
    if xg_filter_active and df_features is not None and not df_features.empty:
        xg_cols_needed = ["home_roll_gf", "home_roll_xg",
                          "away_roll_gf", "away_roll_xg"]
        if all(c in df_features.columns for c in xg_cols_needed):
            xg_slice = (df_features[["Date", "HomeTeam", "AwayTeam"] + xg_cols_needed]
                        .rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}))
            merged = merged.merge(
                xg_slice, on=["Date", "Home", "Away"], how="left",
            )

    bankroll = initial_bankroll
    rows = []
    skipped_min_prob = 0
    skipped_late_season = 0
    skipped_home_title_race = 0
    skipped_calendar = 0
    skipped_max_ev   = 0
    skipped_elo      = 0
    skipped_xg       = 0
    skipped_no_history = 0
    skipped_club_exposure = 0
    club_counts: dict[str, int] | None = {} if max_bets_per_club else None
    sim_factors_observed = []

    # Point-in-time history table, built from the FULL dataset so a team's
    # pre-window matches count toward the threshold.
    rated_from = _rated_from_table(df, min_team_matches)

    # Date-grouped two-pass loop: collect candidates per day, apply optional
    # simultaneous-bet correction, then place. This mirrors v2 and lets us
    # extract the same Kelly variance reduction on Saturday cards.
    for date, day_matches in merged.sort_values("Date").groupby("Date", sort=True):
        if skip_late_season and _is_late_season(date):
            skipped_late_season += int(len(day_matches) * 3)
            continue
        if _should_skip_calendar(date, banned_dows, banned_months):
            skipped_calendar += int(len(day_matches) * 3)
            continue

        candidates: list[dict] = []
        for _, r in day_matches.iterrows():
            # No-history gate, point-in-time so a promoted side is blocked
            # during its own first matches rather than retroactively rated
            if should_skip_unrated_at(r["Home"], r["Away"], date,
                                      rated_from, min_team_matches):
                skipped_no_history += 3
                continue
            if skip_home_title_race and r.get("h_motivation") == "title_race":
                skipped_home_title_race += 1
                continue
            # Main: ELO-profile filter (validated multi-season winner = 1500)
            if elo_filter_active and should_skip_elo_profile(
                    r.get("home_elo"), r.get("away_elo"),
                    min_team_elo, max_team_elo, elo_gap_min, elo_gap_max,
                    require_known_elo=True):
                skipped_elo += 3
                continue
            # Main: xG-regression filter (skip luck-driven teams)
            if xg_filter_active and should_skip_xg_overperform(
                    r.get("home_roll_gf"), r.get("home_roll_xg"),
                    r.get("away_roll_gf"), r.get("away_roll_xg"),
                    xg_overperform_threshold):
                skipped_xg += 3
                continue
            try:
                o_h = float(r[h_col]); o_d = float(r[d_col]); o_a = float(r[a_col])
                d_h = float(r[dh_col]); d_d = float(r[dd_col]); d_a = float(r[da_col])
            except (ValueError, TypeError):
                continue
            if o_h <= 1 or o_d <= 1 or o_a <= 1: continue
            if d_h <= 1 or d_d <= 1 or d_a <= 1: continue

            actual = r["Actual"]
            total_goals = int(r.get("FTHG", 0) or 0) + int(r.get("FTAG", 0) or 0)
            p_h = float(r["DC_H"]) / 100.0
            p_d = float(r["DC_D"]) / 100.0
            p_a = float(r["DC_A"]) / 100.0

            market_list = [
                ("H", p_h, o_h, d_h, "Home Win"),
                ("D", p_d, o_d, d_d, "Draw"),
                ("A", p_a, o_a, d_a, "Away Win"),
            ]
            if has_ou:
                try:
                    p_o25  = float(r["DC_O25"]) / 100.0
                    o_over = float(r[o25_col]);  o_under = float(r[u25_col])
                    d_over = float(r[do25_col]); d_under = float(r[du25_col])
                    if o_over > 1 and o_under > 1 and d_over > 1 and d_under > 1:
                        market_list += [
                            ("over25",  p_o25,       o_over,  d_over,  "Over 2.5"),
                            ("under25", 1.0 - p_o25, o_under, d_under, "Under 2.5"),
                        ]
                except (ValueError, TypeError, KeyError):
                    pass

            for mkt, prob, place_odds, detect_odds, outcome in market_list:
                if allowed_markets and mkt not in allowed_markets:
                    continue
                # Apply isotonic calibration if available (matches live behaviour)
                if calibrators:
                    prob = calibrate_prob(prob, mkt, calibrators)
                if prob < _gate(market_gates, mkt, "min_prob", min_prob):
                    skipped_min_prob += 1
                    continue
                ev_val = compute_ev(prob, detect_odds)
                # NaN guard: missing odds → NaN EV → NaN<threshold is False
                # which would silently let the bet through. Reject explicitly.
                if pd.isna(ev_val):
                    continue
                if ev_val < _gate(market_gates, mkt, "min_ev", min_ev):
                    continue
                # Main: max-EV cap (high-EV bucket calibrates badly per Phase 1)
                if max_ev_pct is not None and ev_val > max_ev_pct:
                    skipped_max_ev += 1
                    continue

                # Compute Kelly fraction (don't multiply by bankroll yet — sim
                # correction may scale this down before placement)
                if place_odds > 1 and prob > 0:
                    b_ = place_odds - 1.0
                    q_ = 1.0 - prob
                    full_kelly = max(0.0, (prob * b_ - q_) / b_)
                else:
                    full_kelly = 0.0
                kelly_pct_raw = min(full_kelly * kelly_frac, max_stake_pct)
                if kelly_pct_raw <= 0:
                    continue

                won = ((total_goals > 2) if mkt == "over25"
                       else (total_goals <= 2) if mkt == "under25"
                       else (actual == outcome))

                candidates.append({
                    "Date":          r["Date"],
                    "Home":          r["Home"],
                    "Away":          r["Away"],
                    "mkt":           mkt,
                    "outcome":       outcome,
                    "prob":          prob,
                    "place_odds":    place_odds,
                    "detect_odds":   detect_odds,
                    "ev_val":        ev_val,
                    "kelly_pct_raw": kelly_pct_raw,
                    "won":           won,
                })

        if not candidates:
            continue

        # Apply simultaneous-bet correction across the day's qualifying set
        if enable_simultaneous_correction and len(candidates) > 1:
            corrected = simultaneous_kelly_correction(
                [c["kelly_pct_raw"] for c in candidates]
            )
            for c, k_new in zip(candidates, corrected):
                c["kelly_pct_final"] = k_new
                c["sim_factor"] = (k_new / c["kelly_pct_raw"]) if c["kelly_pct_raw"] > 0 else 1.0
        else:
            for c in candidates:
                c["kelly_pct_final"] = c["kelly_pct_raw"]
                c["sim_factor"] = 1.0

        # Place bets
        for c in candidates:
            # Club-exposure cap, counted on bets actually placed rather than on
            # candidates, so a club that never clears the gates keeps its
            # allowance intact.
            if should_skip_club_exposure(c["Home"], c["Away"], club_counts,
                                         max_bets_per_club):
                skipped_club_exposure += 1
                continue
            stake = round(bankroll * c["kelly_pct_final"], 2)
            if stake <= 0 or stake > bankroll:
                continue
            if club_counts is not None:
                for club in (c["Home"], c["Away"]):
                    club_counts[club] = club_counts.get(club, 0) + 1
            bankroll -= stake
            if c["won"]:
                ret = round(stake * c["place_odds"], 2)
                profit = round(ret - stake, 2)
                bankroll = round(bankroll + ret, 2)
            else:
                profit = round(-stake, 2)
            sim_factors_observed.append(c["sim_factor"])
            rows.append({
                "Date":       c["Date"],
                "Match":      f"{c['Home']} vs {c['Away']}",
                "Market":     {"H": "Home Win", "D": "Draw", "A": "Away Win",
                               "over25": "Over 2.5", "under25": "Under 2.5"}.get(
                                   c["mkt"], c["mkt"]),
                "Model%":     f"{c['prob']*100:.1f}%",
                "Implied%":   f"{100/c['place_odds']:.1f}%",
                "EV":         f"+{c['ev_val']*100:.1f}%",
                "Odds":       round(c["place_odds"], 2),
                "DetectOdds": round(c["detect_odds"], 2),
                "SimFactor":  round(c["sim_factor"], 3),
                "Stake":      round(stake, 2),
                "Result":     "✅" if c["won"] else "❌",
                "Profit":     round(profit, 2),
                "Bankroll":   round(bankroll, 2),
            })

    # Accas removed from backtest — they consistently lose (-16% to -46% ROI)
    # because needing all legs to hit compounds the miss rate.
    # Singles-only backtest matches the grid search results accurately.

    if not rows:
        return pd.DataFrame(), {
            "error": "No value bets found — try lowering the EV threshold"
        }

    log = pd.DataFrame(rows)
    n_won         = (log["Result"] == "✅").sum()
    staked_total  = log["Stake"].sum()
    profit_total  = log["Profit"].sum()

    summary = {
        "initial":          initial_bankroll,
        "final":            round(bankroll, 2),
        "profit":           round(profit_total, 2),
        "roi":              round(profit_total / staked_total * 100, 2) if staked_total > 0 else 0.0,
        "n_bets":           len(log),
        "win_rate":         round(n_won / len(log) * 100, 1),
        "avg_odds":         round(float(log["Odds"].mean()), 2),
        "total_staked":     round(staked_total, 2),
        "skipped_min_prob": skipped_min_prob,
        "skipped_late_season": skipped_late_season,
        "skipped_home_title_race": skipped_home_title_race,
        "skipped_calendar": skipped_calendar,
        "skipped_max_ev":   skipped_max_ev,
        "skipped_elo":      skipped_elo,
        "skipped_xg":       skipped_xg,
        "skipped_no_history": skipped_no_history,
        "skipped_club_exposure": skipped_club_exposure,
        "odds_source": odds_source,
        "detect_source": detect_source,
        "mean_sim_factor": (round(float(np.mean(sim_factors_observed)), 3)
                            if sim_factors_observed else 1.0),
    }
    return log, summary


def ev_backtest_simulate_v2(
    bt_df: pd.DataFrame,
    df: pd.DataFrame,
    min_ev_pct: float = 5.0,
    base_kelly_frac: float = 0.5,
    max_stake_pct: float = 0.10,
    initial_bankroll: float = 1000.0,
    allowed_markets: set | None = None,
    min_prob: float = 0.0,
    enable_simultaneous_correction: bool = True,
    bin_variances: dict | None = None,
    skip_late_season: bool = False,
    skip_home_title_race: bool = False,
    odds_source: str = "B365",
    detect_source: str | None = None,
    market_gates: dict | None = None,
    # ── Mock Two Phase 2 filters (defaults preserve old behaviour) ──
    team_roi_filter: bool = False,
    team_roi_threshold_pct: float = -25.0,
    team_roi_min_n: int = 3,
    max_ev_pct: float | None = None,
    banned_dows: set[str] | None = None,
    banned_months: set[str] | None = None,
    enable_drawdown_throttle: bool = False,
    drawdown_at_pct: float = 0.20,
    drawdown_min_factor: float = 0.25,
    # ── ELO-profile filters (per-match, no look-ahead — read from df) ──
    min_team_elo: float | None = None,
    max_team_elo: float | None = None,
    elo_gap_min: float | None = None,
    elo_gap_max: float | None = None,
    # No-history gate: both sides must have this many prior top-flight matches
    min_team_matches: int | None = None,
    # Club-exposure cap: most bets one club may carry across the window
    max_bets_per_club: int | None = None,
    df_features: pd.DataFrame | None = None,
    # Optional isotonic calibration — matches live auto_place_value_bets_v2.
    calibrators: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Mock Two historical EV simulator.

    Uses the K-N + XGB + Draw Specialist probabilities produced by
    `backtest_models_v2`, sizes via Baker-McHale uncertainty-shrunk Kelly, and
    applies simultaneous-bet correction across same-day cards (the realistic
    Saturday case where bankroll cannot recompound between settlements).

    `bin_variances`: optional dict {"H": [...], "D": [...], "A": [...]} of
    bin-variance lists (output of `compute_per_bin_variance`). If None, fits
    in-sample on bt_df. O/U markets fall back to a moderate-uncertainty default.

    Returns (bet_log_df, summary_dict). Summary tags `engine="v2"` and includes
    `skipped_min_prob`, `mean_shrinkage`, `mean_sim_factor`.
    """
    if bt_df.empty:
        return pd.DataFrame(), {"error": "Backtest returned no data"}
    if odds_source not in ODDS_SOURCES:
        return pd.DataFrame(), {"error": f"Unknown odds_source '{odds_source}'"}
    if detect_source is None:
        detect_source = odds_source
    if detect_source not in ODDS_SOURCES:
        return pd.DataFrame(), {"error": f"Unknown detect_source '{detect_source}'"}
    place_src  = ODDS_SOURCES[odds_source]
    detect_src = ODDS_SOURCES[detect_source]

    min_ev = min_ev_pct / 100.0
    h_col,  d_col,  a_col  = place_src["H"],  place_src["D"],  place_src["A"]
    dh_col, dd_col, da_col = detect_src["H"], detect_src["D"], detect_src["A"]
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
              h_col, d_col, a_col, dh_col, dd_col, da_col]
    needed = list(dict.fromkeys(needed))
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(), {"error": f"{odds_source}/{detect_source} 1X2 odds not found"}

    o25_col  = place_src.get("O25");  u25_col  = place_src.get("U25")
    do25_col = detect_src.get("O25"); du25_col = detect_src.get("U25")
    has_ou = (o25_col and u25_col
              and o25_col in df.columns and u25_col in df.columns
              and do25_col and du25_col
              and do25_col in df.columns and du25_col in df.columns
              and "DC_O25" in bt_df.columns)
    ou_cols = (list({o25_col, u25_col, do25_col, du25_col}) if has_ou else [])

    df_for_odds = df
    if skip_home_title_race:
        from data import add_motivation_features
        df_for_odds = add_motivation_features(df, df)

    cols_to_keep = needed + ou_cols
    if "h_motivation" in df_for_odds.columns:
        cols_to_keep = cols_to_keep + ["h_motivation", "a_motivation"]
    cols_to_keep = list(dict.fromkeys(cols_to_keep))

    odds_df = (
        df_for_odds[cols_to_keep]
        .rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})
        .dropna(subset=[h_col, d_col, a_col])
    )
    merged = bt_df.merge(odds_df, on=["Date", "Home", "Away"], how="inner")
    if merged.empty:
        return pd.DataFrame(), {
            "error": f"No test matches have {odds_source} odds — try a longer test window"
        }

    # Attach historical ELO at time of match (from df_features if provided).
    # df_features rows have home_elo / away_elo computed from prior matches —
    # no look-ahead. Merge on (Date, Home, Away) with NaN-safe fallback.
    elo_filter_active = any(v is not None for v in (
        min_team_elo, max_team_elo, elo_gap_min, elo_gap_max))
    if elo_filter_active and df_features is not None and not df_features.empty:
        elo_cols_present = [c for c in ("home_elo", "away_elo")
                            if c in df_features.columns]
        if len(elo_cols_present) == 2:
            elo_slice = (df_features[["Date", "HomeTeam", "AwayTeam",
                                       "home_elo", "away_elo"]]
                         .rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}))
            merged = merged.merge(
                elo_slice, on=["Date", "Home", "Away"], how="left",
            )

    if bin_variances is None:
        bin_variances = {
            "H": compute_per_bin_variance(bt_df, "H"),
            "D": compute_per_bin_variance(bt_df, "D"),
            "A": compute_per_bin_variance(bt_df, "A"),
        }

    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    rows = []
    skipped_min_prob = 0
    skipped_late_season = 0
    skipped_home_title_race = 0
    skipped_calendar    = 0
    skipped_team_roi    = 0
    skipped_max_ev      = 0
    skipped_elo         = 0
    skipped_no_history  = 0
    skipped_club_exposure = 0
    club_counts: dict[str, int] | None = {} if max_bets_per_club else None
    rated_from = _rated_from_table(df, min_team_matches)
    shrinkages = []
    sim_factors = []
    dd_factors_applied = []

    # Rolling per-team P&L for team_roi_filter — updated as bets settle
    # so the filter sees only past info, never look-ahead.
    rolling_team_stats: dict[str, dict] = {}

    def _rolling_team_roi() -> dict[str, dict]:
        """Snapshot rolling team ROI filtered by min_n threshold."""
        if not team_roi_filter:
            return {}
        out = {}
        for t, g in rolling_team_stats.items():
            if g["n"] < team_roi_min_n:
                continue
            roi = (g["profit"] / g["stake"] * 100.0) if g["stake"] > 0 else 0.0
            out[t] = {"n": g["n"], "roi": roi,
                      "profit": g["profit"], "stake": g["stake"]}
        return out

    for date, day_matches in merged.sort_values("Date").groupby("Date", sort=True):
        if skip_late_season and _is_late_season(date):
            skipped_late_season += int(len(day_matches) * 3)  # ~3 markets/match approx
            continue
        if _should_skip_calendar(date, banned_dows, banned_months):
            skipped_calendar += int(len(day_matches) * 3)
            continue
        # Snapshot team-ROI table at the start of this day so all candidates
        # in the same day see the same filter (deterministic; no within-day
        # contamination from a bet placed earlier the same day).
        team_roi_snapshot = _rolling_team_roi()
        # 1) Build candidate list for this date
        candidates = []
        for _, r in day_matches.iterrows():
            if skip_home_title_race and r.get("h_motivation") == "title_race":
                skipped_home_title_race += 1
                continue
            # No-history gate, point-in-time (see ev_backtest_simulate)
            if should_skip_unrated_at(r["Home"], r["Away"], date,
                                      rated_from, min_team_matches):
                skipped_no_history += 3
                continue
            # Mock Two team-ROI blacklist
            if team_roi_filter and should_skip_team_pair(
                    team_roi_snapshot, r["Home"], r["Away"], team_roi_threshold_pct):
                skipped_team_roi += 3  # ~3 markets per match
                continue
            # Mock Two ELO-profile filter
            if elo_filter_active and should_skip_elo_profile(
                    r.get("home_elo"), r.get("away_elo"),
                    min_team_elo, max_team_elo,
                    elo_gap_min, elo_gap_max,
                    require_known_elo=True):
                skipped_elo += 3  # ~3 markets per match
                continue
            try:
                o_h = float(r[h_col]); o_d = float(r[d_col]); o_a = float(r[a_col])
                d_h = float(r[dh_col]); d_d = float(r[dd_col]); d_a = float(r[da_col])
            except (ValueError, TypeError):
                continue
            if o_h <= 1 or o_d <= 1 or o_a <= 1: continue
            if d_h <= 1 or d_d <= 1 or d_a <= 1: continue

            actual = r["Actual"]
            total_goals = int(r.get("FTHG", 0) or 0) + int(r.get("FTAG", 0) or 0)
            p_h = float(r["DC_H"]) / 100.0
            p_d = float(r["DC_D"]) / 100.0
            p_a = float(r["DC_A"]) / 100.0

            # Tuples: (code, prob, place_odds, detect_odds, name)
            market_list = [
                ("H", p_h, o_h, d_h, "Home Win"),
                ("D", p_d, o_d, d_d, "Draw"),
                ("A", p_a, o_a, d_a, "Away Win"),
            ]
            if has_ou:
                try:
                    p_o25  = float(r["DC_O25"]) / 100.0
                    o_over = float(r[o25_col]);  o_under = float(r[u25_col])
                    d_over = float(r[do25_col]); d_under = float(r[du25_col])
                    if o_over > 1 and o_under > 1 and d_over > 1 and d_under > 1:
                        market_list += [
                            ("over25",  p_o25,       o_over,  d_over,  "Over 2.5"),
                            ("under25", 1.0 - p_o25, o_under, d_under, "Under 2.5"),
                        ]
                except (ValueError, TypeError, KeyError):
                    pass

            for mkt, prob, place_odds, detect_odds, outcome in market_list:
                if allowed_markets and mkt not in allowed_markets:
                    continue
                # Apply isotonic calibration if available (matches live behaviour)
                if calibrators:
                    prob = calibrate_prob(prob, mkt, calibrators)
                if prob < _gate(market_gates, mkt, "min_prob", min_prob):
                    skipped_min_prob += 1
                    continue
                # EV detection vs detect_odds (sharper); placement at place_odds
                ev_val = compute_ev(prob, detect_odds)
                # NaN guard: missing odds → NaN EV → NaN<threshold is False
                # which would silently let the bet through. Reject explicitly.
                if pd.isna(ev_val):
                    continue
                if ev_val < _gate(market_gates, mkt, "min_ev", min_ev):
                    continue
                # Mock Two: cap claimed EV (high-EV bucket calibrates badly)
                if max_ev_pct is not None and ev_val > max_ev_pct:
                    skipped_max_ev += 1
                    continue

                var_p = (lookup_bin_variance(bin_variances.get(mkt, []), prob)
                         if mkt in ("H", "D", "A") else 0.01)
                stake_raw, shrinkage = kelly_stake_uncertainty_adjusted(
                    prob, place_odds, bankroll, var_p, base_kelly_frac, max_stake_pct,
                )
                if stake_raw <= 0:
                    continue
                kelly_pct_raw = stake_raw / bankroll if bankroll > 0 else 0.0

                won = ((total_goals > 2) if mkt == "over25"
                       else (total_goals <= 2) if mkt == "under25"
                       else (actual == outcome))

                candidates.append({
                    "Date":      r["Date"],
                    "Home":      r["Home"],
                    "Away":      r["Away"],
                    "Match":     f"{r['Home']} vs {r['Away']}",
                    "Market":    {"H": "Home Win", "D": "Draw", "A": "Away Win",
                                  "over25": "Over 2.5", "under25": "Under 2.5"}.get(mkt, mkt),
                    "Model%":    f"{prob*100:.1f}%",
                    "Implied%":  f"{100/place_odds:.1f}%",
                    "EV":        f"+{ev_val*100:.1f}%",
                    "Odds":      round(place_odds, 2),
                    "DetectOdds": round(detect_odds, 2),
                    "_kelly":    kelly_pct_raw,
                    "_shrink":   shrinkage,
                    "_won":      won,
                    "_var_p":    var_p,
                })

        if not candidates:
            continue

        # 2) Simultaneous-bet correction for same-day card
        if enable_simultaneous_correction and len(candidates) > 1:
            adjusted = simultaneous_kelly_correction([c["_kelly"] for c in candidates])
            for c, kj in zip(candidates, adjusted):
                c["_sim_factor"] = (kj / c["_kelly"]) if c["_kelly"] > 0 else 1.0
                c["_kelly_final"] = kj
        else:
            for c in candidates:
                c["_sim_factor"] = 1.0
                c["_kelly_final"] = c["_kelly"]

        # 3) Settle bets in date order (deterministic intra-day)
        # Compute drawdown throttle once per day so all bets in the same card
        # see the same factor (matches the live auto_place_value_bets_v2 path)
        if enable_drawdown_throttle:
            dd_factor = kelly_drawdown_throttle(
                bankroll, peak_bankroll, drawdown_at_pct, drawdown_min_factor,
            )
        else:
            dd_factor = 1.0

        for c in candidates:
            # Club-exposure cap, same rule as Main so the A/B stays honest.
            if should_skip_club_exposure(c["Home"], c["Away"], club_counts,
                                         max_bets_per_club):
                skipped_club_exposure += 1
                continue
            stake = round(bankroll * c["_kelly_final"] * dd_factor, 2)
            if stake <= 0 or stake > bankroll:
                continue
            if club_counts is not None:
                for club in (c["Home"], c["Away"]):
                    club_counts[club] = club_counts.get(club, 0) + 1
            bankroll -= stake
            if c["_won"]:
                ret = round(stake * c["Odds"], 2)
                profit = round(ret - stake, 2)
                bankroll = round(bankroll + ret, 2)
            else:
                profit = round(-stake, 2)
            peak_bankroll = max(peak_bankroll, bankroll)

            shrinkages.append(c["_shrink"])
            sim_factors.append(c["_sim_factor"])
            dd_factors_applied.append(dd_factor)
            # Update rolling per-team stats for the team-ROI filter
            if team_roi_filter:
                # Match field is "Home vs Away"
                m = c["Match"].split(" vs ", 1)
                if len(m) == 2:
                    for team in m:
                        g = rolling_team_stats.setdefault(
                            team, {"n": 0, "stake": 0.0, "profit": 0.0})
                        g["n"] += 1
                        g["stake"] += float(stake)
                        g["profit"] += float(profit)
            rows.append({
                "Date":       c["Date"],
                "Match":      c["Match"],
                "Market":     c["Market"],
                "Model%":     c["Model%"],
                "Implied%":   c["Implied%"],
                "EV":         c["EV"],
                "Odds":       c["Odds"],
                "DetectOdds": c.get("DetectOdds", c["Odds"]),
                "Shrink":     round(c["_shrink"], 3),
                "SimFactor":  round(c["_sim_factor"], 3),
                "DDFactor":   round(dd_factor, 3),
                "Stake":      stake,
                "Result":     "✅" if c["_won"] else "❌",
                "Profit":     round(profit, 2),
                "Bankroll":   round(bankroll, 2),
            })

    if not rows:
        return pd.DataFrame(), {
            "error": "No value bets found — try lowering the EV threshold or min-prob"
        }

    log = pd.DataFrame(rows)
    n_won        = (log["Result"] == "✅").sum()
    staked_total = log["Stake"].sum()
    profit_total = log["Profit"].sum()

    summary = {
        "engine":           "v2",
        "initial":          initial_bankroll,
        "final":            round(bankroll, 2),
        "peak":             round(peak_bankroll, 2),
        "profit":           round(profit_total, 2),
        "roi":              round(profit_total / staked_total * 100, 2) if staked_total > 0 else 0.0,
        "n_bets":           len(log),
        "win_rate":         round(n_won / len(log) * 100, 1),
        "avg_odds":         round(float(log["Odds"].mean()), 2),
        "total_staked":     round(staked_total, 2),
        "skipped_min_prob": skipped_min_prob,
        "skipped_late_season": skipped_late_season,
        "skipped_home_title_race": skipped_home_title_race,
        "skipped_calendar":   skipped_calendar,
        "skipped_team_roi":   skipped_team_roi,
        "skipped_max_ev":     skipped_max_ev,
        "skipped_elo":        skipped_elo,
        "skipped_no_history": skipped_no_history,
        "skipped_club_exposure": skipped_club_exposure,
        "mean_shrinkage":   round(float(np.mean(shrinkages)), 3) if shrinkages else 0.0,
        "mean_sim_factor":  round(float(np.mean(sim_factors)), 3) if sim_factors else 1.0,
        "mean_dd_factor":   round(float(np.mean(dd_factors_applied)), 3)
                             if dd_factors_applied else 1.0,
        "odds_source":      odds_source,
        "detect_source":    detect_source,
    }
    return log, summary


# ── Live odds (The Odds API — optional) ────────────────────────────────────────

_NAME_MAP = {
    "AFC Bournemouth":           "Bournemouth",
    "Brighton and Hove Albion":  "Brighton",
    "Leeds United":              "Leeds",
    "Manchester City":           "Man City",
    "Manchester United":         "Man United",
    "Newcastle United":          "Newcastle",
    "Nottingham Forest":         "Nott'm Forest",
    "Tottenham Hotspur":         "Tottenham",
    "West Ham United":           "West Ham",
    "Wolverhampton Wanderers":   "Wolves",
    "Sunderland AFC":            "Sunderland",
    "Ipswich Town":              "Ipswich",
    "Leicester City":            "Leicester",
}


def fetch_live_odds(api_key: str) -> dict:
    """
    Fetch EPL H2H + totals odds from The Odds API (https://the-odds-api.com).
    Free tier: 500 req/month. Capped at 400 with 6h local cache.

    Pulls UK + EU regions so Pinnacle (the sharp benchmark) is included alongside
    UK books. For each market the returned dict carries:
      - "H"/"D"/"A"/"over25"/"under25" : the *best* available price (Max)
      - "_pinnacle" : sub-dict with Pinnacle's quote per market when present
      - "_books"    : full per-book breakdown (used by clients that want to
                      pick a specific bookmaker for placement)

    Backwards-compat: existing callers reading top-level "H"/"D"/"A" keys still
    work — they just now get the best-of-panel price by default rather than the
    first-bookmaker-found price (this is strictly better for live placement).

    The ODDS_API_KEY environment variable takes precedence over the passed key.
    Every fresh (non-cache) fetch is also persisted to data/odds_snapshots/ to
    build a local line-movement history over the season.
    """
    api_key = resolve_odds_api_key(api_key)
    if not api_key:
        return {}

    if _LIVE_ODDS_CACHE.exists():
        age_h = (datetime.now().timestamp() - _LIVE_ODDS_CACHE.stat().st_mtime) / 3600
        if age_h < _CACHE_HOURS:
            try:
                raw = json.loads(_LIVE_ODDS_CACHE.read_text())
                return {tuple(k.split("|")): v for k, v in raw.items()}
            except Exception:
                pass

    if not _record_api_call():
        if _LIVE_ODDS_CACHE.exists():
            try:
                raw = json.loads(_LIVE_ODDS_CACHE.read_text())
                return {tuple(k.split("|")): v for k, v in raw.items()}
            except Exception:
                pass
        return {}

    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/",
            params={
                "apiKey": api_key.strip(),
                "regions": "uk,eu",          # eu region includes Pinnacle
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    # Fuzzy fallback for names the hardcoded map doesn't cover (promoted teams)
    from data import _resolve_team_name

    def _odds_name(raw: str) -> str:
        return _NAME_MAP.get(raw) or _resolve_team_name(raw)

    result: dict = {}
    for game in data:
        home = _odds_name(game.get("home_team", ""))
        away = _odds_name(game.get("away_team", ""))
        # Per-book quotes for this game: {bm_key: {"H": .., "D": .., "A": .., "over25": .., "under25": ..}}
        books: dict = {}
        for bm in game.get("bookmakers", []):
            bm_key = bm.get("key", "?")
            bo: dict = {}
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for outcome in mkt.get("outcomes", []):
                        fd    = _odds_name(outcome["name"])
                        price = float(outcome["price"])
                        if fd == home:                bo["H"] = price
                        elif fd == away:              bo["A"] = price
                        elif outcome["name"] == "Draw": bo["D"] = price
                elif mkt["key"] == "totals":
                    for outcome in mkt.get("outcomes", []):
                        if float(outcome.get("point", 0)) != 2.5:
                            continue
                        price = float(outcome["price"])
                        if outcome["name"] == "Over":   bo["over25"] = price
                        elif outcome["name"] == "Under": bo["under25"] = price
            if bo:
                books[bm_key] = bo

        if not books:
            continue

        # Build the merged shape: best price per market across all books
        merged: dict = {}
        for mkt_key in ("H", "D", "A", "over25", "under25"):
            quotes = [b[mkt_key] for b in books.values() if mkt_key in b]
            if quotes:
                merged[mkt_key] = round(max(quotes), 3)

        # Pinnacle quote if available — sharp benchmark
        pinn = books.get("pinnacle")
        if pinn:
            merged["_pinnacle"] = pinn
        merged["_books"] = books

        if {"H", "D", "A"}.issubset(merged.keys()):
            result[(home, away)] = merged

    try:
        DATA_DIR.mkdir(exist_ok=True)
        _LIVE_ODDS_CACHE.write_text(
            json.dumps({f"{h}|{a}": v for (h, a), v in result.items()})
        )
    except Exception:
        pass

    # Persist a timestamped snapshot of every fresh fetch — over a season this
    # builds a local line-movement history (model-vs-market drift, own closing
    # lines for markets Pinnacle doesn't close, earlier CLV signal).
    if result:
        try:
            _ODDS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            snap_path = _ODDS_SNAPSHOT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
            snap_path.write_text(json.dumps({
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "games": {f"{h}|{a}": v for (h, a), v in result.items()},
            }))
        except Exception:
            pass

    return result


# ═════════════════════════════════════════════════════════════════════════════
# RESEARCH-TRACK ADDITIONS — observability for the live portfolio + machinery
# for the parallel "Mock Portfolio Two". None of this changes existing model
# behaviour or stake sizing on the main portfolio.
# ═════════════════════════════════════════════════════════════════════════════


# ── Closing-line value (CLV) ────────────────────────────────────────────────

# Mapping of bet markets → ordered list of CSV columns to try for closing odds.
# Pinnacle close (PSH/PSD/PSA) is the gold-standard sharp benchmark; industry
# average (Avg*) is the fall-back when Pinnacle isn't recorded; B365 close is
# last resort. Football-data.co.uk publishes all three for every PL match.
_CLOSING_ODDS_COLUMNS = {
    "H":       ["PSH", "AvgH", "B365H"],
    "D":       ["PSD", "AvgD", "B365D"],
    "A":       ["PSA", "AvgA", "B365A"],
    "over25":  ["P>2.5", "Avg>2.5", "B365>2.5"],
    "under25": ["P<2.5", "Avg<2.5", "B365<2.5"],
}


def extract_closing_odds(home: str, away: str, date_str: str, market: str,
                         df: pd.DataFrame) -> float | None:
    """Look up closing odds from the football-data CSV for a finished match.
    Returns None if the match isn't found or no closing-odds column has a value.
    """
    try:
        md = pd.to_datetime(date_str).normalize()
    except Exception:
        return None
    rows = df[(df["HomeTeam"] == home) & (df["AwayTeam"] == away) & (df["Date"] == md)]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    for col in _CLOSING_ODDS_COLUMNS.get(market, []):
        if col not in row.index:
            continue
        try:
            v = float(row[col])
        except (ValueError, TypeError):
            continue
        if np.isnan(v) or v <= 1.0:
            continue
        return v
    return None


def compute_clv(taken_odds: float, closing_odds: float) -> float | None:
    """CLV as a fractional edge: taken_odds / closing_odds − 1.

    Positive CLV means we got a price *better* than the close (the standard
    sharp definition — beating the closing line is the strongest forward-
    looking signal of long-run profitability).
    """
    if not taken_odds or not closing_odds:
        return None
    if taken_odds <= 1 or closing_odds <= 1:
        return None
    return round(taken_odds / closing_odds - 1.0, 4)


def backfill_clv_for_settled_bets(p: dict, df: pd.DataFrame) -> int:
    """Add `closing_odds` and `clv` fields to every settled single bet that
    doesn't already have them. Returns count of bets newly tagged.

    Skips accumulators (multi-leg CLV is its own design) and pending bets
    (closing odds aren't in the CSV until the match has played).
    """
    n_updated = 0
    for bet in p["bets"]:
        if bet.get("status") not in ("won", "lost"):
            continue
        if bet.get("type") == "acca":
            continue
        if bet.get("clv") is not None and bet.get("closing_odds") is not None:
            continue
        closing = extract_closing_odds(
            bet["home"], bet["away"], bet["date"], bet["market"], df
        )
        if closing is None:
            continue
        bet["closing_odds"] = round(float(closing), 2)
        bet["clv"] = compute_clv(float(bet["odds"]), float(closing))
        n_updated += 1
    return n_updated


def clv_rolling(p: dict, windows=(5, 10, 20)) -> dict:
    """Rolling CLV across recently-settled tagged bets.

    Returns:
      windows  : {window_size: median_clv_over_last_N_bets}
      series   : list of (date, clv) tuples chronologically — for sparklines
      n        : total tagged settled bets
      alert    : "ok" | "watch" | "drift" — based on last-10 median vs zero
    """
    tagged = [b for b in p["bets"]
              if b.get("type") != "acca"
              and b.get("clv") is not None
              and b.get("status") in ("won", "lost")]
    if not tagged:
        return {"windows": {}, "series": [], "n": 0, "alert": "insufficient"}

    # Order chronologically by settlement / placement date
    def _key(b):
        return b.get("settled_at") or b.get("date") or b.get("created_at") or ""
    tagged_sorted = sorted(tagged, key=_key)
    series = [(_key(b)[:10], float(b["clv"])) for b in tagged_sorted]
    clvs   = np.array([c for _, c in series], dtype=float)

    out_windows = {}
    for w in windows:
        if len(clvs) >= w:
            recent = clvs[-w:]
            out_windows[w] = {
                "median":   float(np.median(recent)),
                "mean":     float(np.mean(recent)),
                "pct_pos":  float((recent > 0).mean() * 100),
                "n":        int(w),
            }
        else:
            out_windows[w] = None  # not enough bets

    # Alert: based on last-10 (or fall back to whatever we have)
    ref_window = out_windows.get(10) or out_windows.get(5)
    if ref_window is None:
        alert = "insufficient"
    elif ref_window["median"] < -0.005:   # below -0.5%
        alert = "drift"
    elif ref_window["median"] < 0.005:    # within ±0.5%
        alert = "watch"
    else:
        alert = "ok"

    return {
        "windows": out_windows,
        "series":  series,
        "n":       int(len(tagged)),
        "alert":   alert,
    }


def clv_summary(p: dict) -> dict:
    """Aggregate CLV stats across settled single bets that have a CLV tag."""
    tagged = [b for b in p["bets"]
              if b.get("type") != "acca"
              and b.get("clv") is not None
              and b.get("status") in ("won", "lost")]
    if not tagged:
        return {"n": 0, "median_clv": None, "mean_clv": None,
                "pct_positive": None, "by_market": {}}
    clvs = np.array([b["clv"] for b in tagged], dtype=float)
    by_market: dict = {}
    for b in tagged:
        m = b.get("market", "?")
        by_market.setdefault(m, []).append(b["clv"])
    return {
        "n":             int(len(clvs)),
        "median_clv":    float(np.median(clvs)),
        "mean_clv":      float(np.mean(clvs)),
        "pct_positive":  float((clvs > 0).mean() * 100),
        "by_market":     {m: {"n": len(v),
                              "median": float(np.median(v)),
                              "mean":   float(np.mean(v)),
                              "pct_pos": float((np.array(v) > 0).mean() * 100)}
                          for m, v in by_market.items()},
    }


# ── Calibration drift indicator ─────────────────────────────────────────────

def compute_brier_drift(p: dict, window_size: int = 20,
                        baseline_size: int = 40) -> dict:
    """Compare Brier score of the most recent `window_size` settled bets vs the
    `baseline_size` bets immediately before them. A meaningful positive delta
    means recent calibration has drifted worse — re-fit isotonic.

    Only single bets with a recorded `model_prob` are considered. Returns a
    dict the UI can render as a small banner.
    """
    settled = [b for b in p["bets"]
               if b.get("status") in ("won", "lost")
               and b.get("type") != "acca"
               and b.get("model_prob") is not None]
    settled = sorted(settled, key=lambda b: b.get("settled_at") or "")

    if len(settled) < window_size + 5:
        return {"recent_brier": None, "baseline_brier": None,
                "delta": None, "drift_signal": "insufficient",
                "n_recent": len(settled), "n_baseline": 0}

    def _brier(b: dict) -> float:
        actual = 1.0 if b["status"] == "won" else 0.0
        return (float(b["model_prob"]) - actual) ** 2

    recent = settled[-window_size:]
    if len(settled) >= window_size + baseline_size:
        baseline = settled[-(window_size + baseline_size):-window_size]
    else:
        baseline = settled[:-window_size]

    if not baseline:
        return {"recent_brier": float(np.mean([_brier(b) for b in recent])),
                "baseline_brier": None, "delta": None,
                "drift_signal": "insufficient_baseline",
                "n_recent": len(recent), "n_baseline": 0}

    recent_b   = float(np.mean([_brier(b) for b in recent]))
    baseline_b = float(np.mean([_brier(b) for b in baseline]))
    delta = recent_b - baseline_b

    if delta > 0.05:
        signal = "drift_worse"
    elif delta < -0.05:
        signal = "drift_better"
    else:
        signal = "stable"

    return {
        "recent_brier":   round(recent_b, 4),
        "baseline_brier": round(baseline_b, 4),
        "delta":          round(delta, 4),
        "drift_signal":   signal,
        "n_recent":       len(recent),
        "n_baseline":     len(baseline),
    }


# ── Per-bin probability-estimate variance (for Baker-McHale Kelly) ─────────

def compute_per_bin_variance(bt_df: pd.DataFrame, market: str,
                             n_bins: int = 8) -> list[dict]:
    """Compute the binomial-proportion variance of empirical frequency in each
    probability bin from the backtest. This is the Var(p̂) input for Baker-McHale
    uncertainty-shrunk Kelly.

    Returns a list of dicts: {lo, hi, n, freq, var}. Bins with no observations
    have freq=None and var=None.
    """
    pred_cols = {"H": "_dc_h", "D": "_dc_d", "A": "_dc_a"}
    act_cols  = {"H": "_act_h", "D": "_act_d", "A": "_act_a"}
    pcol = pred_cols.get(market)
    acol = act_cols.get(market)
    if (bt_df is None or bt_df.empty
            or not pcol or pcol not in bt_df.columns
            or not acol or acol not in bt_df.columns):
        return []

    preds = bt_df[pcol].values.astype(float)
    acts  = bt_df[acol].values.astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Use a half-open interval [lo, hi); include 1.0 in the last bucket.
        if hi == 1.0:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "freq": None, "var": None})
            continue
        f = float(acts[mask].mean())
        # Binomial-proportion variance: f*(1-f)/n
        bins.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "freq": f, "var": float(f * (1.0 - f) / max(n, 1))})
    return bins


def lookup_bin_variance(bins: list[dict], prob: float,
                        fallback: float = 0.01) -> float:
    """Find the bin that contains `prob` and return its Var(p̂).
    Falls back to `fallback` (~ s.d. 0.10 — moderate uncertainty) when the bin
    is empty or the prob is out of range.
    """
    for b in bins:
        if b["lo"] <= prob < b["hi"]:
            return float(b["var"]) if b.get("var") is not None else fallback
    # Probability == 1.0 lands in the last bin (closed at hi)
    if bins and prob >= bins[-1]["hi"]:
        return float(bins[-1]["var"]) if bins[-1].get("var") is not None else fallback
    return fallback


# ── Research-track Kelly variants ───────────────────────────────────────────

def kelly_stake_uncertainty_adjusted(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    var_p: float,
    fraction: float = 0.5,
    max_pct: float = 0.10,
) -> tuple[float, float]:
    """Kelly with Baker-McHale (2013) shrinkage for probability-estimate variance.

    Shrinkage k = 1 − Var(p̂) / (p̂(1−p̂)).
    When the bin is well-populated and p̂ is precise, k ≈ 1. When p̂ is uncertain
    relative to its maximum binomial variance, k → 0 and the stake collapses.

    Returns (stake_amount, shrinkage_used) so the caller can record both.
    """
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    if b <= 0 or model_prob <= 0:
        return 0.0, 0.0
    full_kelly = max(0.0, (model_prob * b - q) / b)
    p_var_max = model_prob * (1.0 - model_prob)
    if p_var_max <= 0:
        return 0.0, 0.0
    shrinkage = max(0.0, 1.0 - max(0.0, var_p) / p_var_max)
    capped = min(full_kelly * fraction * shrinkage, max_pct)
    return round(bankroll * capped, 2), round(shrinkage, 4)


def simultaneous_kelly_correction(kelly_pcts: list[float]) -> list[float]:
    """Adjust Kelly fractions for simultaneous (concurrently-settling) bets.

    For each bet i, scale by Π_{j≠i}(1 − kelly_j) — the bankroll cannot
    recompound mid-card, so each independent stake must be reduced for the
    capital tied up in the other bets. Approximation of Busseti/Ryu/Boyd's
    risk-constrained Kelly portfolio for the common Saturday-card case.
    """
    if not kelly_pcts:
        return []
    if len(kelly_pcts) == 1:
        return list(kelly_pcts)
    clamped = [max(0.0, min(1.0, k)) for k in kelly_pcts]
    out: list[float] = []
    for i, _ in enumerate(clamped):
        factor = 1.0
        for j, kj in enumerate(clamped):
            if j != i:
                factor *= (1.0 - kj)
        out.append(round(kelly_pcts[i] * factor, 6))
    return out


# ── Mock Portfolio Two (parallel research-track portfolio) ──────────────────

MOCK2_PORTFOLIO_FILE = DATA_DIR / "portfolio_two.json"

_MOCK2_DEFAULT: dict = {
    "initial_bankroll": 10000.0,
    "bankroll":         10000.0,
    "bets":             [],
    "settings": {
        "min_ev":               0.03,
        "max_stake_pct":        0.10,
        "kelly_fraction":       0.5,    # base × Baker-McHale shrinkage
        "auto_bet_enabled":     False,
        "auto_bet_threshold":   0.03,
        "auto_markets":         ["D", "under25"],
        "min_prob":             0.22,
        "use_calibrated_probs": True,
        # Research-track switches (Mock Two only)
        "use_kn_model":         True,
        "use_uncertainty_kelly": True,
        "use_simultaneous_kelly": True,
        "model_variant":        "dixon-coles-kn",
        "min_team_matches":     6,      # promoted sides aren't rated yet
    },
}


def load_portfolio_two() -> dict:
    """Load Mock Portfolio Two state, back-filling missing settings."""
    if MOCK2_PORTFOLIO_FILE.exists():
        try:
            data = json.loads(MOCK2_PORTFOLIO_FILE.read_text())
            data.setdefault("settings", {})
            for k, v in _MOCK2_DEFAULT["settings"].items():
                data["settings"].setdefault(k, v)
            return data
        except Exception:
            pass
    return {
        "initial_bankroll": _MOCK2_DEFAULT["initial_bankroll"],
        "bankroll":         _MOCK2_DEFAULT["bankroll"],
        "bets":             [],
        "settings":         dict(_MOCK2_DEFAULT["settings"]),
    }


def save_portfolio_two(p: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MOCK2_PORTFOLIO_FILE.write_text(json.dumps(p, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Mock Two — Phase 2 filter & sizing helpers
# (driven by Phase 1 diagnostic in scripts/diagnose_main_losses.py)
# ─────────────────────────────────────────────────────────────────────────────

# Day-of-week + month abbreviation maps (datetime.strftime gives these)
_DOW_ABBR   = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
_MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def compute_team_roi_from_bets(
    bets: list[dict],
    market_filter: set[str] | None = None,
    min_n: int = 3,
) -> dict[str, dict]:
    """Aggregate per-team P&L from a list of settled bet dicts.

    Returns {team: {"n": int, "roi": float (pct), "profit": float, "stake": float}}
    for each team that appears in ≥ min_n settled bets. Includes both home- and
    away-side appearances, since for draw bets neither team is "the bet" — only
    the matchup matters.

    Used by `should_skip_team_pair` to power Mock Two's team-ROI filter. The
    caller is responsible for ensuring `bets` was computed *before* the period
    being filtered (no look-ahead). For backtests, that means bets from prior
    folds; for live, bets from `p["bets"]` already settled.
    """
    agg: dict[str, dict] = {}
    for b in bets:
        if b.get("status") not in ("won", "lost"):
            continue
        if market_filter and b.get("market") not in market_filter:
            continue
        for team in (b.get("home"), b.get("away")):
            if not team:
                continue
            g = agg.setdefault(team, {"n": 0, "stake": 0.0, "profit": 0.0})
            g["n"] += 1
            g["stake"] += float(b.get("stake") or 0.0)
            g["profit"] += float(b.get("profit") or 0.0)

    out: dict[str, dict] = {}
    for team, g in agg.items():
        if g["n"] < min_n:
            continue
        roi = (g["profit"] / g["stake"] * 100.0) if g["stake"] > 0 else 0.0
        out[team] = {
            "n":      g["n"],
            "roi":    round(roi, 2),
            "profit": round(g["profit"], 2),
            "stake":  round(g["stake"], 2),
        }
    return out


def should_skip_team_pair(
    team_roi: dict[str, dict] | None,
    home: str,
    away: str,
    threshold_pct: float = -25.0,
) -> bool:
    """Return True when either team has historical ROI < threshold_pct.

    `team_roi` is the output of `compute_team_roi_from_bets` — only teams with
    n_bets ≥ min_n appear, so threshold checks are statistically meaningful.
    Teams not in the table (insufficient sample) are NEVER blacklisted —
    we err toward letting bets through rather than filtering on noise.
    """
    if not team_roi:
        return False
    for team in (home, away):
        info = team_roi.get(team)
        if info and info["roi"] < threshold_pct:
            return True
    return False


def kelly_drawdown_throttle(
    bankroll: float,
    peak_bankroll: float,
    throttle_at_pct: float = 0.20,
    min_factor: float = 0.25,
) -> float:
    """Linear stake throttle as a function of current drawdown from peak.

    At 0% drawdown:        returns 1.0 (full Kelly).
    At throttle_at_pct DD: returns min_factor (e.g. 0.25 = quarter-Kelly).
    Below min_factor:      clamped to min_factor (never goes lower).

    Phase 1 found Main's max DD = 65% — full Kelly was too aggressive even
    when the long-run edge was real. This caps the bet-size cliff during
    losing runs without the regime-detection complexity of ADWIN.
    """
    if peak_bankroll <= 0 or bankroll <= 0:
        return 1.0
    dd = max(0.0, (peak_bankroll - bankroll) / peak_bankroll)
    if dd <= 0:
        return 1.0
    if dd >= throttle_at_pct:
        return min_factor
    # Linear interpolation between (0, 1.0) and (throttle_at_pct, min_factor)
    return 1.0 - (1.0 - min_factor) * (dd / throttle_at_pct)


def should_skip_xg_overperform(
    home_roll_gf: float | None,
    home_roll_xg: float | None,
    away_roll_gf: float | None,
    away_roll_xg: float | None,
    threshold: float | None = None,
) -> bool:
    """True when EITHER team is over-performing their xG by > threshold.

    Rationale: teams with rolling actual goals significantly > rolling xG
    are "lucky scorers" who tend to regress. Betting on draws when a team
    is on a hot finishing streak is risky — they're either about to keep
    scoring (no draw) or about to cool off dramatically (no goals at all,
    but their opponent might score).

    Both directions of over-/under-performance are skipped because both
    indicate the model's standard probabilities are likely miscalibrated.

    threshold: e.g. 0.25 means skip when |actual − xG| / xG > 25%
    None → filter inactive.
    """
    if threshold is None:
        return False
    for gf, xg in ((home_roll_gf, home_roll_xg),
                   (away_roll_gf, away_roll_xg)):
        if gf is None or xg is None:
            continue
        try:
            gf_f = float(gf); xg_f = float(xg)
        except (TypeError, ValueError):
            continue
        if pd.isna(gf_f) or pd.isna(xg_f) or xg_f <= 0.05:
            continue  # avoid div-by-zero / nonsense small xG
        luck = abs(gf_f - xg_f) / xg_f
        if luck > threshold:
            return True
    return False


def should_skip_elo_profile(
    home_elo: float | None,
    away_elo: float | None,
    min_team_elo: float | None = None,
    max_team_elo: float | None = None,
    elo_gap_min: float | None = None,
    elo_gap_max: float | None = None,
    require_known_elo: bool = False,
) -> bool:
    """True when the ELO profile of this matchup falls outside the allowed
    bands. Each filter is independent — set any to None to disable.

      min_team_elo: BOTH teams must be ≥ this (skip very-weak fixtures)
      max_team_elo: BOTH teams must be ≤ this (skip top-vs-top fixtures)
      elo_gap_min:  |home_elo - away_elo| must be ≥ this (skip too-close)
      elo_gap_max:  |home_elo - away_elo| must be ≤ this (skip too-lopsided)

    Unknown ELO (None / NaN) → never skip, on the principle of "don't filter on
    missing data" — *unless* `require_known_elo` is set.

    That principle is backwards for a floor whose job is to exclude weak and
    unrated sides: a promoted team has no entry in the ratings dict at all, so
    permissive handling of missing data waves through exactly the fixture the
    floor exists to stop. `require_known_elo=True` inverts it, skipping when a
    band is active and either rating is unavailable.
    """
    band_active = any(v is not None for v in
                      (min_team_elo, max_team_elo, elo_gap_min, elo_gap_max))

    def _unknown() -> bool:
        return bool(require_known_elo and band_active)

    # Bail on missing data
    if home_elo is None or away_elo is None:
        return _unknown()
    try:
        h = float(home_elo); a = float(away_elo)
    except (TypeError, ValueError):
        return _unknown()
    if pd.isna(h) or pd.isna(a):
        return _unknown()

    if min_team_elo is not None and (h < min_team_elo or a < min_team_elo):
        return True
    if max_team_elo is not None and (h > max_team_elo or a > max_team_elo):
        return True

    gap = abs(h - a)
    if elo_gap_min is not None and gap < elo_gap_min:
        return True
    if elo_gap_max is not None and gap > elo_gap_max:
        return True
    return False


def should_skip_unrated(
    home: str,
    away: str,
    match_counts: dict[str, int] | None,
    min_matches: int | None,
) -> bool:
    """True when either side has too little top-flight history to be rated.

    A promoted team carries no Dixon-Coles attack/defence rating and no Elo
    until it has played enough matches for the model to learn them, so any
    probability quoted for its fixtures is prior, not signal. Sizing full Kelly
    off that is how the 2025-26 sweep produced a £100k result that wasn't real.

    A team missing from `match_counts` counts as zero matches and is skipped.
    Passing no counts at all disables the gate, matching
    `should_skip_elo_profile`'s stance on absent data.
    """
    if not min_matches or match_counts is None:
        return False
    return (match_counts.get(home, 0) < min_matches
            or match_counts.get(away, 0) < min_matches)


def _season_start(p: dict) -> str | None:
    """First day of the portfolio's season, as an ISO date.

    Used to scope the club-exposure cap to the current season. July is the
    boundary the rest of the app uses for season rollover. Returns None when the
    portfolio predates the season field, which leaves the count unscoped.
    """
    season = str(p.get("season", "") or "")
    head = season.split("-")[0]
    return f"{head}-07-01" if head.isdigit() and len(head) == 4 else None


def club_bet_counts(bets: list[dict], since: str | None = None) -> dict[str, int]:
    """How many bets each club already carries, home plus away.

    Voided bets never carried risk, so they don't use up a club's allowance.
    `since` limits the count to one season: pass the season's start date and
    last year's run on the same club stops suppressing this year's.
    """
    counts: dict[str, int] = {}
    for b in bets:
        if b.get("status") == "void":
            continue
        if since and str(b.get("date", ""))[:10] < since:
            continue
        for club in (b.get("home"), b.get("away")):
            if club:
                counts[club] = counts.get(club, 0) + 1
    return counts


def should_skip_club_exposure(home: str, away: str,
                              club_counts: dict[str, int] | None,
                              max_per_club: int | None) -> bool:
    """True when either side already carries the season's full allowance.

    A risk control, not a profit filter. The 2026-27 re-validation found 11 of
    14 bets in 2025-26 landed on Sunderland, whose +£66,577 was the entire
    season's profit and then some: every other club combined lost £15k. A season
    that rides on one club is one club's variance, however good the model looks.

    Both clubs are checked because a bet exposes both. Passing no cap disables
    the control, so existing portfolios are unaffected.
    """
    if not max_per_club or club_counts is None:
        return False
    return (club_counts.get(home, 0) >= max_per_club
            or club_counts.get(away, 0) >= max_per_club)


def _rated_from_table(df, min_matches: int | None) -> dict | None:
    """Lazy wrapper so the simulators can build the table without importing
    `data` at module scope (data imports nothing from portfolio, but keeping
    the dependency one-way and local avoids any future cycle)."""
    if not min_matches or df is None or getattr(df, "empty", True):
        return None
    from data import team_rated_from
    return team_rated_from(df, int(min_matches))


def should_skip_unrated_at(
    home: str,
    away: str,
    when,
    rated_from: dict | None,
    min_matches: int | None,
) -> bool:
    """Point-in-time twin of `should_skip_unrated`, for the backtest simulators.

    `rated_from` comes from `data.team_rated_from()` and maps each team to the
    date it reached the threshold. A fixture qualifies only when both sides
    reached it *strictly before* kickoff — on the qualifying date itself that
    match has not been played yet.
    """
    if not min_matches or rated_from is None:
        return False
    try:
        kickoff = pd.Timestamp(when)
    except (TypeError, ValueError):
        return False
    for team in (home, away):
        reached = rated_from.get(team)
        if reached is None or not pd.Timestamp(reached) < kickoff:
            return True
    return False


def _log_skip(skip_log: list | None, reason: str, home: str, away: str,
              detail: str) -> None:
    """Record why a fixture was passed over, so skips aren't silent."""
    if skip_log is None:
        return
    skip_log.append({"reason": reason, "home": home, "away": away,
                     "detail": detail})


def _unrated_detail(home: str, away: str, match_counts: dict[str, int] | None,
                    min_matches: int) -> str:
    counts = match_counts or {}
    short = [f"{t} ({counts.get(t, 0)})" for t in (home, away)
             if counts.get(t, 0) < min_matches]
    return (f"{' and '.join(short)} below {min_matches} Premier League "
            f"matches in the dataset")


def _should_skip_calendar(date, banned_dows: set[str] | None,
                          banned_months: set[str] | None) -> bool:
    """True if date falls on a banned DOW or month abbreviation."""
    if not banned_dows and not banned_months:
        return False
    try:
        ts = pd.to_datetime(date)
    except Exception:
        return False
    if banned_dows:
        dow = _DOW_ABBR.get(ts.weekday())
        if dow in banned_dows:
            return True
    if banned_months:
        mon = _MONTH_ABBR.get(ts.month)
        if mon in banned_months:
            return True
    return False


def auto_place_value_bets_v2(
    p: dict,
    candidates: list[dict],
    threshold: float,
    max_auto_bets: int = 3,
    calibrators: dict | None = None,
    bin_variances: dict | None = None,
    match_counts: dict[str, int] | None = None,
    skip_log: list | None = None,
) -> list[dict]:
    """Research-track auto-bet for Mock Portfolio Two.

    Differences vs main `auto_place_value_bets`:
      - Kelly stake is shrunk by Baker-McHale per-bin variance (when available).
      - When multiple candidates qualify in the same call, stakes are scaled
        by the simultaneous-bet correction (capital tied up elsewhere).
      - Each placed bet records `v2_kelly_shrinkage`, `v2_var_p`, and
        `v2_sim_factor` for later auditing.

    `bin_variances`: dict {market_code: list_of_bin_dicts} from
    `compute_per_bin_variance`. Optional — when missing, falls back to no
    shrinkage (acts like vanilla half-Kelly).

    Each candidate is the same shape as for the main auto-bet. Returns the
    list of newly placed bets.
    """
    placed: list[dict] = []
    kelly_frac    = float(p["settings"].get("kelly_fraction", 0.5))
    max_stake_pct = float(p["settings"].get("max_stake_pct", 0.10))
    allowed_mkts  = set(p["settings"].get("auto_markets", list(PROFITABLE_MARKETS)))
    min_prob      = float(p["settings"].get("min_prob", 0.22))
    use_cal       = bool(p["settings"].get("use_calibrated_probs", True))
    use_uncert    = bool(p["settings"].get("use_uncertainty_kelly", True))
    use_sim       = bool(p["settings"].get("use_simultaneous_kelly", True))

    # ── Mock Two Phase 2 filters (defaults: all OFF until WF-validated) ──
    settings_v2 = p["settings"]
    team_filter_on = bool(settings_v2.get("v2_team_roi_filter", False))
    team_threshold = float(settings_v2.get("v2_team_roi_threshold", -25.0))
    team_min_n     = int(settings_v2.get("v2_team_roi_min_n", 3))
    max_ev_pct     = settings_v2.get("v2_max_ev_pct")  # None = no cap
    max_ev_pct     = float(max_ev_pct) if max_ev_pct is not None else None
    banned_dows    = set(settings_v2.get("v2_banned_dows", []))
    banned_months  = set(settings_v2.get("v2_banned_months", []))
    dd_on          = bool(settings_v2.get("v2_drawdown_throttle", False))
    dd_at          = float(settings_v2.get("v2_drawdown_at_pct", 0.20))
    dd_min         = float(settings_v2.get("v2_drawdown_min_factor", 0.25))
    # ELO-profile filter (multi-season grid winner: min_team_elo=1500)
    min_team_elo   = settings_v2.get("v2_min_team_elo")
    max_team_elo   = settings_v2.get("v2_max_team_elo")
    elo_gap_min    = settings_v2.get("v2_elo_gap_min")
    elo_gap_max    = settings_v2.get("v2_elo_gap_max")
    elo_filter_on  = any(v is not None for v in
                         (min_team_elo, max_team_elo, elo_gap_min, elo_gap_max))

    # Pre-compute team-ROI from this portfolio's settled history
    # (only used for filtering, never for bet attribution)
    team_roi_table = (compute_team_roi_from_bets(p.get("bets", []),
                                                 market_filter=allowed_mkts,
                                                 min_n=team_min_n)
                      if team_filter_on else None)

    pending_singles = [b for b in p["bets"]
                       if b["status"] == "pending" and b.get("type") != "acca"]
    pending_stake = sum(b["stake"] for b in p["bets"] if b["status"] == "pending")
    max_exposure  = p["initial_bankroll"] * 0.50

    if len(pending_singles) >= 5:
        return []

    # Pass 1 — enrich each candidate with calibrated prob, EV, variance, and
    # an *un-corrected* Kelly fraction-of-bankroll.
    enriched: list[dict] = []
    for c in candidates:
        raw_prob = float(c.get("model_prob", 0.0))
        eff_prob = (calibrate_prob(raw_prob, c.get("market", ""), calibrators)
                    if use_cal else raw_prob)
        _odds_raw = c.get("odds")
        odds = float(_odds_raw) if _odds_raw is not None else 0.0
        # Detect_odds = sharp benchmark for EV gate (e.g. Pinnacle). Falls back
        # to placement odds when not available — preserves prior behaviour.
        _detect_raw = c.get("detect_odds")
        try:
            detect_odds = float(_detect_raw) if _detect_raw is not None and float(_detect_raw) > 1 else odds
        except (ValueError, TypeError):
            detect_odds = odds
        # EV against the sharp price; Kelly stake against the placement price
        eff_ev = compute_ev(eff_prob, detect_odds) if detect_odds > 1 else -1.0

        var_p = None
        shrink = 1.0
        if use_uncert and bin_variances:
            bins = bin_variances.get(c.get("market", ""), [])
            if bins:
                var_p = lookup_bin_variance(bins, eff_prob)
                p_var_max = eff_prob * (1.0 - eff_prob)
                if p_var_max > 0:
                    shrink = max(0.0, 1.0 - max(0.0, var_p) / p_var_max)

        if odds > 1 and eff_prob > 0:
            b_ = odds - 1.0
            q_ = 1.0 - eff_prob
            full_kelly = max(0.0, (eff_prob * b_ - q_) / b_)
        else:
            full_kelly = 0.0
        kelly_pct = min(full_kelly * kelly_frac * shrink, max_stake_pct)

        enriched.append({**c,
                         "_eff_prob":   eff_prob,
                         "_eff_ev":     eff_ev,
                         "_raw_prob":   raw_prob,
                         "_var_p":      var_p,
                         "_shrink":     shrink,
                         "_kelly_pct":  kelly_pct,
                         "_detect_odds": detect_odds,
                         "_used_sharp_detect": detect_odds != odds})

    # Pass 2 — apply gates and pick the qualifying set, ranked by EV
    skip_late = bool(p["settings"].get("skip_late_season", False))
    market_gates = p["settings"].get("market_gates")
    min_team_matches = p["settings"].get("min_team_matches")
    max_bets_per_club = p["settings"].get("max_bets_per_club")
    club_counts = (club_bet_counts(p["bets"], since=_season_start(p))
                   if max_bets_per_club else None)
    qualifying: list[dict] = []
    for c in sorted(enriched, key=lambda x: x["_eff_ev"], reverse=True):
        if len(qualifying) >= max_auto_bets:
            break
        if not c.get("odds") or c["odds"] <= 1:
            continue
        # NaN guard (parity with ev_backtest_simulate_v2): NaN<threshold is
        # False, so a NaN EV/prob would sail through the gates below.
        if pd.isna(c["_eff_ev"]) or pd.isna(c["_eff_prob"]):
            continue
        # No-history gate: promoted sides the model has never rated
        if should_skip_unrated(c["home"], c["away"], match_counts, min_team_matches):
            _log_skip(skip_log, "no_history", c["home"], c["away"],
                      _unrated_detail(c["home"], c["away"], match_counts,
                                      int(min_team_matches)))
            continue
        # Club-exposure cap: don't let one club carry the season
        if should_skip_club_exposure(c["home"], c["away"], club_counts,
                                     max_bets_per_club):
            _log_skip(skip_log, "club_exposure", c["home"], c["away"],
                      f"already {club_counts.get(c['home'], 0)}/"
                      f"{club_counts.get(c['away'], 0)} bets this season "
                      f"(cap {max_bets_per_club})")
            continue
        if club_counts is not None:
            for club in (c["home"], c["away"]):
                club_counts[club] = club_counts.get(club, 0) + 1
        # Late-season cutoff (Mar-May): empirically 0/7 across 2024-25 + 2025-26
        if skip_late and c.get("date") and _is_late_season(c["date"]):
            continue
        # v2: calendar filter (Mon/Fri/Oct etc.)
        if _should_skip_calendar(c.get("date"), banned_dows, banned_months):
            continue
        # v2: team-ROI blacklist
        if team_filter_on and should_skip_team_pair(
                team_roi_table, c["home"], c["away"], team_threshold):
            continue
        # v2: max-EV cap (model overconfidence guard)
        if max_ev_pct is not None and c["_eff_ev"] > max_ev_pct:
            continue
        # v2: ELO-profile filter (multi-season grid winner)
        if elo_filter_on and should_skip_elo_profile(
                c.get("home_elo"), c.get("away_elo"),
                min_team_elo, max_team_elo, elo_gap_min, elo_gap_max,
                require_known_elo=True):
            _log_skip(skip_log, "elo_profile", c["home"], c["away"],
                      f"home_elo={c.get('home_elo')} away_elo={c.get('away_elo')} "
                      f"outside band (floor {min_team_elo})")
            continue
        if c["market"] not in allowed_mkts:
            continue
        if c["_eff_prob"] < _gate(market_gates, c["market"], "min_prob", min_prob):
            continue
        if c["_eff_ev"] < _gate(market_gates, c["market"], "min_ev", threshold):
            continue
        if any(b.get("type") != "acca" and b["status"] == "pending"
               and b["home"] == c["home"] and b["away"] == c["away"]
               and b["market"] == c["market"]
               for b in p["bets"]):
            continue
        if c["_kelly_pct"] <= 0:
            continue
        qualifying.append(c)

    # Pass 3 — apply simultaneous-bet correction across the qualifying set
    if use_sim and len(qualifying) > 1:
        original_pcts = [q["_kelly_pct"] for q in qualifying]
        corrected = simultaneous_kelly_correction(original_pcts)
        for q, new_pct, orig in zip(qualifying, corrected, original_pcts):
            q["_sim_factor"] = round(new_pct / orig, 4) if orig > 0 else 1.0
            q["_kelly_pct"]  = new_pct
    else:
        for q in qualifying:
            q["_sim_factor"] = 1.0

    # Compute drawdown throttle factor (constant across this call — stake
    # decisions all happen with the same peak/current bankroll snapshot)
    if dd_on:
        peak_br = max(float(p.get("peak_bankroll", p["bankroll"])),
                      float(p["bankroll"]))
        dd_factor = kelly_drawdown_throttle(p["bankroll"], peak_br, dd_at, dd_min)
    else:
        dd_factor = 1.0

    # Pass 4 — place bets in order
    for c in qualifying:
        stake = round(p["bankroll"] * c["_kelly_pct"] * dd_factor, 2)
        if stake <= 0 or stake > p["bankroll"]:
            continue
        if pending_stake >= max_exposure:
            break
        stake = min(stake, max_exposure - pending_stake)
        if stake <= 0:
            break
        bet = place_bet(p, c["home"], c["away"], c["date"],
                        c["market"], c["selection"],
                        c["_raw_prob"], c["odds"], stake)
        # Research-track audit fields
        bet["v2_kelly_shrinkage"] = (round(float(c["_shrink"]), 4)
                                     if c.get("_shrink") is not None else None)
        bet["v2_var_p"]           = (round(float(c["_var_p"]), 6)
                                     if c.get("_var_p") is not None else None)
        bet["v2_sim_factor"]      = round(float(c.get("_sim_factor", 1.0)), 4)
        bet["v2_dd_factor"]       = round(float(dd_factor), 4)
        bet["model_variant"]      = p["settings"].get("model_variant", "dixon-coles-kn")
        placed.append(bet)
        pending_stake += stake

    return placed
