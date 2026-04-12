"""
Mock betting portfolio for F_PRED — EV-based paper trading.
No real money involved. For analysis and model validation only.
"""
from __future__ import annotations

import json
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

_DEFAULT: dict = {
    "initial_bankroll": 1000.0,
    "bankroll": 1000.0,
    "bets": [],
    "settings": {
        "min_ev": 0.05,
        "max_stake_pct": 0.10,
        "kelly_fraction": 0.5,
        "odds_api_key": "",
        "auto_bet_enabled": False,
        "auto_bet_threshold": 0.15,   # 15% EV to auto-place
        "auto_markets": ["D", "over25", "under25"],  # only bet markets with proven edge
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
                          max_auto_bets: int = 3) -> list[dict]:
    """
    Automatically place Kelly-sized bets for every candidate with EV >= threshold.
    Skips duplicates (already have a pending bet on that home/away/market).
    Caps total pending exposure at 50% of initial bankroll.
    Caps auto-bets per run at max_auto_bets.

    Each candidate dict: {home, away, date, market, selection, model_prob, odds, ev}
    Returns list of newly placed bets.
    """
    placed = []
    kelly_frac    = float(p["settings"].get("kelly_fraction", 0.5))
    max_stake_pct = float(p["settings"].get("max_stake_pct", 0.10))
    allowed_mkts  = set(p["settings"].get("auto_markets", list(PROFITABLE_MARKETS)))

    # Exposure guard: don't let total pending stake exceed 50% of initial bankroll
    pending_singles = [b for b in p["bets"] if b["status"] == "pending" and b.get("type") != "acca"]
    pending_stake = sum(b["stake"] for b in p["bets"] if b["status"] == "pending")
    max_exposure  = p["initial_bankroll"] * 0.50

    # Don't pile on if we already have enough pending singles
    if len(pending_singles) >= 5:
        return []

    for c in sorted(candidates, key=lambda x: x["ev"], reverse=True):
        if len(placed) >= max_auto_bets:
            break
        if not c.get("odds") or c["odds"] <= 1:
            continue
        # Only bet on markets where the model has proven edge
        if c["market"] not in allowed_mkts:
            continue
        if c["ev"] < threshold:
            continue
        # Duplicate guard
        if any(
            b.get("type") != "acca"
            and b["status"] == "pending"
            and b["home"] == c["home"] and b["away"] == c["away"]
            and b["market"] == c["market"]
            for b in p["bets"]
        ):
            continue

        # Check exposure before placing
        if pending_stake >= max_exposure:
            break

        stake = kelly_stake_amount(
            c["model_prob"], c["odds"], p["bankroll"], kelly_frac, max_stake_pct
        )
        if stake <= 0 or stake > p["bankroll"]:
            continue
        # Don't exceed remaining exposure room
        stake = min(stake, max_exposure - pending_stake)
        if stake <= 0:
            break

        bet = place_bet(
            p, c["home"], c["away"], c["date"],
            c["market"], c["selection"],
            c["model_prob"], c["odds"], stake,
        )
        placed.append(bet)
        pending_stake += stake

    return placed


# ── Historical EV simulation ────────────────────────────────────────────────────

def ev_backtest_simulate(
    bt_df: pd.DataFrame,
    df: pd.DataFrame,
    min_ev_pct: float = 5.0,
    kelly_frac: float = 0.5,
    max_stake_pct: float = 0.10,
    initial_bankroll: float = 1000.0,
    allowed_markets: set | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate EV-based Kelly betting using DC+XGB model probabilities vs actual
    Bet365 odds from historical CSV data.

    bt_df:  output of backtest_models() — must have DC_H, DC_D, DC_A (0-100 %) columns.
    df:     full match DataFrame with B365H, B365D, B365A columns.
    Returns (bet_log_df, summary_dict).
    """
    if bt_df.empty:
        return pd.DataFrame(), {"error": "Backtest returned no data"}

    min_ev = min_ev_pct / 100.0
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(), {"error": "B365 odds not found in dataset"}

    # Optional O/U columns
    has_ou = "B365>2.5" in df.columns and "B365<2.5" in df.columns and "DC_O25" in bt_df.columns
    ou_cols = (["B365>2.5", "B365<2.5"] if has_ou else [])

    odds_df = (
        df[needed + ou_cols]
        .rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})
        .dropna(subset=["B365H", "B365D", "B365A"])
    )
    merged = bt_df.merge(odds_df, on=["Date", "Home", "Away"], how="inner")

    if merged.empty:
        return pd.DataFrame(), {
            "error": "No test matches have B365 odds — try a longer test window"
        }

    bankroll = initial_bankroll
    rows = []

    for _, r in merged.sort_values("Date").iterrows():
        try:
            b365h = float(r["B365H"])
            b365d = float(r["B365D"])
            b365a = float(r["B365A"])
        except (ValueError, TypeError):
            continue
        if b365h <= 1 or b365d <= 1 or b365a <= 1:
            continue

        actual = r["Actual"]
        total_goals = int(r.get("FTHG", 0) or 0) + int(r.get("FTAG", 0) or 0)
        p_h = float(r["DC_H"]) / 100.0
        p_d = float(r["DC_D"]) / 100.0
        p_a = float(r["DC_A"]) / 100.0

        market_list = [
            ("H", p_h, b365h, "Home Win"),
            ("D", p_d, b365d, "Draw"),
            ("A", p_a, b365a, "Away Win"),
        ]
        if has_ou:
            try:
                b365_o = float(r["B365>2.5"])
                b365_u = float(r["B365<2.5"])
                p_o25  = float(r["DC_O25"]) / 100.0
                if b365_o > 1 and b365_u > 1:
                    actual_ou_o = total_goals > 2
                    market_list += [
                        ("over25",  p_o25,       b365_o, "Over 2.5"),
                        ("under25", 1.0 - p_o25, b365_u, "Under 2.5"),
                    ]
            except (ValueError, TypeError, KeyError):
                pass

        for mkt, prob, odds, outcome in market_list:
            if allowed_markets and mkt not in allowed_markets:
                continue
            ev_val = compute_ev(prob, odds)
            if ev_val < min_ev:
                continue
            stake = kelly_stake_amount(prob, odds, bankroll, kelly_frac, max_stake_pct)
            if stake <= 0 or stake > bankroll:
                continue
            bankroll -= stake
            if mkt in ("over25", "under25"):
                won = (total_goals > 2) if mkt == "over25" else (total_goals <= 2)
            else:
                won = actual == outcome
            if won:
                ret = round(stake * odds, 2)
                profit = round(ret - stake, 2)
                bankroll = round(bankroll + ret, 2)
            else:
                profit = round(-stake, 2)
            rows.append({
                "Date":     r["Date"],
                "Match":    f"{r['Home']} vs {r['Away']}",
                "Market":   {"H": "Home Win", "D": "Draw", "A": "Away Win",
                             "over25": "Over 2.5", "under25": "Under 2.5"}.get(mkt, mkt),
                "Model%":   f"{prob*100:.1f}%",
                "Implied%": f"{100/odds:.1f}%",
                "EV":       f"+{ev_val*100:.1f}%",
                "Odds":     round(odds, 2),
                "Stake":    round(stake, 2),
                "Result":   "✅" if won else "❌",
                "Profit":   round(profit, 2),
                "Bankroll": round(bankroll, 2),
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
        "initial":      initial_bankroll,
        "final":        round(bankroll, 2),
        "profit":       round(profit_total, 2),
        "roi":          round(profit_total / staked_total * 100, 2) if staked_total > 0 else 0.0,
        "n_bets":       len(log),
        "win_rate":     round(n_won / len(log) * 100, 1),
        "avg_odds":     round(float(log["Odds"].mean()), 2),
        "total_staked": round(staked_total, 2),
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
    Fetch EPL H2H odds from The Odds API (https://the-odds-api.com).
    Free tier: 500 req/month. Capped at 400 with 6h local cache.
    Returns {(home_fd_name, away_fd_name): {"H": float, "D": float, "A": float}}
    """
    if not api_key.strip():
        return {}

    # Serve from cache if fresh enough
    if _LIVE_ODDS_CACHE.exists():
        age_h = (datetime.now().timestamp() - _LIVE_ODDS_CACHE.stat().st_mtime) / 3600
        if age_h < _CACHE_HOURS:
            try:
                raw = json.loads(_LIVE_ODDS_CACHE.read_text())
                return {tuple(k.split("|")): v for k, v in raw.items()}
            except Exception:
                pass

    # Check monthly cap before making a live request
    if not _record_api_call():
        # Over cap — serve stale cache if available, else empty
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
                "regions": "uk",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    result: dict = {}
    for game in data:
        home = _NAME_MAP.get(game.get("home_team", ""), game.get("home_team", ""))
        away = _NAME_MAP.get(game.get("away_team", ""), game.get("away_team", ""))
        om: dict = {}
        for bm in game.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h" and len(om.get("H", [])) == 0:
                    for outcome in mkt.get("outcomes", []):
                        fd    = _NAME_MAP.get(outcome["name"], outcome["name"])
                        price = float(outcome["price"])
                        if fd == home:
                            om["H"] = price
                        elif fd == away:
                            om["A"] = price
                        elif outcome["name"] == "Draw":
                            om["D"] = price
                elif mkt["key"] == "totals":
                    for outcome in mkt.get("outcomes", []):
                        # Only grab the 2.5 line
                        if float(outcome.get("point", 0)) != 2.5:
                            continue
                        price = float(outcome["price"])
                        if outcome["name"] == "Over":
                            om["over25"] = price
                        elif outcome["name"] == "Under":
                            om["under25"] = price
            if len(om) >= 3:   # h2h at minimum
                break
        if len(om) >= 3:
            result[(home, away)] = om

    # Cache to disk
    try:
        DATA_DIR.mkdir(exist_ok=True)
        _LIVE_ODDS_CACHE.write_text(
            json.dumps({f"{h}|{a}": v for (h, a), v in result.items()})
        )
    except Exception:
        pass

    return result
