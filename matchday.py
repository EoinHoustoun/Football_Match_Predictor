"""What the live strategy would do with the next matchday, for display.

The app used to answer "would we bet this?" in three different ways: the Value
Bet Scanner repriced every market with no gates at all, Pre-Flight called any
rated fixture ELIGIBLE, and the auto-bet run was the only thing that knew the
truth. This module gives every page the same answer.

The yes/no comes from the REAL placement functions, run on a deep copy of the
portfolio (`dry_run_placed`), so a verdict can never disagree with what the
auto-bet does. `explain_block` only names the gate that stopped a fixture; if
it ever drifts from portfolio.py the worst case is a wrong reason label, never
a wrong verdict. tests/test_matchday.py pins the two together.

Nothing here writes to disk, logs, or changes a portfolio.
"""
from __future__ import annotations

import copy
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

import portfolio as pf

# The app-load run and the launchd runner both fetch fixtures 14 days ahead,
# so a fixture further out cannot be bet yet whatever its edge.
BETTING_WINDOW_DAYS = 14

MARKET_LABELS = {
    "H": "Home win", "D": "Draw", "A": "Away win",
    "over25": "Over 2.5", "under25": "Under 2.5",
}


# ── Candidates ────────────────────────────────────────────────────────────────

def build_candidates(preds: list[dict], live_odds_map: dict,
                     elo_dict: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Main and Mock Two candidates for each predicted fixture with odds.

    `preds` items: {home, away, date (ISO str), main: {home_win, draw,
    away_win}, mt: {...same...}, p_o25}. Same shape and logic as the
    auto-bet run in app.py, which now calls this too.
    """
    elo_dict = elo_dict or {}
    main_cands: list[dict] = []
    mt_cands: list[dict] = []
    for fx in preds:
        h, a = fx["home"], fx["away"]
        api_o = live_odds_map.get((h, a), {})
        if not api_o or "H" not in api_o:
            continue
        pinn = api_o.get("_pinnacle") if isinstance(api_o, dict) else None
        main_res, mt_res, p_o25 = fx["main"], fx["mt"], fx["p_o25"]
        for mkt, m_prob, mt_prob, lbl in [
            ("H",       main_res["home_win"], mt_res["home_win"], f"Home Win ({h})"),
            ("D",       main_res["draw"],     mt_res["draw"],     "Draw"),
            ("A",       main_res["away_win"], mt_res["away_win"], f"Away Win ({a})"),
            ("over25",  p_o25,                p_o25,              "Over 2.5 Goals"),
            ("under25", 1.0 - p_o25,          1.0 - p_o25,        "Under 2.5 Goals"),
        ]:
            place_o = api_o.get(mkt)
            if not place_o:
                continue
            detect_o = pinn.get(mkt) if pinn else None
            ev_ref = detect_o if (detect_o and detect_o > 1) else place_o
            base = dict(home=h, away=a, date=fx["date"], market=mkt, selection=lbl,
                        odds=place_o, detect_odds=detect_o,
                        home_elo=elo_dict.get(h), away_elo=elo_dict.get(a))
            main_cands.append({**base, "model_prob": m_prob,
                               "ev": pf.compute_ev(m_prob, ev_ref) if ev_ref else -1})
            mt_cands.append({**base, "model_prob": mt_prob,
                             "ev": pf.compute_ev(mt_prob, ev_ref) if ev_ref else -1})
    return main_cands, mt_cands


# ── Gate explanation ─────────────────────────────────────────────────────────

_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def candidate_numbers(p: dict, c: dict, calibrators: dict | None) -> dict:
    """Raw, calibrated, EV and Kelly share for one candidate, as the placer sees them."""
    s = p["settings"]
    raw = float(c.get("model_prob", 0.0))
    use_cal = bool(s.get("use_calibrated_probs", True))
    cal = pf.calibrate_prob(raw, c.get("market", ""), calibrators) if use_cal else raw
    odds = float(c.get("odds") or 0.0)
    try:
        det = float(c["detect_odds"]) if c.get("detect_odds") and float(c["detect_odds"]) > 1 else None
    except (TypeError, ValueError):
        det = None
    ref = det if det else odds
    ev = pf.compute_ev(cal, ref) if ref > 1 else -1.0
    if odds > 1 and cal > 0:
        b = odds - 1.0
        full_k = max(0.0, (cal * b - (1.0 - cal)) / b)
    else:
        full_k = 0.0
    return {"raw": raw, "cal": cal, "ev": ev, "odds": odds,
            "detect_odds": det, "full_kelly": full_k}


def explain_block(p: dict, c: dict, threshold: float,
                  calibrators: dict | None = None,
                  match_counts: dict | None = None,
                  line: str = "main") -> dict | None:
    """First gate that stops this candidate, or None if every gate passes.

    Returns {"code", "text"}. Covers the per-candidate gates of
    `auto_place_value_bets` (line="main") and `auto_place_value_bets_v2`
    (line="mt"). Run-level limits (3 bets per run, the 50% exposure cap) are
    not per-candidate, so a candidate can pass here and still not be placed.
    """
    s = p["settings"]
    n = candidate_numbers(p, c, calibrators)
    allowed = set(s.get("auto_markets", list(pf.PROFITABLE_MARKETS)))
    prefix = "main_" if line == "main" else "v2_"

    if c.get("market") not in allowed:
        return {"code": "market", "text": "Not a strategy market"}
    if not c.get("odds") or float(c["odds"]) <= 1:
        return {"code": "no_odds", "text": "No price yet"}
    if pd.isna(n["ev"]) or pd.isna(n["cal"]):
        return {"code": "nan", "text": "Model returned no probability"}
    if line == "mt":
        det_src, place_src = s.get("detect_source"), s.get("odds_source", "Max")
        if det_src and det_src != place_src and n["detect_odds"] is None:
            return {"code": "no_sharp", "text": f"No {det_src} price to measure edge"}
    if pf.should_skip_unrated(c["home"], c["away"], match_counts, s.get("min_team_matches")):
        return {"code": "no_history", "text": "Too little Premier League history"}
    max_club = s.get("max_bets_per_club")
    if max_club:
        counts = pf.club_bet_counts(p["bets"], since=pf._season_start(p))
        if pf.should_skip_club_exposure(c["home"], c["away"], counts, max_club):
            return {"code": "club_cap", "text": f"Club already has {max_club} bets this season"}
    if s.get("skip_late_season") and c.get("date") and pf._is_late_season(c["date"]):
        return {"code": "late_season", "text": "March and April are skipped"}
    dows = set(s.get(prefix + "banned_dows", []))
    months = set(s.get(prefix + "banned_months", []))
    if pf._should_skip_calendar(c.get("date"), dows, months):
        d = pd.Timestamp(c["date"])
        return {"code": "calendar", "text": f"{_DOW[d.dayofweek]} fixtures are skipped"}
    max_ev = s.get(prefix + "max_ev_pct")
    if max_ev is not None and n["ev"] > float(max_ev):
        return {"code": "max_ev", "text": f"EV {n['ev']:+.0%} above the {float(max_ev):.0%} sanity cap"}
    if line == "main":
        lo, hi = s.get("main_min_team_elo"), s.get("main_max_team_elo")
        gmin, gmax = s.get("main_elo_gap_min"), s.get("main_elo_gap_max")
    else:
        lo, hi = s.get("v2_min_team_elo"), s.get("v2_max_team_elo")
        gmin, gmax = s.get("v2_elo_gap_min"), s.get("v2_elo_gap_max")
    if any(v is not None for v in (lo, hi, gmin, gmax)) and pf.should_skip_elo_profile(
            c.get("home_elo"), c.get("away_elo"), lo, hi, gmin, gmax,
            require_known_elo=True):
        return {"code": "elo", "text": "Outside the Elo band"}
    gates = s.get("market_gates")
    min_prob = pf._gate(gates, c["market"], "min_prob", float(s.get("min_prob", 0.22)))
    if n["cal"] < min_prob:
        return {"code": "min_prob",
                "text": f"Calibrated {n['cal']:.0%} under the {min_prob:.0%} minimum"}
    floor = s.get("min_raw_draw_prob")
    if floor is not None and c["market"] == "D" and n["raw"] < float(floor):
        return {"code": "raw_floor",
                "text": f"Raw {n['raw']:.1%} under the {float(floor):.0%} floor"}
    min_ev = pf._gate(gates, c["market"], "min_ev", threshold)
    if n["ev"] < min_ev:
        return {"code": "ev", "text": f"EV {n['ev']:+.0%} under the {min_ev:.0%} gate"}
    if any(b.get("type") != "acca" and b["status"] == "pending"
           and b["home"] == c["home"] and b["away"] == c["away"]
           and b["market"] == c["market"] for b in p["bets"]):
        return {"code": "pending", "text": "Already backed"}
    if n["full_kelly"] <= 0:
        return {"code": "kelly", "text": "No Kelly stake at this price"}
    return None


# ── Verdicts ─────────────────────────────────────────────────────────────────

def dry_run_placed(p: dict, cands: list[dict], threshold: float,
                   placer: Callable, **kwargs) -> list[dict]:
    """Bets the real placer WOULD place now, from a deep copy of the portfolio."""
    shadow = copy.deepcopy(p)
    return placer(shadow, copy.deepcopy(cands), threshold, skip_log=[], **kwargs)


def _as_date(d) -> date | None:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        return pd.Timestamp(d).date()
    except Exception:
        return None


def fixture_verdicts(p: dict, fixtures: list[dict], cands: list[dict],
                     threshold: float, placer: Callable,
                     calibrators: dict | None = None,
                     match_counts: dict | None = None,
                     line: str = "main", today: date | None = None,
                     **placer_kwargs) -> dict[tuple[str, str], dict]:
    """One verdict per fixture for one portfolio line.

    status:
      placed   a pending bet already exists on this fixture
      bet      the real placer would stake it now
      held     every gate passes, but this run's 3-bet or exposure limit binds
      blocked  a gate stops it (reason names the first one)
      window   outside the 14-day betting window
      no_odds  no live price for this fixture
    """
    today = today or date.today()
    s = p["settings"]
    allowed = list(s.get("auto_markets", list(pf.PROFITABLE_MARKETS)))
    in_window_cands = []
    for c in cands:
        d = _as_date(c.get("date"))
        if d is None or (d - today).days <= BETTING_WINDOW_DAYS:
            in_window_cands.append(c)
    placed = dry_run_placed(p, in_window_cands, threshold, placer,
                            calibrators=calibrators, match_counts=match_counts,
                            **placer_kwargs)
    placed_by_fx = {(b["home"], b["away"]): b for b in placed}

    by_fx: dict[tuple[str, str], list[dict]] = {}
    for c in cands:
        by_fx.setdefault((c["home"], c["away"]), []).append(c)

    out: dict[tuple[str, str], dict] = {}
    for fx in fixtures:
        key = (fx["home"], fx["away"])
        fx_cands = by_fx.get(key, [])
        draw = next((c for c in fx_cands if c["market"] == "D"), None)
        nums = candidate_numbers(p, draw, calibrators) if draw else None
        base = {"line": line, "draw": nums}

        pending = [b for b in p["bets"] if b.get("type") != "acca"
                   and b["status"] == "pending"
                   and b["home"] == fx["home"] and b["away"] == fx["away"]]
        if pending:
            b = pending[0]
            out[key] = {**base, "status": "placed", "market": b["market"],
                        "stake": b["stake"], "odds": b["odds"],
                        "text": f"{MARKET_LABELS.get(b['market'], b['market'])} backed "
                                f"£{b['stake']:,.0f} @ {b['odds']:.2f}"}
            continue
        d = _as_date(fx.get("date"))
        if d is not None and (d - today).days > BETTING_WINDOW_DAYS:
            opens = d - timedelta(days=BETTING_WINDOW_DAYS)
            out[key] = {**base, "status": "window",
                        "text": f"Betting opens {opens.strftime('%a %-d %b')}"}
            continue
        if not fx_cands:
            out[key] = {**base, "status": "no_odds", "text": "No live price yet"}
            continue
        if key in placed_by_fx:
            b = placed_by_fx[key]
            out[key] = {**base, "status": "bet", "market": b["market"],
                        "stake": b["stake"], "odds": b["odds"],
                        "text": f"Would bet {MARKET_LABELS.get(b['market'], b['market'])} "
                                f"£{b['stake']:,.0f} @ {b['odds']:.2f}"}
            continue
        strategy = [c for c in fx_cands if c["market"] in allowed]
        reasons = {c["market"]: explain_block(p, c, threshold, calibrators,
                                              match_counts, line)
                   for c in strategy}
        if any(r is None for r in reasons.values()):
            out[key] = {**base, "status": "held",
                        "text": "Passes every gate · held by the run's bet or exposure limit"}
            continue
        lead = "D" if "D" in reasons else (next(iter(reasons)) if reasons else None)
        reason = reasons.get(lead) if lead else {"text": "No strategy market priced"}
        out[key] = {**base, "status": "blocked", "market": lead,
                    "code": (reason or {}).get("code"),
                    "text": (reason or {}).get("text", "Blocked")}
    return out


# ── Next matchday ────────────────────────────────────────────────────────────

def next_matchday(fixtures: list[dict], today: date | None = None) -> dict | None:
    """The next cluster of fixtures, as every page should describe it.

    `fixtures` is what fetch_upcoming_fixtures returns (already clustered to
    one gameweek). Returns {start, end, n, days_away, fixtures} or None.
    """
    today = today or date.today()
    dated = [(d, f) for f in fixtures if (d := _as_date(f.get("date"))) is not None]
    if not dated:
        return None
    dated.sort(key=lambda t: t[0])
    start, end = dated[0][0], dated[-1][0]
    return {"start": start, "end": end, "n": len(dated),
            "days_away": (start - today).days,
            "fixtures": [f for _, f in dated]}


def market_probs(api_o: dict | None) -> dict | None:
    """Margin-free H/D/A probabilities from a fixture's odds entry.

    Pinnacle when it is quoted (the sharp reference the strategy measures edge
    against); otherwise the median price across the listed books, which is a
    fairer read of the market than the best-of-panel maximum. Normalised so the
    three sum to 1. Returns {H, D, A, source} or None.
    """
    if not api_o:
        return None
    src, prices = None, None
    pin = api_o.get("_pinnacle") or {}
    if all(pin.get(k, 0) > 1 for k in ("H", "D", "A")):
        src, prices = "Pinnacle", {k: float(pin[k]) for k in ("H", "D", "A")}
    else:
        books = [b for b in (api_o.get("_books") or {}).values()
                 if all(b.get(k, 0) > 1 for k in ("H", "D", "A"))]
        if books:
            import statistics
            src = f"median of {len(books)} books"
            prices = {k: statistics.median(float(b[k]) for b in books) for k in ("H", "D", "A")}
        elif all(api_o.get(k, 0) > 1 for k in ("H", "D", "A")):
            src, prices = "best price", {k: float(api_o[k]) for k in ("H", "D", "A")}
    if not prices:
        return None
    inv = {k: 1.0 / v for k, v in prices.items()}
    tot = sum(inv.values())
    return {**{k: v / tot for k, v in inv.items()}, "source": src}


def kickoff_local(time_utc: str | None):
    """ESPN's UTC kickoff as a UK-local datetime, or None."""
    if not time_utc:
        return None
    try:
        from zoneinfo import ZoneInfo
        ts = pd.Timestamp(time_utc)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(ZoneInfo("Europe/London")).to_pydatetime()
    except Exception:
        return None


def fmt_money(x: float, signed: bool = True, pence: bool = False) -> str:
    """"+£1,703" / "−£1,703" (true minus sign). signed=False drops the plus."""
    x = float(x)
    body = f"£{abs(x):,.2f}" if pence else f"£{abs(x):,.0f}"
    if x < 0:
        return "−" + body
    return ("+" + body) if signed and x > 0 else body
