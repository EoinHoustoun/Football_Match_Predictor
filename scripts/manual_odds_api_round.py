"""One-off: place bets using The Odds API as the fixture source instead of
ESPN. Useful when ESPN lags (e.g. the final PL gameweek isn't listed yet
but the Odds API already has odds for it).

Bypasses `fetch_upcoming_fixtures()` (ESPN scoreboard) and instead reads
fixtures directly from The Odds API response. All other logic — prediction,
calibration, EV gate, Kelly sizing, max-bets cap, exposure cap — uses the
same code paths as the live `_session_auto_bet()` flow.

USAGE:
  python3 scripts/manual_odds_api_round.py              # dry-run (show only)
  python3 scripts/manual_odds_api_round.py --place      # actually place bets
  python3 scripts/manual_odds_api_round.py --portfolio mt --place   # Mock Two

Safety: dry-run by default. Even with --place, all the same gates apply
(min_ev, min_prob, max_exposure, max_auto_bets) so it won't over-bet.
"""
from __future__ import annotations
import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests        # noqa: E402
import pandas as pd    # noqa: E402

from data import (  # noqa: E402
    load_data, add_rolling_features, get_current_stats, get_current_elo,
    get_current_teams,
)
from models import (  # noqa: E402
    compute_dixon_coles_ratings, compute_draw_dc_ratings,
    compute_dixon_coles_kn_ratings, train_xgb, train_draw_xgb,
    predict_dixon_coles, predict_dixon_coles_kn,
    predict_xgb, predict_draw_xgb,
    blend_dc, blend_draw_specialist,
    backtest_models,
)
import portfolio as pf  # noqa: E402

# Same name-map used by fetch_live_odds (Odds API → football-data names)
_NAME_MAP = pf._NAME_MAP


def _full_predict_main(home, away, dc_r, dc_draw_r, xgb_m, fcols,
                        dxgb_m, dfc, hs, as_):
    dc_pred = predict_dixon_coles(home, away, dc_r)
    xgb_p   = predict_xgb(xgb_m, fcols, hs, as_)
    dc_blend = blend_dc(dc_pred, xgb_p)
    draw_xgb_prob = predict_draw_xgb(dxgb_m, dfc, hs, as_)
    final = blend_draw_specialist(dc_blend, dc_draw_r, draw_xgb_prob, home, away)
    return dc_pred, final


def _full_predict_mt(home, away, dc_kn_r, dc_draw_r, xgb_m, fcols,
                     dxgb_m, dfc, hs, as_):
    kn_pred = predict_dixon_coles_kn(home, away, dc_kn_r)
    xgb_p   = predict_xgb(xgb_m, fcols, hs, as_)
    kn_blend = blend_dc(kn_pred, xgb_p)
    draw_xgb_prob = predict_draw_xgb(dxgb_m, dfc, hs, as_)
    final = blend_draw_specialist(kn_blend, dc_draw_r, draw_xgb_prob, home, away)
    return kn_pred, final


def fetch_odds_api_fixtures(api_key: str) -> list[dict]:
    """Pull EPL fixtures from Odds API directly. Returns list of dicts:
       {date, home, away, place_odds: {H,D,A,over25,under25},
        detect_odds: {H,D,A,over25,under25} or None}
    """
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/",
        params={"apiKey": api_key.strip(), "regions": "uk,eu",
                "markets": "h2h,totals", "oddsFormat": "decimal"},
        timeout=15,
    )
    r.raise_for_status()
    raw = r.json()

    fixtures = []
    for game in raw:
        # Parse kickoff
        try:
            commence = datetime.fromisoformat(
                game["commence_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        if commence < datetime.now(timezone.utc):
            continue  # already kicked off / finished

        home = _NAME_MAP.get(game.get("home_team", ""), game.get("home_team", ""))
        away = _NAME_MAP.get(game.get("away_team", ""), game.get("away_team", ""))

        # Per-bookmaker quotes
        books: dict = {}
        for bm in game.get("bookmakers", []):
            bo = {}
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for outcome in mkt.get("outcomes", []):
                        nm = _NAME_MAP.get(outcome["name"], outcome["name"])
                        p  = float(outcome["price"])
                        if nm == home:                bo["H"] = p
                        elif nm == away:              bo["A"] = p
                        elif outcome["name"] == "Draw": bo["D"] = p
                elif mkt["key"] == "totals":
                    for outcome in mkt.get("outcomes", []):
                        if float(outcome.get("point", 0)) != 2.5:
                            continue
                        p = float(outcome["price"])
                        if outcome["name"] == "Over":   bo["over25"] = p
                        elif outcome["name"] == "Under": bo["under25"] = p
            if bo:
                books[bm.get("key", "?")] = bo

        if not books:
            continue

        # Best-of-panel (Max) for placement
        place_odds = {}
        for mkt_key in ("H", "D", "A", "over25", "under25"):
            quotes = [b[mkt_key] for b in books.values() if mkt_key in b]
            if quotes:
                place_odds[mkt_key] = round(max(quotes), 3)

        if not {"H", "D", "A"}.issubset(place_odds):
            continue

        detect_odds = books.get("pinnacle") if "pinnacle" in books else None

        fixtures.append({
            "date":         commence.date(),
            "kickoff_utc":  commence,
            "home":         home,
            "away":         away,
            "place_odds":   place_odds,
            "detect_odds":  detect_odds,
            "books_count":  len(books),
        })

    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", action="store_true",
                        help="Actually place bets (default: dry-run)")
    parser.add_argument("--portfolio", choices=["main", "mt"], default="main",
                        help="Which portfolio to operate on (default: main)")
    parser.add_argument("--max-bets", type=int, default=3,
                        help="Max bets per run (default: 3). Raise for final-round "
                             "windows when you want every qualifying bet. The 50%% "
                             "exposure cap still applies as a hard safety.")
    args = parser.parse_args()

    is_dry = not args.place
    pf_label = "Main" if args.portfolio == "main" else "Mock Two"
    print(f"=== {'DRY RUN' if is_dry else 'LIVE PLACE'} · Portfolio: {pf_label} ===\n")

    # Load portfolio + API key
    if args.portfolio == "main":
        port = pf.load_portfolio()
    else:
        port = pf.load_portfolio_two()
    api_key = pf.load_portfolio().get("settings", {}).get("odds_api_key", "").strip()
    if not api_key:
        print("ERROR: no Odds API key configured")
        return 1
    if not port["settings"].get("auto_bet_enabled", False):
        print(f"WARNING: auto_bet_enabled=False for {pf_label} — would skip in live mode.")
        if args.place:
            return 1

    # Fetch fixtures from Odds API
    print("Fetching fixtures from Odds API…")
    fixtures = fetch_odds_api_fixtures(api_key)
    if not fixtures:
        print("No upcoming fixtures returned.")
        return 1

    fixtures.sort(key=lambda f: f["kickoff_utc"])
    print(f"\n{len(fixtures)} upcoming EPL fixtures:\n")
    for fx in fixtures:
        pinn_chk = "✓" if fx["detect_odds"] else "—"
        print(f"  {fx['date']!s}  {fx['kickoff_utc'].strftime('%H:%M UTC'):<10}  "
              f"{fx['home']:>16} vs {fx['away']:<16}  "
              f"D@{fx['place_odds'].get('D', '?')!s:<5}  U25@{fx['place_odds'].get('under25', '?')!s:<5}  "
              f"books={fx['books_count']}  PS:{pinn_chk}")

    # Load data + train models
    print("\nLoading data + fitting models…")
    df = load_data()
    df_features = add_rolling_features(df)
    teams = get_current_teams(df)
    elo_dict = get_current_elo(df)

    dc_r      = compute_dixon_coles_ratings(df)
    dc_draw_r = compute_draw_dc_ratings(df)
    xgb_m, fcols = train_xgb(df_features)
    dxgb_m, dfc  = train_draw_xgb(df_features)
    if args.portfolio == "mt":
        dc_kn_r = compute_dixon_coles_kn_ratings(df)

    # Calibrators (10w prior — same as cached_calibrators in app)
    print("Fitting honest calibrator (10w prior backtest)…")
    bt10 = backtest_models(df, df_features, test_weeks=10)
    calibrators = pf.fit_calibrators_from_backtest(bt10)

    bin_variances = None
    if args.portfolio == "mt":
        bin_variances = {
            "H": pf.compute_per_bin_variance(bt10, "H"),
            "D": pf.compute_per_bin_variance(bt10, "D"),
            "A": pf.compute_per_bin_variance(bt10, "A"),
        }

    # Build candidates — same shape as _session_auto_bet
    candidates: list[dict] = []
    for fx in fixtures:
        h, a = fx["home"], fx["away"]
        if h not in teams or a not in teams:
            print(f"  [skip] {h} vs {a}: not in trained team list")
            continue
        try:
            hs  = get_current_stats(df, h, elo_dict=elo_dict)
            as_ = get_current_stats(df, a, elo_dict=elo_dict)
            if args.portfolio == "main":
                dc_p, res = _full_predict_main(h, a, dc_r, dc_draw_r,
                                                xgb_m, fcols, dxgb_m, dfc, hs, as_)
            else:
                dc_p, res = _full_predict_mt(h, a, dc_kn_r, dc_draw_r,
                                              xgb_m, fcols, dxgb_m, dfc, hs, as_)
        except Exception as e:
            print(f"  [skip] {h} vs {a}: predict error {e}")
            continue

        ds = fx["date"].isoformat()
        p_o25 = dc_p.get("over_25", 0.5)

        for mkt, prob, lbl in [
            ("H",       res["home_win"], f"Home Win ({h})"),
            ("D",       res["draw"],     "Draw"),
            ("A",       res["away_win"], f"Away Win ({a})"),
            ("over25",  p_o25,           "Over 2.5 Goals"),
            ("under25", 1.0 - p_o25,     "Under 2.5 Goals"),
        ]:
            place_o = fx["place_odds"].get(mkt)
            if not place_o:
                continue
            detect_o = fx["detect_odds"].get(mkt) if fx["detect_odds"] else None
            ev_ref = detect_o if (detect_o and detect_o > 1) else place_o
            candidates.append({
                "home": h, "away": a, "date": ds,
                "market": mkt, "selection": lbl,
                "model_prob": prob,
                "odds": place_o,
                "detect_odds": detect_o,
                "home_elo": elo_dict.get(h),
                "away_elo": elo_dict.get(a),
                "ev": pf.compute_ev(prob, ev_ref) if ev_ref else -1,
            })

    print(f"\n{len(candidates)} raw candidates built from {len(fixtures)} fixtures.\n")

    # Pre-flight: show top candidates by EV for review
    sorted_c = sorted([c for c in candidates if c["market"] in port["settings"].get("auto_markets", [])],
                      key=lambda x: x["ev"], reverse=True)
    print("Top 10 candidates in auto_markets, ranked by EV:\n")
    print(f"  {'mkt':<8}  {'match':<40}  {'model%':>7}  {'EV':>7}  {'odds':>5}")
    print("-" * 75)
    for c in sorted_c[:10]:
        print(f"  {c['market']:<8}  {c['home']:>18} vs {c['away']:<18}  "
              f"{c['model_prob']*100:>6.1f}%  {c['ev']*100:>+6.1f}%  {c['odds']:>5.2f}")

    if is_dry:
        print(f"\nDRY RUN — no bets placed. To actually place, re-run with --place")
        return 0

    # ── LIVE PLACE ───────────────────────────────────────────────────
    print("\nPlacing bets via auto_place_value_bets…\n")
    threshold = float(port["settings"].get("auto_bet_threshold", 0.40))
    if args.portfolio == "main":
        placed = pf.auto_place_value_bets(
            port, candidates, threshold,
            max_auto_bets=args.max_bets,
            calibrators=calibrators,
        )
    else:
        placed = pf.auto_place_value_bets_v2(
            port, candidates, threshold,
            max_auto_bets=args.max_bets,
            calibrators=calibrators, bin_variances=bin_variances,
        )

    if placed:
        # Backup before save
        from datetime import datetime as _dt
        bkp = ROOT / "data" / f"portfolio{'_two' if args.portfolio == 'mt' else ''}.backup.preManual.{_dt.now():%Y%m%d_%H%M%S}.json"
        bkp.write_text(json.dumps(port if args.portfolio == "main" else port,
                                   indent=2, default=str))
        if args.portfolio == "main":
            pf.save_portfolio(port)
        else:
            pf.save_portfolio_two(port)
        print(f"✅ Placed {len(placed)} bet(s). Portfolio saved. Backup: {bkp.name}\n")
        for b in placed:
            print(f"  {b['home']:>18} vs {b['away']:<18}  "
                  f"{b['market']:<8}  £{b['stake']:>7,.2f} @ {b['odds']:.2f}  "
                  f"EV +{b['ev']*100:.1f}%")
    else:
        print("No bets placed (none passed all gates).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
