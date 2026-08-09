"""Standalone auto-bet runner for F_PRED.

Runs auto_place_value_bets + auto_settle on both portfolios *outside* of
Streamlit's runtime so bets get placed even when nobody has the app open.
Designed for cron/launchd. Activity is appended to data/activity.log
(JSON Lines), which the home-screen activity feed in app.py reads.

Manual usage:
    python3 scripts/run_auto_bet.py

Cron example (every hour, 5 min after the hour):
    5 * * * * cd /path/to/F_PRED && /usr/bin/env python3 scripts/run_auto_bet.py

For macOS launchd, see scripts/com.eoinhoustoun.fpred.plist.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable when invoked from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import (  # noqa: E402
    load_data, add_rolling_features, fetch_upcoming_fixtures,
    get_current_stats, get_current_elo, team_match_counts,
)
from models import (  # noqa: E402
    compute_dixon_coles_ratings, compute_draw_dc_ratings,
    compute_dixon_coles_kn_ratings,
    train_xgb, train_draw_xgb,
    backtest_models,
    blend_dc, blend_draw_specialist,
    predict_dixon_coles, predict_dixon_coles_kn,
    predict_xgb, predict_draw_xgb,
    seed_promoted_teams,
)
import portfolio as pf  # noqa: E402

ACTIVITY_LOG = ROOT / "data" / "activity.log"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(event_type: str, **kwargs) -> None:
    """Append a JSON-Lines event to the activity log."""
    ACTIVITY_LOG.parent.mkdir(exist_ok=True)
    entry = {"ts": _now_iso(), "type": event_type, **kwargs}
    with ACTIVITY_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _full_predict(home, away, dc_r, dc_draw_r, xgb_m, feat_cols,
                  draw_xgb_m, draw_fc, hs, as_):
    dc_pred = predict_dixon_coles(home, away, dc_r)
    xgb_p   = predict_xgb(xgb_m, feat_cols, hs, as_)
    dc_blend = blend_dc(dc_pred, xgb_p)
    draw_xgb_prob = predict_draw_xgb(draw_xgb_m, draw_fc, hs, as_)
    final = blend_draw_specialist(dc_blend, dc_draw_r, draw_xgb_prob, home, away)
    return dc_pred, final


def _full_predict_v2(home, away, dc_kn_r, dc_draw_r, xgb_m, feat_cols,
                     draw_xgb_m, draw_fc, hs, as_):
    kn_pred = predict_dixon_coles_kn(home, away, dc_kn_r)
    xgb_p   = predict_xgb(xgb_m, feat_cols, hs, as_)
    kn_blend = blend_dc(kn_pred, xgb_p)
    draw_xgb_prob = predict_draw_xgb(draw_xgb_m, draw_fc, hs, as_)
    final = blend_draw_specialist(kn_blend, dc_draw_r, draw_xgb_prob, home, away)
    return kn_pred, final


def seeded_ratings_for(fixtures, dc_r, dc_draw_r, dc_kn_r):
    """Seed every unrated side in the fixture list into all three ratings dicts.

    `predict_dixon_coles` falls back to 0.0 for an unknown team, and 0.0 is
    league average rather than "unknown", so a promoted side would otherwise be
    priced as a mid-table club. The three fits have to be seeded together: the
    standard DC, the draw specialist and the K-N variant all price the same
    fixture, and seeding one of them would quote it three different ways.

    Betting on seeded sides stays blocked by the no-history gate in
    `place_candidates`. Seeding is so the prices read honestly, not so they can
    be staked. This mirrors `cached_promoted_seeding` in app.py.
    """
    names = [t for f in fixtures for t in (f["home"], f["away"])]
    if not names:
        return dc_r, dc_draw_r, dc_kn_r
    return (seed_promoted_teams(dc_r, names),
            seed_promoted_teams(dc_draw_r, names),
            seed_promoted_teams(dc_kn_r, names))


def place_candidates(main_port, mt_port, main_cands, mt_cands,
                     calibrators=None, bin_variances=None,
                     match_counts=None, skip_log=None):
    """Run both portfolios' auto-bet paths, with the no-history gate armed.

    `match_counts` is not optional in spirit: `should_skip_unrated` treats
    absent counts as "gate disabled", so omitting it here is what let the runner
    diverge from app.py in the first place. Pass `team_match_counts(df)`.

    Returns (placed_main, placed_mt). A portfolio with auto-bet off is left
    untouched.
    """
    placed_main: list[dict] = []
    placed_mt:   list[dict] = []

    if bool(main_port["settings"].get("auto_bet_enabled", False)):
        thr = float(main_port["settings"].get("auto_bet_threshold", 0.40))
        placed_main = pf.auto_place_value_bets(
            main_port, main_cands, thr,
            calibrators=calibrators,
            match_counts=match_counts,
            skip_log=skip_log,
        )
    if bool(mt_port["settings"].get("auto_bet_enabled", False)):
        thr = float(mt_port["settings"].get("auto_bet_threshold", 0.40))
        placed_mt = pf.auto_place_value_bets_v2(
            mt_port, mt_cands, thr,
            calibrators=calibrators, bin_variances=bin_variances,
            match_counts=match_counts,
            skip_log=skip_log,
        )
    return placed_main, placed_mt


def main() -> int:
    log_event("run_started")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Loading data and fitting models…")

    try:
        df = load_data()
        df_features = add_rolling_features(df)
    except Exception as e:
        log_event("fatal", stage="data_load", error=str(e))
        raise

    # Fit models (no streamlit cache available outside runtime)
    dc_r          = compute_dixon_coles_ratings(df)
    dc_draw_r     = compute_draw_dc_ratings(df)
    dc_kn_r       = compute_dixon_coles_kn_ratings(df)
    xgb_m, fcols  = train_xgb(df_features)
    dxgb_m, dfc   = train_draw_xgb(df_features)
    elo_dict      = get_current_elo(df)
    teams         = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

    # Calibrators (10-week walk-forward backtest)
    print(f"[{datetime.now():%H:%M:%S}] Fitting isotonic calibrators…")
    bt_main = backtest_models(df, df_features, test_weeks=10)
    calibrators = pf.fit_calibrators_from_backtest(bt_main)
    bin_variances = {
        "H": pf.compute_per_bin_variance(bt_main, "H"),
        "D": pf.compute_per_bin_variance(bt_main, "D"),
        "A": pf.compute_per_bin_variance(bt_main, "A"),
    }

    # Load portfolios
    main_port = pf.load_portfolio()
    mt_port   = pf.load_portfolio_two()

    # Auto-settle prior pendings
    n_settled_main = pf.auto_settle(main_port, df)
    n_settled_mt   = pf.auto_settle(mt_port, df)
    if n_settled_main:
        # Per-bet detail for the activity feed
        for b in main_port["bets"]:
            if (b.get("settled_at", "")[:10] == datetime.now().date().isoformat()
                    and b["status"] in ("won", "lost")):
                log_event("settled", portfolio="main",
                          match=f"{b.get('home','?')} vs {b.get('away','?')}",
                          market=b.get("market"), result=b["status"],
                          profit=b.get("profit") or 0)
    if n_settled_mt:
        for b in mt_port["bets"]:
            if (b.get("settled_at", "")[:10] == datetime.now().date().isoformat()
                    and b["status"] in ("won", "lost")):
                log_event("settled", portfolio="mt",
                          match=f"{b.get('home','?')} vs {b.get('away','?')}",
                          market=b.get("market"), result=b["status"],
                          profit=b.get("profit") or 0)

    # Closing-line value backfill
    pf.backfill_clv_for_settled_bets(main_port, df)
    pf.backfill_clv_for_settled_bets(mt_port, df)

    # Auto-bet — needs API key + at least one portfolio with auto_bet_enabled
    api_key = main_port.get("settings", {}).get("odds_api_key", "").strip()
    main_enabled = bool(main_port["settings"].get("auto_bet_enabled", False))
    mt_enabled   = bool(mt_port["settings"].get("auto_bet_enabled", False))

    if not api_key:
        log_event("warning", message="No Odds API key configured — auto-bet skipped")
    elif not (main_enabled or mt_enabled):
        log_event("info", message="Auto-bet is disabled on both portfolios")
    else:
        print(f"[{datetime.now():%H:%M:%S}] Fetching upcoming fixtures + live odds…")
        try:
            fixtures = fetch_upcoming_fixtures(lookahead_days=14)
        except Exception as e:
            fixtures = []
            log_event("error", stage="fixtures", error=str(e))
        try:
            live_odds_map = pf.fetch_live_odds(api_key)
        except Exception as e:
            live_odds_map = {}
            log_event("error", stage="live_odds", error=str(e))

        if not fixtures:
            log_event("warning", message="No upcoming fixtures found")
        elif not live_odds_map:
            log_event("warning", message="Live odds map empty (cache miss + no API headroom?)")
        else:
            # Seed promoted sides before pricing, exactly as app.py does, so a
            # promoted fixture is quoted off the empirical prior instead of
            # league average — and stays visible instead of vanishing below.
            dc_r, dc_draw_r, dc_kn_r = seeded_ratings_for(
                fixtures, dc_r, dc_draw_r, dc_kn_r)
            teams = set(teams) | set(dc_r.get("attacks", {}))

            main_cands: list[dict] = []
            mt_cands:   list[dict] = []
            for fix in fixtures:
                h, a = fix["home"], fix["away"]
                if h not in teams or a not in teams:
                    # Was a bare continue, which made promoted sides invisible:
                    # in 2026-27 that silently hid 2 of the 10 opening fixtures.
                    unknown = [t for t in (h, a) if t not in teams]
                    log_event("fixture_skipped", reason="unknown_team",
                              match=f"{h} vs {a}",
                              detail=f"{', '.join(unknown)} not in the model's team list")
                    continue
                api_o = live_odds_map.get((h, a), {})
                if not api_o or "H" not in api_o:
                    continue
                try:
                    hs = get_current_stats(df, h, elo_dict=elo_dict)
                    as_ = get_current_stats(df, a, elo_dict=elo_dict)
                    dc_p, main_res = _full_predict(h, a, dc_r, dc_draw_r,
                                                    xgb_m, fcols, dxgb_m, dfc, hs, as_)
                    _kn_p, mt_res = _full_predict_v2(h, a, dc_kn_r, dc_draw_r,
                                                      xgb_m, fcols, dxgb_m, dfc, hs, as_)
                except Exception as e:
                    log_event("error", stage="predict",
                              match=f"{h} vs {a}", error=str(e))
                    continue

                ds = fix["date"].isoformat()
                pinn = api_o.get("_pinnacle") if isinstance(api_o, dict) else None
                p_o25 = dc_p.get("over_25", 0.5)

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
                    base = dict(home=h, away=a, date=ds, market=mkt, selection=lbl,
                                odds=place_o, detect_odds=detect_o,
                                home_elo=elo_dict.get(h), away_elo=elo_dict.get(a))
                    main_cands.append({**base, "model_prob": m_prob,
                                        "ev": pf.compute_ev(m_prob, ev_ref) if ev_ref else -1})
                    mt_cands.append({**base, "model_prob": mt_prob,
                                      "ev": pf.compute_ev(mt_prob, ev_ref) if ev_ref else -1})

            # Place via the existing portfolio paths, with the no-history gate
            # armed. Counted once and shared by both lines, as in app.py.
            match_counts = team_match_counts(df)
            skip_log: list[dict] = []
            placed_main, placed_mt = place_candidates(
                main_port, mt_port, main_cands, mt_cands,
                calibrators=calibrators, bin_variances=bin_variances,
                match_counts=match_counts, skip_log=skip_log,
            )
            for portfolio, placed in (("main", placed_main), ("mt", placed_mt)):
                for b in placed:
                    log_event("auto_bet_placed", portfolio=portfolio,
                              match=f"{b['home']} vs {b['away']}",
                              market=b["market"], selection=b["selection"],
                              stake=b["stake"], odds=b["odds"], ev=b["ev"])
            if placed_main:
                print(f"  Main: placed {len(placed_main)} bet(s)")
            if placed_mt:
                print(f"  Mock Two: placed {len(placed_mt)} bet(s)")

            # A silent skip is how two invisible fixtures happened. Log one
            # line per distinct fixture rather than one per market.
            for match in dict.fromkeys(
                    f"{e['home']} vs {e['away']}" for e in skip_log
                    if e.get("reason") == "no_history"):
                log_event("fixture_skipped", reason="no_history", match=match,
                          detail="too little top-flight history to rate")

    # Persist
    pf.save_portfolio(main_port)
    pf.save_portfolio_two(mt_port)

    # CLV snapshot at end of run — informs the home-screen "edge alive" indicator
    main_clv = pf.clv_summary(main_port)
    mt_clv   = pf.clv_summary(mt_port)
    log_event("clv_snapshot", portfolio="main",
              median_clv=main_clv.get("median_clv"),
              pct_positive=main_clv.get("pct_positive"),
              n=main_clv.get("n"))
    log_event("clv_snapshot", portfolio="mt",
              median_clv=mt_clv.get("median_clv"),
              pct_positive=mt_clv.get("pct_positive"),
              n=mt_clv.get("n"))
    log_event("run_completed")
    print(f"[{datetime.now():%H:%M:%S}] Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        log_event("fatal", error=str(e), trace=traceback.format_exc())
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
