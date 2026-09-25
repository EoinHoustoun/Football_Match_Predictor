"""Matchday verdicts must say exactly what the auto-bet would do.

Every page that shows "would we bet this?" reads matchday.py. The verdict itself
comes from the real placers on a copy of the portfolio; `explain_block` only
names the gate. These tests pin the explainer to the placers so the named reason
cannot drift from the behaviour, and prove the dry run never touches the
portfolio it was given.
"""
from __future__ import annotations

import copy
import json
import random
from datetime import date

import pytest

import matchday as md
import portfolio as pf


def _port(line: str, **overrides) -> dict:
    base = pf._DEFAULT if line == "main" else pf._MOCK2_DEFAULT
    p = {"initial_bankroll": 20000.0, "bankroll": 20000.0, "bets": [],
         "settings": copy.deepcopy(base["settings"])}
    p["settings"].update(overrides)
    return p


# The live-style configuration: draws + U2.5, raw floor, Monday ban.
LIVE_MAIN = dict(auto_markets=["D", "under25"], min_prob=0.21, min_raw_draw_prob=0.30,
                 main_banned_dows=["Mon"], main_min_team_elo=None, max_bets_per_club=None,
                 min_team_matches=None, use_calibrated_probs=False,
                 market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
                 skip_late_season=True)
LIVE_MT = dict(auto_markets=["D", "under25"], min_prob=0.21, min_raw_draw_prob=0.30,
               v2_banned_dows=["Mon"], v2_min_team_elo=None, max_bets_per_club=None,
               min_team_matches=None, use_calibrated_probs=False,
               use_uncertainty_kelly=False, detect_source=None,
               market_gates={"under25": {"min_prob": 0.50, "min_ev": 0.05}},
               skip_late_season=True)

DATES = ["2026-10-10", "2026-10-11", "2026-10-12", "2026-03-14", "2026-12-26"]


def _random_cand(rng: random.Random) -> dict:
    mkt = rng.choice(["D", "D", "D", "H", "A", "over25", "under25"])
    return dict(home="H" + str(rng.randint(0, 999)), away="A" + str(rng.randint(0, 999)),
                date=rng.choice(DATES), market=mkt, selection=mkt,
                model_prob=round(rng.uniform(0.15, 0.65), 3),
                odds=round(rng.uniform(1.4, 8.0), 2), detect_odds=None,
                home_elo=1500, away_elo=1500)


@pytest.mark.parametrize("line,placer,settings", [
    ("main", pf.auto_place_value_bets, LIVE_MAIN),
    ("mt", pf.auto_place_value_bets_v2, LIVE_MT),
])
def test_explainer_agrees_with_the_real_placer(line, placer, settings):
    rng = random.Random(7)
    p = _port(line, **settings)
    thr = 0.236
    agree = 0
    for _ in range(600):
        c = _random_cand(rng)
        placed = md.dry_run_placed(p, [c], thr, placer)
        reason = md.explain_block(p, c, thr, line=line)
        assert (reason is None) == bool(placed), (c, reason, placed)
        agree += 1
    assert agree == 600


def test_dry_run_never_mutates_the_portfolio():
    p = _port("main", **LIVE_MAIN)
    before = json.dumps(p, sort_keys=True)
    c = dict(home="X", away="Y", date="2026-10-10", market="D", selection="Draw",
             model_prob=0.40, odds=4.5, detect_odds=None)
    assert md.dry_run_placed(p, [c], 0.2, pf.auto_place_value_bets)
    assert json.dumps(p, sort_keys=True) == before


def _fx(h, a, d):
    return {"home": h, "away": a, "date": date.fromisoformat(d)}


def test_verdict_statuses():
    p = _port("main", **LIVE_MAIN)
    p["bets"].append({"type": "single", "status": "pending", "home": "P", "away": "Q",
                      "market": "D", "stake": 1000.0, "odds": 4.0})
    fixtures = [_fx("P", "Q", "2026-10-10"),   # already backed
                _fx("B", "C", "2026-10-10"),   # clears every gate
                _fx("D", "E", "2026-10-10"),   # raw 0.27 under the 0.30 floor
                _fx("F", "G", "2026-10-12"),   # Monday
                _fx("J", "K", "2026-10-10"),   # no odds
                _fx("L", "M", "2026-11-30")]   # outside the betting window
    def c(h, a, d, prob, odds, mkt="D"):
        return dict(home=h, away=a, date=d, market=mkt, selection=mkt,
                    model_prob=prob, odds=odds, detect_odds=None)
    cands = [c("B", "C", "2026-10-10", 0.36, 4.0),
             c("D", "E", "2026-10-10", 0.27, 5.5),
             c("F", "G", "2026-10-12", 0.40, 4.0),
             c("L", "M", "2026-11-30", 0.40, 4.0)]
    v = md.fixture_verdicts(p, fixtures, cands, 0.236, pf.auto_place_value_bets,
                            line="main", today=date(2026, 9, 30))
    assert v[("P", "Q")]["status"] == "placed"
    assert v[("B", "C")]["status"] == "bet" and v[("B", "C")]["stake"] > 0
    assert v[("D", "E")]["status"] == "blocked" and v[("D", "E")]["code"] == "raw_floor"
    assert "27.0%" in v[("D", "E")]["text"]
    assert v[("F", "G")]["code"] == "calendar" and "Mon" in v[("F", "G")]["text"]
    assert v[("J", "K")]["status"] == "no_odds"
    assert v[("L", "M")]["status"] == "window"


def test_non_strategy_markets_never_count_as_value():
    p = _port("main", **LIVE_MAIN)
    # A huge home-win edge must not turn into a verdict: H is not a strategy market.
    cands = [dict(home="B", away="C", date="2026-10-10", market="H", selection="H",
                  model_prob=0.70, odds=3.0, detect_odds=None),
             dict(home="B", away="C", date="2026-10-10", market="D", selection="D",
                  model_prob=0.22, odds=4.0, detect_odds=None)]
    v = md.fixture_verdicts(p, [_fx("B", "C", "2026-10-10")], cands, 0.236,
                            pf.auto_place_value_bets, line="main",
                            today=date(2026, 10, 1))
    assert v[("B", "C")]["status"] == "blocked"
    assert v[("B", "C")]["market"] == "D"


def test_next_matchday_and_money_format():
    nm = md.next_matchday([_fx("A", "B", "2026-10-12"), _fx("C", "D", "2026-10-10")],
                          today=date(2026, 9, 25))
    assert nm["start"] == date(2026, 10, 10) and nm["end"] == date(2026, 10, 12)
    assert nm["n"] == 2 and nm["days_away"] == 15
    assert md.next_matchday([]) is None
    assert md.fmt_money(-1703.33) == "−£1,703"
    assert md.fmt_money(15016) == "+£15,016"
    assert md.fmt_money(21994.69, signed=False, pence=True) == "£21,994.69"
