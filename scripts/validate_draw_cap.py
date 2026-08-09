"""Validate the DRAW_PROB_CAP guard and pick next-season default settings.

For each candidate config, run the honest 4-season backtest twice — with the
calibrated-draw-prob cap ON (0.45) and OFF — and compare per-season finals,
median, worst season, and the 4-season compounded path.

The cap is 'better' if it kills the small-sample calibrator artifact (the
66.7%-draw-prob max-Kelly bets) without materially hurting normal-season
results. Candidates include the deployed live config and the most robust
configs from sweep_100k + the May honest search.

Run:  python3 scripts/validate_draw_cap.py
Out:  data/diagnostics/draw_cap_validation_<ts>.json
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data import load_data, add_rolling_features   # noqa: E402
from models import backtest_models                  # noqa: E402
import portfolio as pf                              # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"

U25_GATES = {"under25": {"min_prob": 0.50, "min_ev": 0.05}}

CANDIDATES = [
    # The deployed live config (v158 simplified — what Main actually runs)
    dict(name="live-v158", mp=0.21, ev=23.6, kelly=1.0, max_stake=0.25,
         dows={"Mon"}, min_elo=None, max_ev=None, markets={"D", "under25"},
         odds="Max", detect="PS"),
    # v158 as originally found by the May honest search (filters on)
    dict(name="v158-full", mp=0.21, ev=23.6, kelly=1.0, max_stake=0.25,
         dows={"Mon"}, min_elo=1550, max_ev=1.0, markets={"D", "under25"},
         odds="Max", detect="PS", gap_min=80, gap_max=250),
    # May honest-search robustness winners
    dict(name="v170-may", mp=0.19, ev=23.6, kelly=0.5, max_stake=0.50,
         dows={"Mon"}, min_elo=None, max_ev=1.0, markets={"D", "under25"},
         odds="Max", detect="PS", gap_max=350),
    # sweep_100k robust top set (selected post-hoc — treat with suspicion)
    dict(name="s219", mp=0.23, ev=46.0, kelly=1.0, max_stake=0.25,
         dows={"Mon"}, min_elo=1500, max_ev=None, markets={"D", "under25"},
         odds="Max", detect=None),
    dict(name="s526", mp=0.17, ev=40.0, kelly=0.75, max_stake=0.25,
         dows={"Mon"}, min_elo=1500, max_ev=None, markets={"D"},
         odds="Max", detect="PS"),
    dict(name="s592", mp=0.20, ev=37.0, kelly=1.0, max_stake=0.33,
         dows={"Fri", "Mon"}, min_elo=1500, max_ev=None, markets={"D"},
         odds="Max", detect="PS"),
    dict(name="s776", mp=0.22, ev=40.0, kelly=0.5, max_stake=0.25,
         dows={"Mon"}, min_elo=1500, max_ev=None, markets={"D", "under25"},
         odds="Max", detect=None),
]


def _season_label(d: pd.Timestamp) -> str:
    return f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8 \
        else f"{d.year-1}-{str(d.year)[-2:]}"


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Loading data + season packs…", flush=True)
    df = load_data().copy()
    df["SeasonLbl"] = df["Date"].apply(_season_label)
    ftr = add_rolling_features(df)
    seasons = sorted(df["SeasonLbl"].unique())[1:]

    pack: dict[str, tuple] = {}
    for season in seasons:
        end   = df[df.SeasonLbl == season]["Date"].max()
        start = df[df.SeasonLbl == season]["Date"].min()
        dfs  = df[df.Date <= end].copy()
        fts  = ftr[ftr.Date <= end].copy()
        bt = backtest_models(dfs, fts, test_weeks=40)
        bts = bt[bt["Date"].apply(_season_label) == season].copy()
        pre, pre_ft = dfs[dfs.Date < start], fts[fts.Date < start]
        cal = ({} if len(pre) < 200 else
               pf.fit_calibrators_from_backtest(
                   backtest_models(pre, pre_ft, test_weeks=10)))
        pack[season] = (dfs, fts, bts, cal)
        print(f"  {season}: {len(bts)} rows", flush=True)

    ordered = sorted(pack.keys())

    def run(cfg, season, bankroll):
        dfs, fts, bts, cal = pack[season]
        mg = U25_GATES if "under25" in cfg["markets"] else None
        _, s = pf.ev_backtest_simulate(
            bts, dfs, df_features=fts, initial_bankroll=bankroll,
            min_ev_pct=cfg["ev"], kelly_frac=cfg["kelly"],
            max_stake_pct=cfg["max_stake"], min_prob=cfg["mp"],
            banned_dows=cfg["dows"], min_team_elo=cfg["min_elo"],
            max_ev_pct=cfg["max_ev"],
            elo_gap_min=cfg.get("gap_min"), elo_gap_max=cfg.get("gap_max"),
            calibrators=cal, allowed_markets=cfg["markets"], market_gates=mg,
            odds_source=cfg["odds"], detect_source=cfg["detect"],
            skip_late_season=True, skip_home_title_race=False,
            enable_simultaneous_correction=True)
        if "error" in s:
            return None, 0
        return float(s["final"]), int(s["n_bets"])

    results = []
    for cfg in CANDIDATES:
        row = {"name": cfg["name"], "config": {k: (sorted(v) if isinstance(v, set) else v)
                                               for k, v in cfg.items() if k != "name"}}
        for mode, cap in (("cap_off", None), ("cap_on", 0.45)):
            pf.DRAW_PROB_CAP = cap
            finals, total_n = {}, 0
            for season in ordered:
                f, n = run(cfg, season, 10_000.0)
                finals[season] = f if f is not None else 10_000.0
                total_n += n
            comp = 10_000.0
            for season in ordered:
                if comp < 100.0:
                    break
                f, _ = run(cfg, season, comp)
                if f is not None:
                    comp = f
            flist = sorted(finals.values())
            row[mode] = {
                "finals": {s: round(v, 0) for s, v in finals.items()},
                "median": round(flist[len(flist) // 2], 0),
                "min": round(flist[0], 0),
                "compounded": round(comp, 0),
                "n_bets": total_n,
            }
        pf.DRAW_PROB_CAP = 0.45   # restore default
        results.append(row)
        a, b = row["cap_off"], row["cap_on"]
        print(f"\n{cfg['name']}", flush=True)
        for mode, r in (("OFF", a), ("ON ", b)):
            per = " ".join(f"{s[-5:]}:£{v/1000:.1f}k" for s, v in sorted(r['finals'].items()))
            print(f"  cap {mode}: {per} | med £{r['median']:,.0f} min £{r['min']:,.0f} "
                  f"comp £{r['compounded']:,.0f} n={r['n_bets']}", flush=True)

    out = OUT_DIR / f"draw_cap_validation_{ts}.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seasons": ordered, "results": results,
    }, indent=1))
    print(f"\nSaved → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
