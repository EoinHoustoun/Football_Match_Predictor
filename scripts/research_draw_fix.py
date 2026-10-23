"""Mock Two research: is the draw specialist inflating draws, and does fixing it pay?

Background (17 Oct 2026). Averaged over the 20 upcoming fixtures, Dixon-Coles
says 24.6% draw and Pinnacle 24.0%, but the final model says 31.0%. The lift
comes from the draw specialist (35% weight), half of which is a binary draw
XGBoost trained with scale_pos_weight = (1 - rate) / rate ~ 3, which multiplies
its draw odds ~3x (it averages 50%), and half a Dixon-Coles refit that counts
draws three times (31%).

Variants, all on the SAME refits (only the recombination differs):
  incumbent         0.65 * dc_blend + 0.35 * (0.5 * drawDC(boost 3) + 0.5 * drawXGB)
  xgb corrected     drawXGB prior-corrected: p / (p + (1 - p) * spw)
  xgb + no boost    as above, and the draw DC refit with boost 1
  no specialist     dc_blend draw alone

Step 1 (`build`) refits every model per season exactly as backtest_models does
(train to 40 weeks before season end, test the season) and stores each layer.
It checks the recomposed incumbent against the stored exposure-cap pack.
Step 2 (`run`) scores each variant: draw log loss and bias against outcomes and
Pinnacle, then the live gameweek simulator (35% pro-rata cap, raw floor 0.30,
trailing 280-day isotonic calibrator refit per variant, detect PS / place Max,
Main's deployed gates and Kelly). Worst season is the headline.

Usage:
  .venv/bin/python scripts/research_draw_fix.py build
  .venv/bin/python scripts/research_draw_fix.py run
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import models as m  # noqa: E402
import portfolio as pf  # noqa: E402
import validate_2026_27 as v  # noqa: E402
from validate_draw_calibration import make_fn  # noqa: E402

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"
LAYERS = ROOT / "data" / "diagnostics" / "draw_fix_layers.pkl"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
LIVE_KW = dict(gameweek_mode=True, max_pending_bets=None, exposure_cap_pct=0.35,
               exposure_cap_basis="gameweek", exposure_cap_mode="prorata",
               min_raw_draw_prob=0.30)
EPS = 1e-4


# ── step 1: refit and record every layer ─────────────────────────────────────
def layers_for(df: pd.DataFrame, ftr: pd.DataFrame, test_weeks: int = 40) -> pd.DataFrame:
    from data import get_current_elo, get_current_stats
    cutoff = df["Date"].max() - pd.Timedelta(weeks=test_weeks)
    train_df, train_ft = df[df["Date"] <= cutoff], ftr[ftr["Date"] <= cutoff]
    test_df = df[df["Date"] > cutoff]

    dc_r = m.compute_dixon_coles_ratings(train_df)
    ddc3 = m.compute_draw_dc_ratings(train_df)
    ddc1 = m.compute_draw_dc_ratings(train_df, draw_boost=1.0)
    xgb_m, fc = m.train_xgb(train_ft)
    dxgb_m, dfc = m.train_draw_xgb(train_ft)
    rate = float((train_ft.dropna(subset=["FTR"])["FTR"] == "D").mean())
    spw = float(dxgb_m.get_params()["scale_pos_weight"]) if dxgb_m is not None else 1.0
    elo = get_current_elo(train_df)

    rows = []
    for _, r in test_df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hs = get_current_stats(train_df, h, elo_dict=elo)
        as_ = get_current_stats(train_df, a, elo_dict=elo)
        db = m.blend_dc(m.predict_dixon_coles(h, a, dc_r), m.predict_xgb(xgb_m, fc, hs, as_))
        rows.append({
            "Date": r["Date"], "Home": h, "Away": a, "FTR": r["FTR"],
            "db_h": db["home_win"], "db_d": db["draw"], "db_a": db["away_win"],
            "ddc3": m.predict_dixon_coles(h, a, ddc3)["draw"],
            "ddc1": m.predict_dixon_coles(h, a, ddc1)["draw"],
            "dxgb": m.predict_draw_xgb(dxgb_m, dfc, hs, as_),
            "spw": spw, "train_draw_rate": rate,
        })
    return pd.DataFrame(rows)


def build() -> None:
    pack = pickle.loads(PACK.read_bytes())
    out = {}
    for s in SEASONS:
        t0 = time.time()
        dfs, ftr, bt, _ = pack[s]
        lay = layers_for(dfs, ftr)
        lay = lay[lay["Date"].isin(set(bt["Date"]))].copy()
        # Sanity: the incumbent recombination must reproduce the stored pack.
        chk = lay.merge(bt[["Date", "Home", "Away", "_dc_d"]], on=["Date", "Home", "Away"])
        err = np.abs(recombine(chk, "incumbent")[1] - chk["_dc_d"]).max()
        out[s] = lay
        print(f"{s}: {len(lay)} matches, spw {lay['spw'].iloc[0]:.2f}, "
              f"max |incumbent - pack| = {err:.2e}, {time.time() - t0:.0f}s", flush=True)
    LAYERS.write_bytes(pickle.dumps(out))
    print(f"Written {LAYERS}")


# ── step 2: recombine and score ───────────────────────────────────────────────
def recombine(lay: pd.DataFrame, variant: str):
    dx = lay["dxgb"].to_numpy(float)
    if variant != "incumbent":
        spw = lay["spw"].to_numpy(float)
        dx = dx / (dx + (1 - dx) * spw)
    ddc = lay["ddc1" if variant == "xgb + no boost" else "ddc3"].to_numpy(float)
    spec = 0.5 * ddc + 0.5 * dx
    main_d = lay["db_d"].to_numpy(float)
    d = main_d if variant == "no specialist" else 0.65 * main_d + 0.35 * spec
    nd = lay["db_h"].to_numpy(float) + lay["db_a"].to_numpy(float)
    scale = np.maximum(1 - d, 0.01) / nd
    h, a = lay["db_h"].to_numpy(float) * scale, lay["db_a"].to_numpy(float) * scale
    tot = h + d + a
    return h / tot, d / tot, a / tot


VARIANTS = ["incumbent", "xgb corrected", "xgb + no boost", "no specialist"]


def as_bt(lay: pd.DataFrame, variant: str) -> pd.DataFrame:
    h, d, a = recombine(lay, variant)
    act = lay["FTR"].map({"H": "Home Win", "D": "Draw", "A": "Away Win"})
    return pd.DataFrame({
        "Date": lay["Date"].values, "Home": lay["Home"].values, "Away": lay["Away"].values,
        "Actual": act.values,
        "DC_H": np.round(h * 100, 1), "DC_D": np.round(d * 100, 1), "DC_A": np.round(a * 100, 1),
        "_dc_h": h, "_dc_d": d, "_dc_a": a,
        "_act_h": (lay["FTR"] == "H").astype(float).values,
        "_act_d": (lay["FTR"] == "D").astype(float).values,
        "_act_a": (lay["FTR"] == "A").astype(float).values,
    })


def _ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(bt: pd.DataFrame, dfs: pd.DataFrame) -> dict:
    odds = dfs[["Date", "HomeTeam", "AwayTeam", "PSH", "PSD", "PSA"]].rename(
        columns={"HomeTeam": "Home", "AwayTeam": "Away"}).dropna()
    j = bt.merge(odds, on=["Date", "Home", "Away"], how="inner")
    inv = 1 / j[["PSH", "PSD", "PSA"]].to_numpy(float)
    mkt = inv[:, 1] / inv.sum(1)
    y = j["_act_d"].to_numpy(float)
    p = j["_dc_d"].to_numpy(float)
    hi = p >= 0.30
    return {"n": len(j), "mean_model": round(p.mean() * 100, 1),
            "mean_pinnacle": round(mkt.mean() * 100, 1), "draw_rate": round(y.mean() * 100, 1),
            "logloss_model": round(_ll(p, y), 4), "logloss_pinnacle": round(_ll(mkt, y), 4),
            "n_raw_ge_30": int(hi.sum()),
            "draw_rate_when_raw_ge_30": round(y[hi].mean() * 100, 1) if hi.any() else None,
            "pinnacle_when_raw_ge_30": round(mkt[hi].mean() * 100, 1) if hi.any() else None,
            "model_when_raw_ge_30": round(p[hi].mean() * 100, 1) if hi.any() else None}


def simulate(pack, bts: dict, live: dict) -> dict:
    hist = pd.concat([bts[s] for s in SEASONS], ignore_index=True).sort_values("Date")
    cal_fn = make_fn(hist, "isotonic", 280)
    per = {}
    for s in TEST_SEASONS:
        dfs, ftr, _, _ = pack[s]
        log, summ = pf.ev_backtest_simulate(
            bts[s], dfs, df_features=ftr, initial_bankroll=v.INITIAL,
            enable_simultaneous_correction=True, odds_source="Max",
            detect_source="PS", calibrators=None, calibrator_fn=cal_fn,
            **LIVE_KW, **live)
        if "error" in summ or log.empty:
            per[s] = {"profit": 0.0, "n_bets": 0, "max_dd": 0.0, "median_clv": None,
                      "win_rate": None}
            continue
        won = (log["Result"] == "✅").mean() if "Result" in log.columns else None
        per[s] = {"profit": round(summ["profit"], 0), "n_bets": summ["n_bets"],
                  "max_dd": round(v._max_drawdown_pct(log), 1),
                  "median_clv": v._median_clv(log, dfs),
                  "win_rate": None if won is None else round(float(won) * 100, 1)}
    profits = [x["profit"] for x in per.values()]
    final = v.INITIAL
    for p in profits:
        final *= 1 + p / v.INITIAL
    return {"per_season": per, "worst_season": min(profits), "total_profit": sum(profits),
            "max_dd": max(x["max_dd"] for x in per.values()),
            "bets": sum(x["n_bets"] for x in per.values()),
            "compounded_x": round(final / v.INITIAL, 3)}


def run() -> None:
    pack = pickle.loads(PACK.read_bytes())
    layers = pickle.loads(LAYERS.read_bytes())
    live = v.live_main_config()
    live.pop("min_raw_draw_prob", None)
    print("Live Main config:", {k: live[k] for k in sorted(live)})

    res: dict = {"accuracy": {}, "betting": {}}
    print("\nDraw accuracy, test seasons pooled (lower log loss is better):")
    print(f"{'variant':16s} {'mean%':>6} {'pinn%':>6} {'act%':>5} {'LL model':>9} {'LL pinn':>8}"
          f" {'n>=30%':>7} {'model|>=30':>10} {'pinn|>=30':>9} {'act|>=30':>8}")
    for var in VARIANTS:
        bts = {s: as_bt(layers[s], var) for s in SEASONS}
        both = pd.concat([bts[s] for s in TEST_SEASONS])
        dfs = pd.concat([pack[s][0] for s in TEST_SEASONS]).drop_duplicates(
            ["Date", "HomeTeam", "AwayTeam"])
        acc = accuracy(both, dfs)
        res["accuracy"][var] = acc
        print(f"{var:16s} {acc['mean_model']:>6} {acc['mean_pinnacle']:>6} {acc['draw_rate']:>5}"
              f" {acc['logloss_model']:>9} {acc['logloss_pinnacle']:>8} {acc['n_raw_ge_30']:>7}"
              f" {str(acc['model_when_raw_ge_30']):>10} {str(acc['pinnacle_when_raw_ge_30']):>9}"
              f" {str(acc['draw_rate_when_raw_ge_30']):>8}")

    print("\nBetting, live mechanics, Main's gates and Kelly:")
    hdr = (f"{'variant':16s} " + " ".join(f"{s:>9}" for s in TEST_SEASONS)
           + f" {'WORST':>8} {'maxDD':>6} {'bets':>5} {'x3yr':>6}  CLV by season")
    print(hdr)
    for var in VARIANTS:
        bts = {s: as_bt(layers[s], var) for s in SEASONS}
        r = simulate(pack, bts, live)
        res["betting"][var] = r
        clv = " ".join(str(r["per_season"][s]["median_clv"]) for s in TEST_SEASONS)
        print(f"{var:16s} " + " ".join(f"{r['per_season'][s]['profit']:>+9,.0f}"
                                     for s in TEST_SEASONS)
              + f" {r['worst_season']:>+8,.0f} {r['max_dd']:>5.1f}% {r['bets']:>5}"
              + f" {r['compounded_x']:>6.2f}  {clv}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"draw_fix_{ts}.json"
    path.write_text(json.dumps({"live_kw": LIVE_KW, "live": live, "results": res},
                               indent=2, default=str))
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
