"""Mock Two research: size stakes from a market-shrunk edge, not the raw model EV.

Background (17 Oct 2026): on Main's 33 settled 2026-27 bets the model expected
12.3 wins, the closing market 9.1, and 10 landed. Median model EV is +44%;
median CLV is +4%. The edge looks real but an order of magnitude smaller than
the model says, and Kelly stakes are sized off the big number.

What this tests. Selection is UNCHANGED: the same gates pick the same bets.
Only the probability fed to Kelly changes, pulled toward the de-vigged
Pinnacle price in logit space:

    logit p_size = logit m + a + beta * (logit p_model - logit m)

beta = 1 is the incumbent exactly; beta = 0 trusts the market entirely. A
smaller p_size can only shrink a stake, so every variant is a tightening.

Two ways to choose beta:
  1. Fixed grid (robustness: does a REGION work, or one spike?).
  2. Hierarchical, walk-forward. Each past season gets its own (a, beta) by
     offset logistic regression of draw outcomes on the model-vs-market gap;
     seasons are partially pooled (normal-normal, DerSimonian-Laird tau^2).
     The current season's estimate is shrunk toward the pool by its own
     precision, so early in a season it sits near the pooled value and moves
     only as evidence accumulates. Refit every gameweek on matches strictly
     before it.

Mechanics match live: gameweek staking, 35% pro-rata exposure cap, raw draw
floor 0.30, trailing 280-day isotonic calibrator, detect at Pinnacle, place at
best price, Main's deployed gates and Kelly fraction read from portfolio.json.

Usage: .venv/bin/python scripts/research_shrunk_sizing.py
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import portfolio as pf  # noqa: E402
import validate_2026_27 as v  # noqa: E402
from validate_draw_calibration import make_fn  # noqa: E402

PACK = ROOT / "data" / "diagnostics" / "exposure_cap_pack.pkl"
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
LIVE_KW = dict(gameweek_mode=True, max_pending_bets=None, exposure_cap_pct=0.35,
               exposure_cap_basis="gameweek", exposure_cap_mode="prorata",
               min_raw_draw_prob=0.30)
BETAS = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]
KELLYS = [0.5, 1.0]
EPS = 1e-4
# Fit only where we bet: the raw draw floor. Across ALL matches the model adds
# nothing over Pinnacle (beta ~ 0); the question is whether it does above 0.30.
REGION_FLOOR = 0.30
MIN_ROWS = 30


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def market_draw(psh, psd, psa) -> float:
    inv = np.array([1 / psh, 1 / psd, 1 / psa])
    return float(inv[1] / inv.sum())


# ── history: every OOS match with raw model draw prob, Pinnacle prob, outcome ──
def build_history(pack) -> pd.DataFrame:
    rows = []
    for s, (dfs, _ftr, bt, _) in pack.items():
        if s == "2026-27":
            continue
        odds = dfs[["Date", "HomeTeam", "AwayTeam", "PSH", "PSD", "PSA"]].rename(
            columns={"HomeTeam": "Home", "AwayTeam": "Away"}).dropna()
        m = bt[["Date", "Home", "Away", "_dc_d", "_act_d"]].merge(
            odds, on=["Date", "Home", "Away"], how="inner")
        m = m[(m["PSH"] > 1) & (m["PSD"] > 1) & (m["PSA"] > 1)]
        m["mkt_d"] = [market_draw(a, b, c) for a, b, c in zip(m["PSH"], m["PSD"], m["PSA"])]
        m["season"] = s
        rows.append(m)
    return pd.concat(rows, ignore_index=True).sort_values("Date")


# ── offset logistic: y ~ Bern(sigmoid(offset + a + beta * x)) by Newton ──
def fit_offset_logit(x, off, y, ridge=1e-3):
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(50):
        p = sigmoid(off + X @ w)
        g = X.T @ (y - p) - ridge * w
        H = -(X.T * (p * (1 - p))) @ X - ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    p = sigmoid(off + X @ w)
    cov = np.linalg.inv((X.T * (p * (1 - p))) @ X + ridge * np.eye(2))
    return w, cov


def pooled_beta(hist: pd.DataFrame, current_season: str) -> tuple[float, float]:
    """Partially pooled (a, beta) for the current season from strictly-past rows."""
    ests = []
    cur = None
    for s, g in hist.groupby("season"):
        if len(g) < MIN_ROWS:
            if s == current_season and len(g) > 0:
                cur = g
            continue
        x = (logit(g["_dc_d"].to_numpy()) - logit(g["mkt_d"].to_numpy()))
        w, cov = fit_offset_logit(x, logit(g["mkt_d"].to_numpy()), g["_act_d"].to_numpy(float))
        if s == current_season:
            cur = (w, cov)
        else:
            ests.append((w, cov))
    if not ests:
        return 0.0, 1.0
    W = np.array([e[0] for e in ests])
    V = np.array([np.diag(e[1]) for e in ests])
    # DerSimonian-Laird per parameter
    mu, tau2 = np.zeros(2), np.zeros(2)
    for j in range(2):
        wj = 1 / V[:, j]
        m0 = np.sum(wj * W[:, j]) / wj.sum()
        if len(ests) > 1:
            Q = np.sum(wj * (W[:, j] - m0) ** 2)
            c = wj.sum() - np.sum(wj ** 2) / wj.sum()
            tau2[j] = max(0.0, (Q - (len(ests) - 1)) / c)
        wj2 = 1 / (V[:, j] + tau2[j])
        mu[j] = np.sum(wj2 * W[:, j]) / wj2.sum()
    # A between-season prior that is never zero: one season of history cannot
    # prove seasons agree. Floor tau at 0.15 on both parameters.
    tau2 = np.maximum(tau2, 0.15 ** 2)
    out = mu.copy()
    if isinstance(cur, tuple):
        wc, cc = cur
        for j in range(2):
            k = tau2[j] / (tau2[j] + cc[j, j])
            out[j] = mu[j] + k * (wc[j] - mu[j])
    return float(out[0]), float(out[1])


def make_hier_fn(hist: pd.DataFrame, log: list):
    cache: dict = {}
    seasons = {s: (g["Date"].min(), g["Date"].max()) for s, g in hist.groupby("season")}

    def season_of(d):
        for s, (lo, hi) in seasons.items():
            if lo - pd.Timedelta(days=60) <= d <= hi:
                return s
        return None

    def fn(prob, mkt, r):
        if mkt != "D":
            return prob
        d = pd.Timestamp(r["Date"])
        wk = d - pd.Timedelta(days=d.weekday())
        if wk not in cache:
            past = hist[hist["Date"] < wk]
            cache[wk] = pooled_beta(past, season_of(d))
            log.append({"week": str(wk.date()), "a": cache[wk][0], "beta": cache[wk][1]})
        a, beta = cache[wk]
        m = market_draw(float(r["PSH"]), float(r["PSD"]), float(r["PSA"]))
        raw = float(r["DC_D"]) / 100.0
        return float(sigmoid(logit(m) + a + beta * (logit(raw) - logit(m))))
    return fn


def make_grid_fn(beta: float):
    def fn(prob, mkt, r):
        if beta == 1.0 or mkt != "D":
            return prob
        m = market_draw(float(r["PSH"]), float(r["PSD"]), float(r["PSA"]))
        return float(sigmoid(logit(m) + beta * (logit(prob) - logit(m))))
    return fn


def run(pack, cal_fn, size_fn, live: dict) -> dict:
    per: dict = {}
    for s in TEST_SEASONS:
        dfs, ftr, bt, _ = pack[s]
        log, summ = pf.ev_backtest_simulate(
            bt, dfs, df_features=ftr, initial_bankroll=v.INITIAL,
            enable_simultaneous_correction=True, odds_source="Max",
            detect_source="PS", calibrators=None, calibrator_fn=cal_fn,
            sizing_prob_fn=size_fn, **LIVE_KW, **live)
        if "error" in summ or log.empty:
            per[s] = {"profit": 0.0, "n_bets": 0, "max_dd": 0.0, "median_stake_pct": 0.0}
            continue
        stake_col = "Stake" if "Stake" in log.columns else None
        med_stake = None
        if stake_col and "Bankroll" in log.columns:
            med_stake = round(float((log[stake_col] / (log["Bankroll"] - log.get("Profit", 0))).median() * 100), 2)
        per[s] = {"profit": round(summ["profit"], 0), "n_bets": summ["n_bets"],
                  "max_dd": round(v._max_drawdown_pct(log), 1),
                  "median_clv": v._median_clv(log, dfs),
                  "median_stake_pct": med_stake}
    profits = [x["profit"] for x in per.values()]
    final = v.INITIAL
    for p in profits:
        final *= 1 + p / v.INITIAL
    return {"per_season": per, "worst_season": min(profits), "total_profit": sum(profits),
            "max_dd": max(x["max_dd"] for x in per.values()),
            "bets": sum(x["n_bets"] for x in per.values()),
            "compounded_x": round(final / v.INITIAL, 3)}


def main() -> int:
    pack = pickle.loads(PACK.read_bytes())
    history_bt = pd.concat([pack[s][2] for s in pack if s != "2026-27"],
                           ignore_index=True).sort_values("Date")
    cal_fn = make_fn(history_bt, "isotonic", 280)
    hist = build_history(pack)
    live = v.live_main_config()
    live.pop("min_raw_draw_prob", None)
    base_kelly = live.pop("kelly_frac")

    # Diagnostic: how much of the model-vs-market gap is real, per season
    print(f"Per-season offset-logit fit (raw draw >= {REGION_FLOOR}, model vs Pinnacle):")
    diag = {}
    for s, g in hist[hist["_dc_d"] >= REGION_FLOOR].groupby("season"):
        x = logit(g["_dc_d"].to_numpy()) - logit(g["mkt_d"].to_numpy())
        w, cov = fit_offset_logit(x, logit(g["mkt_d"].to_numpy()), g["_act_d"].to_numpy(float))
        diag[s] = {"n": len(g), "a": round(w[0], 3), "beta": round(w[1], 3),
                   "beta_se": round(float(np.sqrt(cov[1, 1])), 3)}
        print(f"  {s}: n={len(g):4d}  a={w[0]:+.3f}  beta={w[1]:.3f} ± {np.sqrt(cov[1,1]):.3f}")

    results: dict = {}
    hdr = (f"{'variant':26s} " + " ".join(f"{s:>9}" for s in TEST_SEASONS)
           + f" {'WORST':>8} {'maxDD':>6} {'bets':>5} {'x3yr':>6}")
    for k in KELLYS:
        live_k = dict(live, kelly_frac=k)
        print(f"\n== Kelly fraction {k}" + ("  (LIVE)" if k == base_kelly else ""))
        print(hdr)
        variants = {f"grid beta={b}" + (" (incumbent)" if b == 1.0 else ""): make_grid_fn(b)
                    for b in BETAS}
        hier_log: list = []
        variants["hierarchical WF"] = make_hier_fn(hist[hist["_dc_d"] >= REGION_FLOOR], hier_log)
        for name, fn in variants.items():
            r = run(pack, cal_fn, fn, live_k)
            results[f"kelly {k} | {name}"] = r
            print(f"{name:26s} "
                  + " ".join(f"{r['per_season'][s]['profit']:>+9,.0f}" for s in TEST_SEASONS)
                  + f" {r['worst_season']:>+8,.0f} {r['max_dd']:>5.1f}% {r['bets']:>5}"
                  + f" {r['compounded_x']:>6.2f}")
        if hier_log:
            betas = [h["beta"] for h in hier_log]
            print(f"   hierarchical beta over time: min {min(betas):.2f}  "
                  f"median {np.median(betas):.2f}  max {max(betas):.2f}")
            results[f"kelly {k} | hierarchical beta path"] = hier_log

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ROOT / "data" / "diagnostics" / f"shrunk_sizing_{ts}.json"
    path.write_text(json.dumps({"live_kw": LIVE_KW, "seasons": TEST_SEASONS,
                                "season_fits": diag, "results": results},
                               indent=2, default=str))
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
