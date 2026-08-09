"""Phase 1 diagnostic — slice Main's 52-week backtest by every dimension that
might leak edge, so Mock Two's new filters can be data-driven instead of guessed.

Slices:
  - Per-team P&L (home + away separately, plus combined)
  - Per-matchup P&L
  - Per-month / per-DOW
  - Per-EV-bucket calibration (do +20% EV bets actually convert?)
  - Per-prob-bucket calibration
  - Per-odds-bucket
  - Per-market (D vs under25)
  - Drawdown / streak sequencing
  - Cumulative bankroll path

Outputs:
  data/diagnostics/main_losses_<timestamp>.json   — machine-readable
  data/diagnostics/main_losses_findings.md        — human-readable

This is read-only. It does not touch portfolios, the live model, or live odds.
Reproduces the live Main config (min_prob=0.30, min_ev=0.40, kelly=1.0,
max_stake=0.33, markets=D+under25 with U2.5 separate gates, skip_late,
sim_correction on).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from data import load_data, add_rolling_features    # noqa: E402
from models import backtest_models                  # noqa: E402
import portfolio as pf                              # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_pnl(rows: list[dict], key_fn, label: str) -> list[dict]:
    """Aggregate bet rows by an arbitrary key, return per-group P&L stats."""
    groups: dict = defaultdict(lambda: {"n": 0, "won": 0, "stake": 0.0,
                                         "profit": 0.0})
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        g = groups[k]
        g["n"] += 1
        g["won"] += 1 if r["won"] else 0
        g["stake"] += r["stake"]
        g["profit"] += r["profit"]

    out = []
    for k, g in groups.items():
        roi = (g["profit"] / g["stake"] * 100) if g["stake"] > 0 else 0.0
        out.append({
            label: k,
            "n_bets":    g["n"],
            "win_rate":  round(g["won"] / g["n"] * 100, 1) if g["n"] else 0.0,
            "stake":     round(g["stake"], 2),
            "profit":    round(g["profit"], 2),
            "roi_pct":   round(roi, 2),
        })
    return sorted(out, key=lambda x: x["profit"])  # worst → best


def _ev_bucket(ev_val: float) -> str:
    if ev_val < 0.40: return "(filtered)"
    if ev_val < 0.50: return "0.40–0.50"
    if ev_val < 0.60: return "0.50–0.60"
    if ev_val < 0.70: return "0.60–0.70"
    if ev_val < 0.80: return "0.70–0.80"
    if ev_val < 1.00: return "0.80–1.00"
    return "1.00+"


def _prob_bucket(prob: float) -> str:
    if prob < 0.30: return "(filtered)"
    if prob < 0.35: return "0.30–0.35"
    if prob < 0.40: return "0.35–0.40"
    if prob < 0.45: return "0.40–0.45"
    if prob < 0.50: return "0.45–0.50"
    return "0.50+"


def _odds_bucket(odds: float) -> str:
    if odds < 1.50: return "<1.50"
    if odds < 2.00: return "1.50–2.00"
    if odds < 2.50: return "2.00–2.50"
    if odds < 3.00: return "2.50–3.00"
    if odds < 3.50: return "3.00–3.50"
    if odds < 4.00: return "3.50–4.00"
    if odds < 5.00: return "4.00–5.00"
    return "5.00+"


def _drawdown_stats(profits: list[float], initial: float) -> dict:
    """Max drawdown, longest losing streak, profit autocorrelation lag-1."""
    bankroll = [initial]
    for p in profits:
        bankroll.append(bankroll[-1] + p)
    bankroll_arr = np.array(bankroll)
    peaks = np.maximum.accumulate(bankroll_arr)
    drawdowns = (peaks - bankroll_arr) / peaks
    max_dd_pct = float(drawdowns.max()) * 100
    max_dd_at  = int(drawdowns.argmax())
    # Streak analysis
    longest_loss = cur_loss = 0
    longest_win  = cur_win  = 0
    for p in profits:
        if p < 0:
            cur_loss += 1; cur_win = 0
            longest_loss = max(longest_loss, cur_loss)
        elif p > 0:
            cur_win += 1; cur_loss = 0
            longest_win = max(longest_win, cur_win)
        else:
            cur_loss = cur_win = 0
    # Autocorrelation lag-1 on win/loss sequence (1=win, 0=loss)
    seq = np.array([1 if p > 0 else 0 for p in profits])
    if len(seq) > 2 and seq.std() > 0:
        ac1 = float(np.corrcoef(seq[:-1], seq[1:])[0, 1])
    else:
        ac1 = 0.0
    return {
        "max_drawdown_pct":      round(max_dd_pct, 2),
        "max_drawdown_at_bet":   max_dd_at,
        "longest_loss_streak":   longest_loss,
        "longest_win_streak":    longest_win,
        "ac1_lag1_winloss":      round(ac1, 4),
        "n_bets":                len(profits),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(test_weeks: int = 52) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{datetime.now():%H:%M:%S}] Loading data…")
    df = load_data()
    df_features = add_rolling_features(df)

    print(f"[{datetime.now():%H:%M:%S}] Fitting models + running {test_weeks}-week backtest…")
    bt_df = backtest_models(df, df_features, test_weeks=test_weeks)
    if bt_df.empty:
        print("EMPTY backtest output — aborting")
        return 1

    # Reproduce live Main config
    settings = json.load(open(ROOT / "data" / "portfolio.json"))["settings"]
    market_gates = {"under25": {"min_prob": 0.50, "min_ev": 0.05}}

    print(f"[{datetime.now():%H:%M:%S}] Running ev_backtest_simulate…")
    log_df, summary = pf.ev_backtest_simulate(
        bt_df, df,
        min_ev_pct=float(settings["min_ev"]) * 100,  # min_ev is fraction
        kelly_frac=float(settings["kelly_fraction"]),
        max_stake_pct=float(settings["max_stake_pct"]),
        initial_bankroll=10_000.0,
        allowed_markets=set(settings["auto_markets"]),
        min_prob=float(settings["min_prob"]),
        skip_late_season=bool(settings.get("skip_late_season", False)),
        skip_home_title_race=bool(settings.get("skip_home_title_race", False)),
        odds_source="Max",
        detect_source="PS",
        enable_simultaneous_correction=bool(settings.get("use_simultaneous_kelly", True)),
        market_gates=market_gates,
    )
    if "error" in summary:
        print(f"BACKTEST ERROR: {summary['error']}")
        return 1

    print(f"[{datetime.now():%H:%M:%S}] {len(log_df)} bets placed. Final bankroll: £{summary['final']:,.0f}")

    # Re-build raw rows so we can group on richer fields
    rows: list[dict] = []
    for _, r in log_df.iterrows():
        match = r["Match"]
        if " vs " in match:
            home, away = match.split(" vs ", 1)
        else:
            home, away = "?", "?"
        d = pd.to_datetime(r["Date"])
        market = r["Market"]
        market_code = ("D" if market == "Draw"
                       else "under25" if "Under" in market
                       else "over25" if "Over" in market
                       else "H" if "Home" in market
                       else "A" if "Away" in market
                       else market)
        # Parse Model% / EV strings back to floats
        try:
            prob = float(r["Model%"].rstrip("%")) / 100.0
        except Exception:
            prob = 0.0
        try:
            ev_val = float(r["EV"].lstrip("+").rstrip("%")) / 100.0
        except Exception:
            ev_val = 0.0

        rows.append({
            "date":   d,
            "home":   home,
            "away":   away,
            "market": market_code,
            "prob":   prob,
            "ev":     ev_val,
            "odds":   float(r["Odds"]),
            "stake":  float(r["Stake"]),
            "profit": float(r["Profit"]),
            "won":    r["Result"] == "✅",
        })

    n_bets = len(rows)
    if n_bets == 0:
        print("No bets to analyse")
        return 1

    # ── Group A: per-team (home / away / combined) ──────────────────────────
    by_home_team    = _group_pnl(rows, lambda r: r["home"],    "team_home")
    by_away_team    = _group_pnl(rows, lambda r: r["away"],    "team_away")

    # Combined per-team — sum of P&L when team X is in the match (either side)
    combined: dict = defaultdict(lambda: {"n": 0, "won": 0, "stake": 0.0,
                                           "profit": 0.0})
    for r in rows:
        for t in (r["home"], r["away"]):
            g = combined[t]
            g["n"] += 1
            g["won"] += 1 if r["won"] else 0
            g["stake"] += r["stake"]
            g["profit"] += r["profit"]
    by_team_combined = sorted([
        {
            "team":     t,
            "n_bets":   g["n"],
            "win_rate": round(g["won"] / g["n"] * 100, 1) if g["n"] else 0.0,
            "stake":    round(g["stake"], 2),
            "profit":   round(g["profit"], 2),
            "roi_pct":  round(g["profit"] / g["stake"] * 100, 2)
                          if g["stake"] > 0 else 0.0,
        }
        for t, g in combined.items()
    ], key=lambda x: x["profit"])

    # ── Group B: per-matchup ────────────────────────────────────────────────
    by_matchup = _group_pnl(rows, lambda r: f"{r['home']} vs {r['away']}", "matchup")

    # ── Group C: per-month / per-DOW ────────────────────────────────────────
    by_month = _group_pnl(rows, lambda r: r["date"].strftime("%b"), "month")
    by_dow   = _group_pnl(rows, lambda r: r["date"].strftime("%a"), "day_of_week")

    # ── Group D: per-EV-bucket / per-prob-bucket / per-odds-bucket ──────────
    by_ev   = _group_pnl(rows, lambda r: _ev_bucket(r["ev"]),     "ev_bucket")
    by_prob = _group_pnl(rows, lambda r: _prob_bucket(r["prob"]), "prob_bucket")
    by_odds = _group_pnl(rows, lambda r: _odds_bucket(r["odds"]), "odds_bucket")

    # ── Group E: per-market ─────────────────────────────────────────────────
    by_market = _group_pnl(rows, lambda r: r["market"], "market")

    # ── Group F: drawdown / streaks ─────────────────────────────────────────
    profits = [r["profit"] for r in rows]
    dd = _drawdown_stats(profits, 10_000.0)

    # ── Persist ─────────────────────────────────────────────────────────────
    findings = {
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
        "test_weeks":      test_weeks,
        "config":          {**{k: v for k, v in settings.items() if k != "odds_api_key"},
                            "odds_source": "Max", "detect_source": "PS"},
        "summary":         summary,
        "drawdown_stats":  dd,
        "by_team_combined": by_team_combined,
        "by_home_team":    by_home_team,
        "by_away_team":    by_away_team,
        "by_matchup":      by_matchup,
        "by_month":        by_month,
        "by_dow":          by_dow,
        "by_ev_bucket":    by_ev,
        "by_prob_bucket":  by_prob,
        "by_odds_bucket":  by_odds,
        "by_market":       by_market,
    }

    json_path = OUT_DIR / f"main_losses_{ts}.json"
    json_path.write_text(json.dumps(findings, indent=2, default=str))
    print(f"[{datetime.now():%H:%M:%S}] JSON written: {json_path.relative_to(ROOT)}")

    # ── Human-readable findings markdown ────────────────────────────────────
    md_path = OUT_DIR / "main_losses_findings.md"
    md_path.write_text(_render_markdown(findings))
    print(f"[{datetime.now():%H:%M:%S}] Markdown written: {md_path.relative_to(ROOT)}")

    # ── Console highlights ──────────────────────────────────────────────────
    print("\n=== HIGHLIGHTS ===")
    print(f"Final bankroll:    £{summary['final']:,.0f}")
    print(f"ROI:               {summary['roi']:+.1f}%")
    print(f"Bets placed:       {summary['n_bets']}  ({summary['win_rate']}% win)")
    print(f"Max drawdown:      {dd['max_drawdown_pct']:.1f}%")
    print(f"Longest loss run:  {dd['longest_loss_streak']} bets")
    print(f"AC1 win/loss:      {dd['ac1_lag1_winloss']:+.4f}  "
          f"({'streaks' if dd['ac1_lag1_winloss'] > 0.1 else 'random'})")

    print("\n=== BOTTOM-10 TEAMS (combined home + away involvement) ===")
    for t in by_team_combined[:10]:
        print(f"  {t['team']:25s}  n={t['n_bets']:3d}  "
              f"win={t['win_rate']:5.1f}%  stake £{t['stake']:>9,.0f}  "
              f"P&L £{t['profit']:>+10,.0f}  ROI {t['roi_pct']:+6.1f}%")

    print("\n=== TOP-10 TEAMS ===")
    for t in by_team_combined[-10:][::-1]:
        print(f"  {t['team']:25s}  n={t['n_bets']:3d}  "
              f"win={t['win_rate']:5.1f}%  stake £{t['stake']:>9,.0f}  "
              f"P&L £{t['profit']:>+10,.0f}  ROI {t['roi_pct']:+6.1f}%")

    print("\n=== EV-BUCKET CALIBRATION ===")
    for b in by_ev:
        print(f"  {b['ev_bucket']:12s}  n={b['n_bets']:3d}  "
              f"win={b['win_rate']:5.1f}%  ROI {b['roi_pct']:+6.1f}%")

    print("\n=== PROB-BUCKET CALIBRATION ===")
    for b in by_prob:
        print(f"  {b['prob_bucket']:12s}  n={b['n_bets']:3d}  "
              f"win={b['win_rate']:5.1f}%  ROI {b['roi_pct']:+6.1f}%")

    print("\n=== PER-MONTH ===")
    for b in by_month:
        print(f"  {b['month']:6s}  n={b['n_bets']:3d}  "
              f"win={b['win_rate']:5.1f}%  ROI {b['roi_pct']:+6.1f}%")

    print("\n=== PER-DOW ===")
    for b in by_dow:
        print(f"  {b['day_of_week']:6s}  n={b['n_bets']:3d}  "
              f"win={b['win_rate']:5.1f}%  ROI {b['roi_pct']:+6.1f}%")

    print("\n=== PER-MARKET ===")
    for b in by_market:
        print(f"  {b['market']:8s}  n={b['n_bets']:3d}  "
              f"win={b['win_rate']:5.1f}%  ROI {b['roi_pct']:+6.1f}%")

    return 0


def _render_markdown(f: dict) -> str:
    lines = []
    lines.append(f"# Main 52-week loss diagnostic")
    lines.append(f"_Generated {f['generated_at']}_")
    lines.append("")
    lines.append("## Summary")
    s = f["summary"]; dd = f["drawdown_stats"]
    lines.append(f"- Final bankroll: **£{s['final']:,.0f}**  (start £10,000)")
    lines.append(f"- ROI: **{s['roi']:+.1f}%**  on £{s['total_staked']:,.0f} staked")
    lines.append(f"- Bets placed: **{s['n_bets']}**  ({s['win_rate']}% win)")
    lines.append(f"- Avg odds: {s['avg_odds']}")
    lines.append(f"- Max drawdown: **{dd['max_drawdown_pct']:.1f}%** "
                 f"(at bet #{dd['max_drawdown_at_bet']})")
    lines.append(f"- Longest losing run: **{dd['longest_loss_streak']}** bets")
    lines.append(f"- Win/loss autocorrelation lag-1: {dd['ac1_lag1_winloss']:+.4f}")
    lines.append("")

    def _table(label_col: str, label_key: str, rows_: list[dict],
               fmt_label=lambda v: v) -> list[str]:
        out = []
        out.append(f"| {label_col} | n | win% | stake | P&L | ROI |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for row in rows_:
            out.append(
                f"| {fmt_label(row[label_key])} | {row['n_bets']} | "
                f"{row['win_rate']}% | £{row['stake']:,.0f} | "
                f"£{row['profit']:+,.0f} | {row['roi_pct']:+.1f}% |"
            )
        return out

    lines.append("## Per-team (combined home + away involvement)")
    lines.append("")
    lines.append("### Worst 15 teams")
    lines.extend(_table("Team", "team", f["by_team_combined"][:15]))
    lines.append("")
    lines.append("### Best 15 teams")
    lines.extend(_table("Team", "team", f["by_team_combined"][-15:][::-1]))
    lines.append("")

    lines.append("## Per-team — home only")
    lines.append("### Worst 10")
    lines.extend(_table("Home", "team_home", f["by_home_team"][:10]))
    lines.append("")
    lines.append("### Best 10")
    lines.extend(_table("Home", "team_home", f["by_home_team"][-10:][::-1]))
    lines.append("")

    lines.append("## Per-team — away only")
    lines.append("### Worst 10")
    lines.extend(_table("Away", "team_away", f["by_away_team"][:10]))
    lines.append("")
    lines.append("### Best 10")
    lines.extend(_table("Away", "team_away", f["by_away_team"][-10:][::-1]))
    lines.append("")

    lines.append("## EV-bucket calibration")
    lines.extend(_table("EV bucket", "ev_bucket", f["by_ev_bucket"][::-1]))
    lines.append("")

    lines.append("## Probability-bucket calibration")
    lines.extend(_table("Prob bucket", "prob_bucket", f["by_prob_bucket"][::-1]))
    lines.append("")

    lines.append("## Odds-bucket")
    lines.extend(_table("Odds", "odds_bucket", f["by_odds_bucket"][::-1]))
    lines.append("")

    lines.append("## Per-month")
    lines.extend(_table("Month", "month", f["by_month"][::-1]))
    lines.append("")

    lines.append("## Per-day-of-week")
    lines.extend(_table("DOW", "day_of_week", f["by_dow"][::-1]))
    lines.append("")

    lines.append("## Per-market")
    lines.extend(_table("Market", "market", f["by_market"][::-1]))
    lines.append("")

    lines.append("## Top-20 worst matchups")
    lines.extend(_table("Matchup", "matchup", f["by_matchup"][:20]))
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main(test_weeks=52))
