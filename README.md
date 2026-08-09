# Premier League Match Predictor

A full-stack football analytics app that predicts Premier League match outcomes using an ensemble of statistical and machine learning models, with a mock betting portfolio powered by live odds.

Built with Python, Streamlit, and Plotly.

![App Screenshot](screenshot.png)

![Mock Portfolio Backtest](backtest.png)

## Features

**Tile-based landing screen** — Opens on a simplistic home view: large Premier League symbol, gradient title, and a 7-tile grid that routes to each feature. Compact top-bar on sub-pages keeps focus on the current tab's content.

**Match Prediction** — Select any two PL teams and get win/draw/loss probabilities, most likely scorelines, score matrix heatmap, goal distributions, recent form, head-to-head history, and a "Why this prediction" panel surfacing the 2–3 most differentiating rolling stats (xG, form, venue record, Elo gap).

**Weekend / Next-Gameweek Fixtures** — Auto-fetched from ESPN (weekend *and* midweek rounds) with probability bars, team badges, expected goals, and — when an Odds API key is set — a value-vs-market edge badge on every fixture.

**Gameweek Review** — Compare model predictions against actual results with accuracy stats, best calls, biggest surprises, and large team badges.

**Backtesting** — Evaluate model performance on historical data with configurable test windows, accuracy metrics, Brier scores, **Expected Calibration Error (ECE)**, and a reliability diagram. Includes a **diagnostics matrix** that slices the Draw-model residuals by model-prob bucket, Pinnacle close-odds bucket, |xG diff|, and team — surfacing where the model is mispriced — and an **equaliser-vs-winner empirical study** testing late-game goal asymmetry across 5 PL seasons.

**Season Simulation** — Monte Carlo simulation (10,000 seasons) with parametric bootstrap on the Dixon-Coles ratings for honest title-race uncertainty, plus a team drill-down showing remaining fixtures, expected points per match, and the distribution of projected final points.

**Mock Portfolio (Main)** — Paper trading with Kelly criterion sizing, auto-bet, **multi-bookmaker odds support** (Bet365 / Max-of-panel / Avg / Pinnacle), live odds integration, value scanner, bankroll tracking with green/red fill zones and win/loss arrow markers, **isotonic probability calibration** fitted from backtest data, **per-market gates** (Draws and Under 2.5 use independently-tuned thresholds), **simultaneous-bet correction** (reduces stakes when multiple bets settle the same day), **temporal filters** (skips Mar-May where the model has historically gone 0/7), and a **CLV (closing-line value) sharpness panel** with rolling trend dashboard that flags edge drift before ROI does.

**Mock Portfolio Two — Research-Track A/B** — A parallel paper-trading portfolio running an experimental stack alongside the main model, so you can A/B-compare without risking a working production setup. Implements:
- **Karlis-Ntzoufras γ-inflation** Dixon-Coles variant (lifts the draw diagonal)
- **Baker-McHale uncertainty-shrunk Kelly** (per-bin Var(p̂) shrinkage)
- **Simultaneous-bet Kelly correction** (Saturday-card concurrent-settlement adjustment)
- **Detect-at-Pinnacle / place-at-best-book** workflow — measure EV against the sharpest reference price while collecting payoffs at the highest available bookmaker, the standard sharp-bettor approach.

Includes a side-by-side comparison header (P&L, ROI, settled count, median CLV, bankroll for both portfolios) and a verdict banner. End-of-season delta on CLV decides what graduates to Main.

## Models

The prediction engine blends five models into a single probability estimate:

| Model | Role | Detail |
|-------|------|--------|
| **Dixon-Coles** | Primary (85%) | MLE-fitted attack/defence ratings with xG-based Poisson likelihood and low-score tau correction. 14-week exponential decay. |
| **XGBoost 3-Class** | Secondary (15%) | H/D/A classifier trained on ~40 rolling form features (goals, xG, Elo, venue stats, days rest). |
| **Draw Specialist DC** | Draw adjustment | Dixon-Coles variant with 3x draw-weighted training. Fitted rho ~ -0.46 vs standard -0.17. |
| **XGBoost Binary Draw** | Draw adjustment | Dedicated draw detector with xG convergence, Elo gap, and draw-prone interaction features. |
| **Poisson** | Baseline | Time-decayed multiplicative attack/defence ratings for comparison. |

**Ensemble pipeline:** DC (85%) + XGB (15%) -> Draw Specialist replaces draw probability at 35% weight -> H/A re-normalised.

### Research-track variant (Mock Portfolio Two)

Mock Two swaps the standard Dixon-Coles for the **Karlis & Ntzoufras (2003) diagonal-inflated bivariate Poisson** (`compute_dixon_coles_kn_ratings`) — adds an MLE-fitted γ parameter that lifts probability mass on the entire draw diagonal, where standard D-C only adjusts (0,0)/(1,1)/(0,1)/(1,0). γ is bounded at ±0.20 (penaltyblog convention; data wants more). Stake sizing uses **Baker-McHale (2013) uncertainty-shrunk Kelly** with shrinkage factor `k = 1 - Var(p̂) / (p̂(1-p̂))` computed per probability bin from the backtest, plus **Busseti-Ryu-Boyd-style simultaneous-bet correction** for concurrently-settling Saturday cards.

## Mock Portfolio Performance

The deployed config has been **walk-forward cross-validated across 14 folds** (10 OOS + 4 IS) with CLV measurement per bet, then confirmed via random search across 60 parameter configurations. The current settings sit at the OOS profit optimum.

### In-sample 52-week backtest (B365 + filters + per-market gates)

| Metric | Value |
|--------|-------|
| Starting bankroll | £10,000 |
| Final bankroll | **£93,808** |
| Total bets | 74 |
| ROI | +19.8% |
| Markets | Draw + Under 2.5 (independent gates) |

### Walk-forward CV (10 OOS folds spanning 2022-09 to 2025-05)

| Config | OOS profit (sum of 10 folds) | Median CLV | Bets beating close |
|--------|------------------------------|------------|---------------------|
| Pre-retune live config | **−£40,810** | −2.51% | 24.5% |
| Current Main settings | +£14,000 (estimate) | −0.20% | 49.1% |
| Mock Two (K-N + B-M Kelly) | +£12,064 | +2.68% | **77.7%** |

The Mock Two stack produces lower absolute profit than the simpler Main stack but generates the strongest sharpness signal — 77.7% of its bets get prices better than Pinnacle's close, the standard pro-bettor benchmark.

## Tech Stack

- **Python** — pandas, numpy, scipy, scikit-learn, xgboost
- **Streamlit** — interactive UI with custom CSS dark theme
- **Plotly** — charts, heatmaps, and visualisations
- **The Odds API** — live betting odds (free tier)
- **ESPN API** — fixtures and team badges
- **football-data.co.uk** — historical results and odds
- **Understat** — expected goals (xG) data

## Quick Start

```bash
git clone https://github.com/EoinHoustoun/Football_Match_Predictor.git
cd Football_Match_Predictor
bash run.sh
```

This installs dependencies and opens the app at `http://localhost:8501`.

**Requirements:** Python 3.9+. All data is fetched automatically on first run.

**Optional:** Add a free [The Odds API](https://the-odds-api.com/) key in the Portfolio tab settings for live odds integration.

## Project Structure

```
app.py         — Streamlit UI (home screen + 7 tabs, Plotly charts, team badges, dark theme)
                  · tab_portfolio (Main) and tab_portfolio_two (research-track A/B)
                  · full_predict / full_predict_v2 (K-N research variant)
data.py        — Data loading, feature engineering, Elo ratings, xG enrichment, ESPN fixtures
                  · keeps Pinnacle close (PSH/PSD/PSA), industry avg close, HT goals
models.py      — Dixon-Coles MLE, Poisson, XGBoost, draw specialist, simulate_season
                  · Karlis-Ntzoufras γ-inflation variant (compute_dixon_coles_kn_ratings)
portfolio.py   — Betting engine for both portfolios + research-track Kelly machinery
                  · auto_place_value_bets / auto_place_value_bets_v2 (research)
                  · kelly_stake_uncertainty_adjusted (Baker-McHale shrinkage)
                  · simultaneous_kelly_correction (concurrent-bet Kelly)
                  · CLV: extract_closing_odds, compute_clv, backfill_clv_for_settled_bets
                  · clv_summary, compute_brier_drift (calibration drift detector)
                  · compute_per_bin_variance, lookup_bin_variance
                  · load_portfolio_two, save_portfolio_two (Mock Two persistence)
tests/         — pytest suite (63 tests across three files)
assets/        — Embedded logos (Premier League symbol used as app hero)
run.sh         — Startup script
CLAUDE.md      — Working guide for AI-pair-programming sessions
```

## Tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```

63 tests across three files:

- **`test_models.py`** (15) — Dixon-Coles probability math, home-advantage effects, season-simulation invariants (probabilities sum to 1, noise widens title race, points monotonic).
- **`test_portfolio.py`** (17) — Kelly edge cases, bankroll-scaling, EV math, isotonic calibration correctness, the min-probability auto-bet gate.
- **`test_research_track.py`** (31) — Karlis-Ntzoufras γ variant (γ=0 reduces exactly to standard D-C, γ>0 lifts draw, γ<0 reduces draw, MLE convergence on synthetic data); Baker-McHale uncertainty-shrunk Kelly (zero-variance matches flat Kelly, max-variance collapses to zero stake, intermediate variance partial shrinkage); simultaneous-bet correction (clamping, single-bet identity, three-concurrent reduction); CLV computation + closing-odds extraction with Pinnacle/Avg/B365 fallback chain; per-bin variance helpers; Mock Two persistence.
