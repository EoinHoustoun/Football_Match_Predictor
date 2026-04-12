"""
Premier League Match Predictor — Streamlit App
Poisson · Dixon-Coles · XGBoost ensemble
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio as pf
from data import (
    add_rolling_features,
    fetch_remaining_season_fixtures,
    fetch_upcoming_fixtures,
    get_current_elo,
    get_current_stats,
    get_current_table,
    get_current_teams,
    get_head_to_head,
    get_team_form,
    load_data,
)
from models import (
    backtest_models,
    blend,
    blend_dc,
    blend_draw_specialist,
    compute_backtest_summary,
    compute_dixon_coles_ratings,
    compute_draw_dc_ratings,
    compute_poisson_ratings,
    predict_dixon_coles,
    predict_draw_xgb,
    predict_poisson,
    predict_xgb,
    simulate_season,
    train_draw_xgb,
    train_xgb,
)

# ─────────────────────────────────────────────────────────────────────────────
# Team badges (ESPN CDN)
# ─────────────────────────────────────────────────────────────────────────────
_BADGE_URL: dict[str, str] = {
    "Arsenal":        "https://a.espncdn.com/i/teamlogos/soccer/500/359.png",
    "Aston Villa":    "https://a.espncdn.com/i/teamlogos/soccer/500/362.png",
    "Bournemouth":    "https://a.espncdn.com/i/teamlogos/soccer/500/349.png",
    "Brentford":      "https://a.espncdn.com/i/teamlogos/soccer/500/337.png",
    "Brighton":       "https://a.espncdn.com/i/teamlogos/soccer/500/331.png",
    "Burnley":        "https://a.espncdn.com/i/teamlogos/soccer/500/379.png",
    "Chelsea":        "https://a.espncdn.com/i/teamlogos/soccer/500/363.png",
    "Crystal Palace": "https://a.espncdn.com/i/teamlogos/soccer/500/384.png",
    "Everton":        "https://a.espncdn.com/i/teamlogos/soccer/500/368.png",
    "Fulham":         "https://a.espncdn.com/i/teamlogos/soccer/500/370.png",
    "Ipswich":        "https://a.espncdn.com/i/teamlogos/soccer/500/373.png",
    "Leeds":          "https://a.espncdn.com/i/teamlogos/soccer/500/357.png",
    "Leicester":      "https://a.espncdn.com/i/teamlogos/soccer/500/375.png",
    "Liverpool":      "https://a.espncdn.com/i/teamlogos/soccer/500/364.png",
    "Man City":       "https://a.espncdn.com/i/teamlogos/soccer/500/382.png",
    "Man United":     "https://a.espncdn.com/i/teamlogos/soccer/500/360.png",
    "Newcastle":      "https://a.espncdn.com/i/teamlogos/soccer/500/361.png",
    "Nott'm Forest":  "https://a.espncdn.com/i/teamlogos/soccer/500/393.png",
    "Southampton":    "https://a.espncdn.com/i/teamlogos/soccer/500/376.png",
    "Sunderland":     "https://a.espncdn.com/i/teamlogos/soccer/500/366.png",
    "Tottenham":      "https://a.espncdn.com/i/teamlogos/soccer/500/367.png",
    "West Ham":       "https://a.espncdn.com/i/teamlogos/soccer/500/371.png",
    "Wolves":         "https://a.espncdn.com/i/teamlogos/soccer/500/380.png",
}


def badge(team: str, size: int = 20) -> str:
    """Return an <img> tag for the team badge, or empty string if not found."""
    url = _BADGE_URL.get(team, "")
    if not url:
        return ""
    return (
        f'<img src="{url}" width="{size}" height="{size}" '
        f'style="vertical-align:middle;margin-right:4px;border-radius:2px" />'
    )


def tb(team: str, size: int = 20) -> str:
    """Team name with badge: <img> + name."""
    return f'{badge(team, size)}{team}'


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0a0e1a; }
.main .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1300px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 0.4rem;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #556 !important;
    font-weight: 600;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c4dff22, #00e5ff22) !important;
    color: #ccd !important;
    border-bottom: 2px solid #7c4dff !important;
}

/* ── Header ── */
.pl-header { text-align: center; padding: 2rem 0 1rem; }
.pl-title {
    font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 900;
    background: linear-gradient(135deg, #7c4dff 0%, #00e5ff 60%, #00ff87 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.1; margin-bottom: 0.4rem;
}
.pl-subtitle { font-size: 0.9rem; color: #556; letter-spacing: 2px; text-transform: uppercase; font-weight: 500; }
.pl-meta { font-size: 0.75rem; color: #445; margin-top: 0.6rem; }

/* ── Divider ── */
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(124,77,255,0.4), transparent); margin: 2rem 0; }

/* ── Section label ── */
.section-label { font-size: 0.7rem; font-weight: 700; color: #556; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 1.2rem; }

.vs-badge { text-align: center; font-size: 1.6rem; font-weight: 900; color: #334; padding-top: 1.9rem; }

/* ── Probability cards ── */
.prob-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 1.8rem 1rem; text-align: center;
    position: relative; overflow: hidden;
}
.prob-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; }
.prob-card-home::before { background: linear-gradient(90deg, #3d6eff, #00b4ff); }
.prob-card-draw::before { background: linear-gradient(90deg, #ffd600, #ff6f00); }
.prob-card-away::before { background: linear-gradient(90deg, #ff4081, #f50057); }

.prob-team { font-size: 0.95rem; font-weight: 700; color: #aab; margin-bottom: 0.6rem;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob-pct  { font-size: 3.2rem; font-weight: 900; line-height: 1; margin-bottom: 0.3rem; }
.prob-pct-home { color: #3d6eff; }
.prob-pct-draw { color: #ffd600; }
.prob-pct-away { color: #ff4081; }
.prob-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #556; }

/* ── Model comparison cards ── */
.model-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 0.5rem;
}
.model-title { font-size: 0.65rem; font-weight: 700; color: #556; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.8rem; }

/* ── Scoreline card ── */
.score-hero {
    background: linear-gradient(135deg, rgba(124,77,255,0.15), rgba(0,229,255,0.08));
    border: 1px solid rgba(124,77,255,0.35); border-radius: 24px; padding: 2.5rem 2rem; text-align: center;
}
.score-hero-label { font-size: 0.65rem; font-weight: 700; color: #556; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1rem; }
.score-digits { font-size: clamp(3.5rem, 8vw, 5.5rem); font-weight: 900; color: #e8eaf0; letter-spacing: -2px; line-height: 1; }
.score-dash { color: #334; margin: 0 0.5rem; }
.score-prob { margin-top: 0.8rem; font-size: 0.8rem; color: #7c4dff; font-weight: 600; }
.score-ci { margin-top: 1.2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; }
.ci-item { font-size: 0.75rem; color: #778; }
.ci-range { font-weight: 700; color: #aab; }

/* ── Top scorelines table ── */
.score-table { width: 100%; border-collapse: collapse; }
.score-table th { font-size: 0.65rem; font-weight: 700; color: #445; text-transform: uppercase;
                  letter-spacing: 2px; padding: 0.4rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.06); text-align: left; }
.score-table td { padding: 0.55rem 0.8rem; font-size: 0.85rem; color: #ccd; border-bottom: 1px solid rgba(255,255,255,0.04); }
.score-table tr:hover td { background: rgba(124,77,255,0.06); }
.score-table tr:first-child td { font-weight: 700; color: #e8eaf0; }
.badge-H { background: rgba(61,110,255,0.15); color: #3d6eff; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
.badge-D { background: rgba(255,214,0,0.15);  color: #ffd600; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
.badge-A { background: rgba(255,64,129,0.15); color: #ff4081; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }

/* ── Form badges ── */
.form-row { display: flex; gap: 6px; align-items: flex-start; flex-wrap: wrap; }
.form-item { display: flex; flex-direction: column; align-items: center; gap: 3px; }
.form-badge { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 800; }
.form-W { background: #00e676; color: #003; }
.form-D { background: #ffd600; color: #332200; }
.form-L { background: #ff4081; color: #fff; }
.form-score { font-size: 0.6rem; color: #556; font-weight: 600; }
.form-opp   { font-size: 0.55rem; color: #445; max-width: 36px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Form card ── */
.form-card { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 1.4rem 1.6rem; }
.form-team-name { font-size: 1rem; font-weight: 700; color: #ccd; margin-bottom: 0.4rem; }
.form-stats { font-size: 0.72rem; color: #556; margin-bottom: 1rem; }
.form-stats span { color: #8892a4; font-weight: 600; margin-right: 0.8rem; }

/* ── H2H section ── */
.h2h-summary { display: flex; justify-content: center; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.h2h-stat { text-align: center; }
.h2h-num { font-size: 2rem; font-weight: 900; }
.h2h-lbl { font-size: 0.65rem; color: #445; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #7c4dff, #00e5ff) !important; color: #fff !important;
    border: none !important; border-radius: 50px !important; padding: 0.85rem 2.5rem !important;
    font-size: 1rem !important; font-weight: 800 !important; letter-spacing: 3px !important;
    text-transform: uppercase !important; width: 100% !important;
    box-shadow: 0 6px 30px rgba(124,77,255,0.35) !important;
}
div.stButton > button:hover { box-shadow: 0 10px 40px rgba(124,77,255,0.55) !important; transform: translateY(-2px) !important; }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important; font-size: 1rem !important; font-weight: 600 !important;
}

/* ── Fixture card (Weekend tab) ── */
.fixture-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 1.4rem 1.8rem; margin-bottom: 1rem;
}
.fixture-teams {
    display: grid; grid-template-columns: 1fr 3fr 1fr;
    align-items: center; gap: 1rem;
}
.fixture-team-name {
    font-size: 1rem; font-weight: 800; color: #e8eaf0;
}
.fixture-team-home { text-align: right; }
.fixture-team-away { text-align: left; }
.fixture-prob-bar {
    display: flex; border-radius: 8px; overflow: hidden; height: 32px;
}
.bar-home { background: #3d6eff; display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 800; color: #fff; min-width: 24px; }
.bar-draw { background: #ffd600; display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 800; color: #332200; min-width: 24px; }
.bar-away { background: #ff4081; display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 800; color: #fff; min-width: 24px; }
.fixture-footer {
    display: flex; justify-content: center; gap: 2rem;
    margin-top: 0.8rem; font-size: 0.75rem; color: #556; flex-wrap: wrap;
}
.fixture-xg { font-weight: 700; color: #7c4dff; }
.fixture-score { font-weight: 700; color: #aab; }

/* ── Backtest table ── */
.bt-correct   { color: #00e676; font-weight: 700; }
.bt-incorrect { color: #ff4081; }
.metric-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 900; color: #e8eaf0; }
.metric-label { font-size: 0.65rem; color: #556; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.3rem; }
.metric-better { color: #00e676; }
.metric-worse  { color: #ff4081; }

.stSpinner > div { border-top-color: #7c4dff !important; }

/* ── Portfolio: P&L Hero ── */
.pnl-hero {
    border-radius: 24px; padding: 2.5rem 2rem; text-align: center;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.pnl-profit {
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,200,83,0.05));
    border: 2px solid rgba(0,230,118,0.35);
}
.pnl-loss {
    background: linear-gradient(135deg, rgba(255,64,129,0.12), rgba(213,0,0,0.05));
    border: 2px solid rgba(255,64,129,0.35);
}
.pnl-neutral {
    background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.1);
}
.pnl-tag { font-size: 0.6rem; font-weight: 700; letter-spacing: 4px; text-transform: uppercase; color: #445; margin-bottom: 0.8rem; }
.pnl-amount { font-size: clamp(2.8rem, 7vw, 5rem); font-weight: 900; line-height: 1; margin-bottom: 0.4rem; }
.pnl-profit .pnl-amount { color: #00e676; }
.pnl-loss   .pnl-amount { color: #ff4081; }
.pnl-neutral .pnl-amount { color: #e8eaf0; }
.pnl-subtitle { font-size: 0.9rem; color: #778; font-weight: 500; }

/* ── Portfolio: Stat cards ── */
.pstat-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1rem; text-align: center;
}
.pstat-val { font-size: 1.6rem; font-weight: 800; color: #e8eaf0; line-height: 1.1; }
.pstat-lbl { font-size: 0.58rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #445; margin-top: 0.3rem; }

/* ── Portfolio: Pending bet cards ── */
.pend-card {
    background: rgba(255,214,0,0.05); border: 1px solid rgba(255,214,0,0.12);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
}
.pend-match  { font-size: 0.88rem; font-weight: 700; color: #e8eaf0; }
.pend-sel    { font-size: 0.78rem; font-weight: 700; margin-top: 0.2rem; }
.pend-meta   { display: flex; gap: 1rem; font-size: 0.72rem; color: #556; margin-top: 0.35rem; flex-wrap: wrap; }
.pend-date   { font-size: 0.65rem; color: #334; margin-top: 0.25rem; }
.pend-return {
    display: flex; align-items: center; gap: 0.6rem;
    margin-top: 0.45rem; padding: 0.5rem 0.75rem;
    background: rgba(0,230,118,0.06); border: 1px solid rgba(0,230,118,0.15);
    border-radius: 8px;
}
.pend-return-label { font-size: 0.65rem; color: #556; text-transform: uppercase; letter-spacing: 1px; }
.pend-return-val   { font-size: 1.15rem; font-weight: 800; color: #00e676; }
.pend-return-profit { font-size: 0.75rem; color: #69f0ae; font-weight: 600; }
.bet-return-box {
    background: linear-gradient(135deg, rgba(0,230,118,0.08), rgba(124,77,255,0.06));
    border: 1px solid rgba(0,230,118,0.2);
    border-radius: 12px; padding: 0.7rem 1rem; margin: 0.5rem 0;
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
}
.bet-return-total { font-size: 1.4rem; font-weight: 800; color: #00e676; }
.bet-return-detail { font-size: 0.78rem; color: #69f0ae; font-weight: 600; }

/* ── Portfolio: Value scanner cards ── */
.scan-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
}
.scan-match { font-size: 0.95rem; font-weight: 700; color: #e8eaf0; flex: 1; min-width: 180px; }
.scan-probs { display: flex; gap: 1.2rem; align-items: center; }
.scan-prob-item { text-align: center; }
.scan-pct  { font-size: 1.3rem; font-weight: 800; line-height: 1; }
.scan-lbl  { font-size: 0.55rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #445; }
.scan-odds-source { font-size: 0.68rem; color: #778; }

/* ── EV tags ── */
.ev-tag { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.5px; }
.ev-strong  { background: rgba(0,230,118,0.18); color: #00e676; }
.ev-mild    { background: rgba(105,240,174,0.15); color: #69f0ae; }
.ev-neutral { background: rgba(255,255,255,0.06); color: #667; }
.ev-neg     { background: rgba(255,64,129,0.12); color: #ff4081; }

/* ── Last Gameweek result cards ── */
.res-card {
    border-radius: 16px; padding: 1.1rem 1.5rem; margin-bottom: 0.8rem;
    border-left: 4px solid; display: flex; flex-direction: column; gap: 0.5rem;
}
.res-correct { background: rgba(0,230,118,0.04); border-color: rgba(0,230,118,0.4); }
.res-wrong   { background: rgba(255,64,129,0.04); border-color: rgba(255,64,129,0.3); }
.res-teams   { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; }
.res-team    { font-size: 0.95rem; font-weight: 700; color: #e8eaf0; }
.res-score   { font-size: 1.8rem; font-weight: 900; color: #e8eaf0; letter-spacing: -1px; padding: 0 0.3rem; }
.res-badge   { font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-left: auto; }
.res-win  { background: rgba(0,230,118,0.15); color: #00e676; }
.res-loss { background: rgba(255,64,129,0.15); color: #ff4081; }
.res-meta  { font-size: 0.75rem; color: #556; display: flex; gap: 1.2rem; flex-wrap: wrap; }
.res-meta b { color: #8892a4; }
.res-prob-bar { display: flex; border-radius: 6px; overflow: hidden; height: 18px; margin-top: 0.2rem; }

/* ── Summary stat chips ── */
.gw-stat { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
           border-radius: 14px; padding: 0.9rem 1rem; text-align: center; }
.gw-stat-val { font-size: 1.5rem; font-weight: 800; color: #e8eaf0; line-height: 1.1; }
.gw-stat-lbl { font-size: 0.58rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #445; margin-top: 0.25rem; }

.scan-value-alert {
    background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.2);
    border-radius: 10px; padding: 0.65rem 1rem; margin-bottom: 0.6rem;
    font-size: 0.82rem; color: #aab;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_data():
    df = load_data()
    df_features = add_rolling_features(df, window=5)
    return df, df_features


@st.cache_resource(show_spinner=False)
def cached_models(df_hash: str):
    df, df_features = cached_data()
    poisson_r  = compute_poisson_ratings(df)
    dc_r       = compute_dixon_coles_ratings(df)
    dc_draw_r  = compute_draw_dc_ratings(df)
    xgb_m, fc  = train_xgb(df_features)
    draw_xgb_m, draw_fc = train_draw_xgb(df_features)
    elo_dict   = get_current_elo(df)
    return poisson_r, dc_r, dc_draw_r, xgb_m, fc, draw_xgb_m, draw_fc, elo_dict


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fixtures():
    return fetch_upcoming_fixtures(lookahead_days=30)


@st.cache_data(show_spinner=False)
def cached_backtest(df_hash: int, test_weeks: int):
    df, df_features = cached_data()
    bt = backtest_models(df, df_features, test_weeks=test_weeks)
    summary = compute_backtest_summary(bt)
    return bt, summary


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper (applies draw specialist automatically)
# ─────────────────────────────────────────────────────────────────────────────

def full_predict(home, away, dc_r, dc_draw_r, xgb_m, feat_cols,
                 draw_xgb_m, draw_fc, hs, as_):
    """Run full prediction pipeline: DC + XGB + draw specialist blend."""
    dc_pred  = predict_dixon_coles(home, away, dc_r)
    xgb_p    = predict_xgb(xgb_m, feat_cols, hs, as_)
    dc_blend = blend_dc(dc_pred, xgb_p)
    draw_xgb_prob = predict_draw_xgb(draw_xgb_m, draw_fc, hs, as_)
    final = blend_draw_specialist(dc_blend, dc_draw_r, draw_xgb_prob, home, away)
    return dc_pred, dc_blend, final


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8892a4", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def heatmap_fig(matrix, home_team, away_team, most_likely):
    n = 7
    z = matrix[:n, :n] * 100
    text = [[f"{z[i, j]:.1f}%" for j in range(n)] for i in range(n)]
    colorscale = [[0.0, "#0a0e1a"], [0.15, "#1a1040"], [0.4, "#3d1f8a"], [0.7, "#6a35d9"], [1.0, "#b388ff"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(i) for i in range(n)], y=[str(i) for i in range(n)],
        colorscale=colorscale, text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="#e8eaf0"), showscale=False,
        hovertemplate="Score %{y}–%{x}: %{text}<extra></extra>",
    ))
    ml_r, ml_c = most_likely
    if ml_r < n and ml_c < n:
        fig.add_shape(type="rect", x0=ml_c-0.5, x1=ml_c+0.5, y0=ml_r-0.5, y1=ml_r+0.5,
                      line=dict(color="#ffd600", width=2.5))
    fig.update_layout(**DARK, height=380,
        title=dict(text="Score Probability Matrix", font=dict(size=13, color="#8892a4"), x=0.5),
        xaxis=dict(title=f"{away_team} goals", gridcolor="rgba(255,255,255,0.04)", title_font=dict(size=11)),
        yaxis=dict(title=f"{home_team} goals", gridcolor="rgba(255,255,255,0.04)", title_font=dict(size=11)))
    return fig


def goal_dist_fig(home_pmf, away_pmf, home_team, away_team, lam_h, lam_a, home_ci, away_ci):
    goals = list(range(8))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=goals, y=[float(home_pmf[i])*100 for i in goals], name=home_team,
        marker=dict(color="rgba(61,110,255,0.75)", line=dict(color="#3d6eff", width=1)),
        hovertemplate="P(%{x} goals) = %{y:.1f}%<extra>" + home_team + "</extra>"))
    fig.add_trace(go.Bar(x=goals, y=[float(away_pmf[i])*100 for i in goals], name=away_team,
        marker=dict(color="rgba(255,64,129,0.75)", line=dict(color="#ff4081", width=1)),
        hovertemplate="P(%{x} goals) = %{y:.1f}%<extra>" + away_team + "</extra>"))
    for lam, color, name in [(lam_h, "#7c9dff", home_team), (lam_a, "#ff80ab", away_team)]:
        fig.add_vline(x=lam, line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=f"xG {name[:3]}: {lam:.2f}", annotation_position="top",
            annotation_font=dict(size=9, color=color))
    fig.update_layout(**DARK, height=380, barmode="group", bargap=0.15, bargroupgap=0.05,
        title=dict(text="Goal Probability Distribution", font=dict(size=13, color="#8892a4"), x=0.5),
        xaxis=dict(title="Goals Scored", tickvals=goals, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def h2h_fig(h2h_df, home_team, away_team):
    hw = int(((h2h_df["HomeTeam"] == home_team) & (h2h_df["FTR"] == "H")).sum() +
             ((h2h_df["AwayTeam"] == home_team) & (h2h_df["FTR"] == "A")).sum())
    d  = int((h2h_df["FTR"] == "D").sum())
    aw = int(((h2h_df["HomeTeam"] == away_team) & (h2h_df["FTR"] == "H")).sum() +
             ((h2h_df["AwayTeam"] == away_team) & (h2h_df["FTR"] == "A")).sum())
    fig = go.Figure()
    for cat, val, color, border in zip(
        [home_team[:15], "Draw", away_team[:15]],
        [hw, d, aw],
        ["rgba(61,110,255,0.7)", "rgba(255,214,0,0.7)", "rgba(255,64,129,0.7)"],
        ["#3d6eff", "#ffd600", "#ff4081"],
    ):
        fig.add_trace(go.Bar(x=[cat], y=[val], marker=dict(color=color, line=dict(color=border, width=1.5)),
            text=[str(val)], textposition="outside", textfont=dict(size=13, color="#e8eaf0"),
            showlegend=False, hovertemplate=f"{cat}: {val}<extra></extra>"))
    fig.update_layout(**DARK, height=280, barmode="group",
        title=dict(text=f"Head-to-Head (Last {len(h2h_df)} meetings)", font=dict(size=13, color="#8892a4"), x=0.5),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), yaxis=dict(gridcolor="rgba(255,255,255,0.04)", dtick=1))
    return fig


def win_prob_gauge(final, home_team, away_team):
    h = round(final["home_win"] * 100, 1)
    d = round(final["draw"]     * 100, 1)
    a = round(final["away_win"] * 100, 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[h], y=[""], orientation="h", name=home_team, marker_color="#3d6eff",
        text=[f"  {home_team[:12]}  {h}%"], textposition="inside",
        textfont=dict(size=13, color="#fff", family="Inter"), insidetextanchor="start",
        hovertemplate=f"{home_team} Win: {h}%<extra></extra>"))
    fig.add_trace(go.Bar(x=[d], y=[""], orientation="h", name="Draw", marker_color="#ffd600",
        text=[f"  Draw  {d}%"], textposition="inside",
        textfont=dict(size=13, color="#332200", family="Inter"), insidetextanchor="middle",
        hovertemplate=f"Draw: {d}%<extra></extra>"))
    fig.add_trace(go.Bar(x=[a], y=[""], orientation="h", name=away_team, marker_color="#ff4081",
        text=[f"{away_team[:12]}  {a}%  "], textposition="inside",
        textfont=dict(size=13, color="#fff", family="Inter"), insidetextanchor="end",
        hovertemplate=f"{away_team} Win: {a}%<extra></extra>"))
    gauge_layout = {k: v for k, v in DARK.items() if k != "margin"}
    fig.update_layout(**gauge_layout, barmode="stack", height=70,
        xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False),
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────
def form_badges_html(form_data):
    items = ""
    for m in form_data:
        items += f"""
        <div class="form-item">
            <div class="form-badge form-{m['result']}">{m['result']}</div>
            <div class="form-score">{m['gf']}–{m['ga']}</div>
            <div class="form-opp" title="{m['opponent']}">{m['opponent'][:6]}</div>
        </div>"""
    return f'<div class="form-row">{items}</div>'


def top5_table_html(top5, home_team, away_team):
    rows = ""
    for hg, ag, pct in top5:
        if hg > ag:
            badge = '<span class="badge-H">Home Win</span>'
        elif hg == ag:
            badge = '<span class="badge-D">Draw</span>'
        else:
            badge = '<span class="badge-A">Away Win</span>'
        rows += f"<tr><td><b>{hg} – {ag}</b></td><td>{pct}%</td><td>{badge}</td></tr>"
    return f"""<table class="score-table">
        <thead><tr><th>Score</th><th>Probability</th><th>Result</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


def fixture_card_html(home_team, away_team, result, lam_h, lam_a):
    """Render a fixture card with blue/yellow/pink probability bar."""
    h_pct = round(result["home_win"] * 100, 1)
    d_pct = round(result["draw"]     * 100, 1)
    a_pct = round(result["away_win"] * 100, 1)
    return f"""
    <div class="fixture-card">
        <div class="fixture-teams">
            <div class="fixture-team-name fixture-team-home">{tb(home_team)}</div>
            <div class="fixture-prob-bar">
                <div class="bar-home" style="width:{h_pct}%">{h_pct}%</div>
                <div class="bar-draw" style="width:{d_pct}%">{d_pct}%</div>
                <div class="bar-away" style="width:{a_pct}%">{a_pct}%</div>
            </div>
            <div class="fixture-team-name fixture-team-away">{tb(away_team)}</div>
        </div>
        <div class="fixture-footer">
            <span class="fixture-xg">xG: {lam_h:.1f} – {lam_a:.1f}</span>
            <span style="color:#445">·</span>
            <span style="color:#556">Home {h_pct}% · Draw {d_pct}% · Away {a_pct}%</span>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Match Predictor
# ─────────────────────────────────────────────────────────────────────────────
def tab_predict(df, df_features, poisson_r, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict):
    st.markdown('<p class="section-label" style="margin-top:1rem">Select Teams</p>',
                unsafe_allow_html=True)

    col_l, col_vs, col_r = st.columns([5, 1, 5])
    with col_l:
        st.markdown('<div class="section-label">Home Team</div>', unsafe_allow_html=True)
        home_team = st.selectbox("Home Team", teams,
            index=teams.index("Arsenal") if "Arsenal" in teams else 0,
            label_visibility="collapsed", key="pred_home")
    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="section-label">Away Team</div>', unsafe_allow_html=True)
        default_away = "Chelsea" if "Chelsea" in teams else (teams[1] if len(teams) > 1 else teams[0])
        away_team = st.selectbox("Away Team", teams,
            index=teams.index(default_away),
            label_visibility="collapsed", key="pred_away")

    st.markdown("<br>", unsafe_allow_html=True)
    if home_team == away_team:
        st.warning("Please select two different teams.")
        return

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        predict = st.button("⚡  Predict Match", key="predict_btn")

    if not predict:
        return

    with st.spinner(f"Analysing {home_team} vs {away_team}..."):
        # Models
        p_pred   = predict_poisson(home_team, away_team, poisson_r)
        hs       = get_current_stats(df, home_team, elo_dict=elo_dict)
        as_      = get_current_stats(df, away_team, elo_dict=elo_dict)
        xgb_p    = predict_xgb(xgb_m, feat_cols, hs, as_)
        p_blend  = blend(p_pred, xgb_p)
        dc_pred, dc_blend, final = full_predict(
            home_team, away_team, dc_r, dc_draw_r, xgb_m, feat_cols,
            draw_xgb_m, draw_fc, hs, as_,
        )

        home_form = get_team_form(df, home_team, n=5)
        away_form = get_team_form(df, away_team, n=5)
        h2h_df    = get_head_to_head(df, home_team, away_team, n=10)

    # ── Model comparison ──────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="text-align:center">Model Comparison</p>',
                unsafe_allow_html=True)

    cA, cB, cC = st.columns(3)
    for col, title, result, tag_color in [
        (cA, "Poisson + XGB", p_blend,  "#7c4dff"),
        (cB, "DC + XGB", dc_blend, "#00e5ff"),
        (cC, "DC + XGB + Draw Specialist", final, "#00e676"),
    ]:
        with col:
            h_p = round(result["home_win"] * 100, 1)
            d_p = round(result["draw"]     * 100, 1)
            a_p = round(result["away_win"] * 100, 1)
            st.markdown(f"""
            <div class="model-card">
                <div class="model-title" style="color:{tag_color}">{title}</div>
                <div style="display:flex; border-radius:8px; overflow:hidden; height:28px; margin-bottom:0.6rem;">
                    <div style="width:{h_p}%; background:#3d6eff; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:800; color:#fff; min-width:20px;">{h_p}%</div>
                    <div style="width:{d_p}%; background:#ffd600; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:800; color:#332200; min-width:20px;">{d_p}%</div>
                    <div style="width:{a_p}%; background:#ff4081; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:800; color:#fff; min-width:20px;">{a_p}%</div>
                </div>
                <div style="font-size:0.72rem; color:#556;">
                    <span style="color:#3d6eff; font-weight:700;">{tb(home_team, 16)}</span> {h_p}% &nbsp;·&nbsp;
                    <span style="color:#ffd600; font-weight:700;">Draw</span> {d_p}% &nbsp;·&nbsp;
                    <span style="color:#ff4081; font-weight:700;">{tb(away_team, 16)}</span> {a_p}%
                </div>
            </div>""", unsafe_allow_html=True)

    # Use DC+XGB+Draw Specialist as the primary for the detailed breakdown below
    # (final already set by full_predict above)
    dc_lam_h = dc_pred["lambda_home"]
    dc_lam_a = dc_pred["lambda_away"]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Stacked probability bar (DC blend) ───────────────────────────────
    st.markdown('<p class="section-label" style="text-align:center">Dixon-Coles · Overall Probability</p>',
                unsafe_allow_html=True)
    st.plotly_chart(win_prob_gauge(final, home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, card_cls, label, pct, pct_cls, sub in [
        (c1, "prob-card-home", home_team, round(final["home_win"]*100,1), "prob-pct-home", "Home Win"),
        (c2, "prob-card-draw", "Draw",   round(final["draw"]*100,1),     "prob-pct-draw", "Draw"),
        (c3, "prob-card-away", away_team, round(final["away_win"]*100,1), "prob-pct-away", "Away Win"),
    ]:
        with col:
            st.markdown(f"""<div class="prob-card {card_cls}">
                <div class="prob-team">{label}</div>
                <div class="prob-pct {pct_cls}">{pct}%</div>
                <div class="prob-label">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Scoreline hero (Dixon-Coles) ──────────────────────────────────────
    ml_h, ml_a = dc_pred["most_likely"]
    ml_prob    = round(dc_pred["score_matrix"][ml_h, ml_a] * 100, 1)
    h_lo, h_hi = dc_pred["home_ci"]
    a_lo, a_hi = dc_pred["away_ci"]

    _, score_col, _ = st.columns([1, 2, 1])
    with score_col:
        st.markdown(f"""<div class="score-hero">
            <div class="score-hero-label">Most Likely Scoreline · Dixon-Coles</div>
            <div class="score-digits">{ml_h}<span class="score-dash"> — </span>{ml_a}</div>
            <div class="score-prob">{ml_prob}% probability</div>
            <div class="score-ci">
                <div class="ci-item">{home_team[:14]}<br>
                    xG <span class="ci-range">{dc_lam_h:.2f}</span> &nbsp;·&nbsp;
                    80% CI <span class="ci-range">{h_lo}–{h_hi}</span> goals</div>
                <div class="ci-item">{away_team[:14]}<br>
                    xG <span class="ci-range">{dc_lam_a:.2f}</span> &nbsp;·&nbsp;
                    80% CI <span class="ci-range">{a_lo}–{a_hi}</span> goals</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Additional markets ────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="text-align:center">Additional Markets · Dixon-Coles</p>',
                unsafe_allow_html=True)

    markets = [
        ("Over 1.5 Goals",   dc_pred["over_15"],   "#7c4dff"),
        ("Over 2.5 Goals",   dc_pred["over_25"],   "#00e5ff"),
        ("Over 3.5 Goals",   dc_pred["over_35"],   "#00ff87"),
        ("Both Teams Score", dc_pred["btts"],       "#ffd600"),
        (f"{home_team} Clean Sheet", dc_pred["home_cs"], "#3d6eff"),
        (f"{away_team} Clean Sheet", dc_pred["away_cs"], "#ff4081"),
    ]
    mkt_cols = st.columns(len(markets))
    for col, (label, prob, color) in zip(mkt_cols, markets):
        pct = round(prob * 100, 1)
        with col:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:14px;padding:0.9rem 0.5rem;text-align:center;">
                <div style="font-size:1.5rem;font-weight:900;color:{color}">{pct}%</div>
                <div style="font-size:0.6rem;color:#556;text-transform:uppercase;letter-spacing:2px;
                margin-top:0.3rem;font-weight:700">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Heatmap + goal distribution ──────────────────────────────────────
    st.markdown('<p class="section-label">Detailed Probability Analysis · Dixon-Coles</p>',
                unsafe_allow_html=True)
    map_col, dist_col = st.columns(2)
    with map_col:
        st.plotly_chart(heatmap_fig(dc_pred["score_matrix"], home_team, away_team, dc_pred["most_likely"]),
                        use_container_width=True, config={"displayModeBar": False})
    with dist_col:
        st.plotly_chart(goal_dist_fig(dc_pred["home_pmf"], dc_pred["away_pmf"],
            home_team, away_team, dc_lam_h, dc_lam_a, dc_pred["home_ci"], dc_pred["away_ci"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Top 5 scorelines ─────────────────────────────────────────────────
    st.markdown('<p class="section-label">Top 5 Most Likely Scorelines · Dixon-Coles</p>',
                unsafe_allow_html=True)
    _, t5_col, _ = st.columns([1, 2, 1])
    with t5_col:
        st.markdown(top5_table_html(dc_pred["top5"], home_team, away_team), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Recent form ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Recent Form (Last 5 Matches)</p>', unsafe_allow_html=True)
    fl, fr = st.columns(2)
    for col, team, form, stats in [(fl, home_team, home_form, hs), (fr, away_team, away_form, as_)]:
        with col:
            st.markdown(f"""<div class="form-card">
                <div class="form-team-name">{tb(team, 24)}</div>
                <div class="form-stats">
                    <span>Avg Scored: {stats['avg_gf']:.1f}</span>
                    <span>Avg Conceded: {stats['avg_ga']:.1f}</span>
                    <span>Pts/Game: {stats['avg_pts']:.1f}</span>
                </div>
                {form_badges_html(form)}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Head-to-head ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Head-to-Head Record</p>', unsafe_allow_html=True)
    if len(h2h_df) == 0:
        st.info("No head-to-head data found for this fixture.")
    else:
        st.plotly_chart(h2h_fig(h2h_df, home_team, away_team),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Recent Meetings</p>', unsafe_allow_html=True)
        rows_html = ""
        for _, row in h2h_df.tail(5).iloc[::-1].iterrows():
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            winner = row["HomeTeam"] if row["FTR"] == "H" else (row["AwayTeam"] if row["FTR"] == "A" else "Draw")
            bc = "badge-H" if row["FTR"] == "H" else ("badge-A" if row["FTR"] == "A" else "badge-D")
            label = winner if row["FTR"] != "D" else "Draw"
            rows_html += f"""<tr>
                <td style="color:#556">{row['Date'].strftime('%d %b %Y')}</td>
                <td style="font-weight:700">{tb(row['HomeTeam'], 18)}</td>
                <td style="font-size:1.1rem;font-weight:900;color:#e8eaf0;text-align:center">{hg} – {ag}</td>
                <td style="font-weight:700">{tb(row['AwayTeam'], 18)}</td>
                <td><span class="{bc}">{label}</span></td>
            </tr>"""
        st.markdown(f"""<table class="score-table" style="text-align:left">
            <thead><tr><th>Date</th><th>Home</th>
            <th style="text-align:center">Score</th><th>Away</th><th>Result</th></tr></thead>
            <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — This Weekend
# ─────────────────────────────────────────────────────────────────────────────
def tab_weekend(df, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict):
    st.markdown('<p class="section-label" style="margin-top:1rem">Upcoming Fixtures</p>',
                unsafe_allow_html=True)

    with st.spinner("Fetching fixtures from ESPN..."):
        fixtures = cached_fixtures()

    if not fixtures:
        st.warning("Could not fetch fixtures automatically. Check back later or add them manually below.")
        fixtures = []

    # ── Date header ───────────────────────────────────────────────────────
    if fixtures:
        dates = sorted({f["date"] for f in fixtures})
        date_range_str = (
            dates[0].strftime("%-d %b")
            if len(dates) == 1
            else f"{dates[0].strftime('%-d %b')} – {dates[-1].strftime('%-d %b %Y')}"
        )
        st.markdown(
            f'<p style="font-size:0.82rem;color:#556;margin-bottom:0.3rem;">'
            f'Next gameweek · <b style="color:#8892a4">{date_range_str}</b>'
            f' · {len(fixtures)} matches</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.72rem;color:#445;margin-bottom:1.4rem;">'
            '<span style="color:#3d6eff;font-weight:700;">■</span> Home Win &nbsp;'
            '<span style="color:#ffd600;font-weight:700;">■</span> Draw &nbsp;'
            '<span style="color:#ff4081;font-weight:700;">■</span> Away Win</p>',
            unsafe_allow_html=True,
        )

        # Group cards by date
        current_date = None
        for fix in fixtures:
            home, away = fix["home"], fix["away"]

            # Skip if team not in our model (e.g. cup games slipping through)
            if home not in teams or away not in teams:
                continue

            # Date subheader
            if fix["date"] != current_date:
                current_date = fix["date"]
                st.markdown(
                    f'<p style="font-size:0.68rem;font-weight:700;color:#556;'
                    f'text-transform:uppercase;letter-spacing:2px;margin:1.2rem 0 0.6rem;">'
                    f'{current_date.strftime("%A %-d %B")}</p>',
                    unsafe_allow_html=True,
                )

            hs      = get_current_stats(df, home, elo_dict=elo_dict)
            as_     = get_current_stats(df, away, elo_dict=elo_dict)
            dc_pred, _, result = full_predict(
                home, away, dc_r, dc_draw_r, xgb_m, feat_cols,
                draw_xgb_m, draw_fc, hs, as_,
            )

            st.markdown(
                fixture_card_html(home, away, result, dc_pred["lambda_home"], dc_pred["lambda_away"]),
                unsafe_allow_html=True,
            )

    # ── Manual add (for fixtures not yet in ESPN or future rounds) ────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("➕ Add a fixture manually", expanded=False):
        fa, fb, fc, fd = st.columns([4, 1, 4, 2])
        with fa:
            wk_home = st.selectbox("Home team", teams,
                index=teams.index("Arsenal") if "Arsenal" in teams else 0,
                key="wk_home", label_visibility="collapsed")
        with fb:
            st.markdown('<div style="text-align:center;padding-top:0.5rem;font-weight:800;color:#556;font-size:1.2rem">vs</div>',
                        unsafe_allow_html=True)
        with fc:
            default_away_wk = "Chelsea" if "Chelsea" in teams else (teams[1] if len(teams) > 1 else teams[0])
            wk_away = st.selectbox("Away team", teams,
                index=teams.index(default_away_wk),
                key="wk_away", label_visibility="collapsed")
        with fd:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Predict", key="add_manual"):
                if wk_home == wk_away:
                    st.warning("Select two different teams.")
                else:
                    hs      = get_current_stats(df, wk_home, elo_dict=elo_dict)
                    as_     = get_current_stats(df, wk_away, elo_dict=elo_dict)
                    dc_pred, _, result = full_predict(
                        wk_home, wk_away, dc_r, dc_draw_r, xgb_m, feat_cols,
                        draw_xgb_m, draw_fc, hs, as_,
                    )
                    st.markdown(
                        fixture_card_html(wk_home, wk_away, result,
                                          dc_pred["lambda_home"], dc_pred["lambda_away"]),
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Backtesting
# ─────────────────────────────────────────────────────────────────────────────
def tab_backtest(df_hash: int):
    st.markdown('<p class="section-label" style="margin-top:1rem">Model Backtesting</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:#556;margin-bottom:1.5rem;">'
        'Train each model on historical data, test on recent matches. '
        'Compares Poisson+XGBoost (baseline) vs Dixon-Coles+XGBoost (enhanced).</p>',
        unsafe_allow_html=True,
    )

    test_weeks = st.slider("Test window (weeks)", min_value=4, max_value=20, value=10, step=2,
                           key="bt_weeks")

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button("⚡  Run Backtest", key="run_bt")

    if not run:
        return

    with st.spinner("Training models and running backtest..."):
        bt, summary = cached_backtest(df_hash, test_weeks)

    if bt.empty:
        st.error("Not enough data for backtesting. Try a smaller test window.")
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Performance Summary</p>', unsafe_allow_html=True)

    n  = summary["n_matches"]
    pa, da, ta = summary["pois_accuracy"], summary["dc_accuracy"], summary["table_accuracy"]
    pb, db     = summary["pois_brier"],    summary["dc_brier"]
    rb         = summary["rand_brier"]

    cols = st.columns(6)
    metrics = [
        (cols[0], f"{n}",    "Matches Tested",         None),
        (cols[1], "33.3%",   "Random Baseline",         None),
        (cols[2], f"{ta}%",  "Table Position Baseline", ta > 33.3),
        (cols[3], f"{pa}%",  "Poisson + XGB",           pa > ta),
        (cols[4], f"{da}%",  "DC + XGB",                da > ta),
        (cols[5], f"{rb}",   "Random Brier ↓",          None),
    ]
    for col, val, label, good in metrics:
        cls = "metric-better" if good is True else ("metric-worse" if good is False else "")
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.78rem;color:#556;text-align:center;">'
        'Table baseline = always predict the higher-placed team wins (home wins ties). '
        'Green = beats the table baseline. '
        'Brier: lower is better (0 = perfect, 0.33 = random).</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Per-match results table ────────────────────────────────────────────
    st.markdown('<p class="section-label">Match-by-Match Results</p>', unsafe_allow_html=True)

    rows_html = ""
    for _, row in bt.sort_values("Date", ascending=False).iterrows():
        t_cls  = "bt-correct" if row["Table_Correct"] else "bt-incorrect"
        p_cls  = "bt-correct" if row["Pois_Correct"]  else "bt-incorrect"
        dc_cls = "bt-correct" if row["DC_Correct"]    else "bt-incorrect"
        t_icon  = "✓" if row["Table_Correct"] else "✗"
        p_icon  = "✓" if row["Pois_Correct"]  else "✗"
        dc_icon = "✓" if row["DC_Correct"]    else "✗"
        pos_str = f"#{row['HomePos']} vs #{row['AwayPos']}"
        rows_html += f"""<tr>
            <td style="color:#556;white-space:nowrap">{row['Date'].strftime('%d %b')}</td>
            <td style="font-weight:700">{row['Home']}</td>
            <td style="text-align:center;font-weight:900;color:#e8eaf0">{row['Score']}</td>
            <td style="font-weight:700">{row['Away']}</td>
            <td style="color:#aab">{row['Actual']}</td>
            <td>
                <span style="color:#445;font-size:0.68rem">{pos_str}</span><br>
                <span class="{t_cls}">{t_icon} {row['Table_Pred']}</span>
            </td>
            <td>
                <span style="color:#556;font-size:0.7rem">{row['Pois_H']}·{row['Pois_D']}·{row['Pois_A']}</span><br>
                <span class="{p_cls}">{p_icon} {row['Pois_Pred']}</span>
            </td>
            <td>
                <span style="color:#556;font-size:0.7rem">{row['DC_H']}·{row['DC_D']}·{row['DC_A']}</span><br>
                <span class="{dc_cls}">{dc_icon} {row['DC_Pred']}</span>
            </td>
        </tr>"""

    st.markdown(f"""<div style="overflow-x:auto">
    <table class="score-table" style="text-align:left;min-width:800px">
        <thead><tr>
            <th>Date</th><th>Home</th>
            <th style="text-align:center">Score</th>
            <th>Away</th><th>Actual</th>
            <th>Table Baseline</th>
            <th>Poisson+XGB</th><th>DC+XGB</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Reliability diagram ────────────────────────────────────────────────
    st.markdown('<p class="section-label">📐  Model Calibration — Reliability Diagram</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:#556;margin-bottom:1.2rem">'
        'A well-calibrated model sits on the diagonal. '
        'Points <b style="color:#00e676">above the line</b> = model under-estimates that outcome. '
        'Points <b style="color:#ff4081">below</b> = model over-estimates it. '
        'Marker size = number of samples in that probability bin.</p>',
        unsafe_allow_html=True,
    )

    # Collect (predicted, actual) pairs for DC+XGB and Poisson+XGB
    cal_pairs = []
    for _, row in bt.iterrows():
        for prefix, col_h, col_d, col_a in [
            ("DC+XGB",      "_dc_h", "_dc_d", "_dc_a"),
            ("Poisson+XGB", "_p_h",  "_p_d",  "_p_a"),
        ]:
            cal_pairs += [
                (prefix, "Home Win",  row[col_h], row["_act_h"]),
                (prefix, "Draw",      row[col_d], row["_act_d"]),
                (prefix, "Away Win",  row[col_a], row["_act_a"]),
            ]
    cal_df = pd.DataFrame(cal_pairs, columns=["Model", "Market", "pred", "actual"])
    cal_df["bin"] = pd.cut(cal_df["pred"], bins=np.linspace(0, 1, 11), labels=False,
                           include_lowest=True)

    fig_cal = go.Figure()
    # Perfect calibration reference
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.18)", dash="dot", width=1.5),
        name="Perfect calibration",
        showlegend=True,
    ))

    style_map = {
        ("DC+XGB",      "Home Win"):  ("#3d6eff", "circle",  "solid"),
        ("DC+XGB",      "Draw"):      ("#ffd600", "diamond", "solid"),
        ("DC+XGB",      "Away Win"):  ("#ff4081", "square",  "solid"),
        ("Poisson+XGB", "Home Win"):  ("#3d6eff", "circle",  "dot"),
        ("Poisson+XGB", "Draw"):      ("#ffd600", "diamond", "dot"),
        ("Poisson+XGB", "Away Win"):  ("#ff4081", "square",  "dot"),
    }
    for (model, market), (color, symbol, dash) in style_map.items():
        sub = cal_df[(cal_df["Model"] == model) & (cal_df["Market"] == market)]
        binned = (
            sub.groupby("bin", observed=True)
            .agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"),
                 count=("actual", "count"))
            .reset_index()
            .dropna()
        )
        if binned.empty:
            continue
        fig_cal.add_trace(go.Scatter(
            x=binned["mean_pred"].tolist(),
            y=binned["mean_actual"].tolist(),
            mode="lines+markers",
            name=f"{model} — {market}",
            line=dict(color=color, width=1.8, dash=dash),
            marker=dict(
                symbol=symbol,
                size=binned["count"].apply(lambda n: max(7, min(18, int(n / 3)))).tolist(),
                color=color,
                opacity=0.85,
            ),
            hovertemplate=(
                f"<b>{model} · {market}</b><br>"
                "Predicted: %{x:.1%}<br>"
                "Actual:    %{y:.1%}<br>"
                "<extra></extra>"
            ),
        ))

    fig_cal.update_layout(
        **DARK,
        height=400,
        xaxis=dict(title="Predicted Probability", tickformat=".0%",
                   gridcolor="rgba(255,255,255,0.04)", range=[-0.02, 1.02]),
        yaxis=dict(title="Actual Frequency",       tickformat=".0%",
                   gridcolor="rgba(255,255,255,0.04)", range=[-0.02, 1.02]),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
    )
    st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<p style="font-size:0.72rem;color:#445;text-align:center">'
        'Solid lines = DC+XGB · Dashed = Poisson+XGB · '
        '🔵 Home Win &nbsp;🟡 Draw &nbsp;🔴 Away Win</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Season Outlook (Monte Carlo simulator)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 12, show_spinner=False)
def cached_season_sim(dc_hash: str, n_sims: int = 10_000):
    df, _ = cached_data()
    table_df  = get_current_table(df)
    remaining = fetch_remaining_season_fixtures()
    result    = simulate_season(remaining, table_df, st.session_state["_dc_r_"], n_sims=n_sims)
    return table_df, result, len(remaining)


def tab_season(df: pd.DataFrame, dc_r: dict):
    st.markdown('<p class="section-label" style="margin-top:1rem">Season Outlook · Monte Carlo Simulation</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:#556;margin-bottom:1.5rem;">'
        'Simulates the remaining season 10,000 times using Dixon-Coles match probabilities '
        'to estimate each team\'s final position distribution.</p>',
        unsafe_allow_html=True,
    )

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button("⚡  Run Season Simulation", key="run_sim")

    if not run:
        return

    # Store dc_r in session state so the cached function can access it
    st.session_state["_dc_r_"] = dc_r

    with st.spinner("Fetching remaining fixtures and simulating 10,000 seasons…"):
        try:
            table_df, sim_df, n_remaining = cached_season_sim(
                dc_hash=str(dc_r.get("home_adv", 0))[:8]
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

    if sim_df.empty:
        st.warning("No remaining fixtures found — season may already be complete.")
        return

    st.markdown(
        f'<p style="font-size:0.78rem;color:#556;text-align:center;margin-bottom:1rem;">'
        f'{n_remaining} remaining fixtures · 10,000 simulations</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Projected Final Table</p>', unsafe_allow_html=True)

    # Current table for live points
    cur_pts = table_df.set_index("Team")["Pts"].to_dict()
    cur_played = table_df.set_index("Team")["Played"].to_dict()

    rows_html = ""
    for pos, row in sim_df.iterrows():
        team = row["Team"]
        pts  = cur_pts.get(team, 0)
        played = cur_played.get(team, 0)
        mean_pts = round(row["mean_pts"], 1)

        # Colour-code probability pills
        def pill(prob, color, label):
            if prob < 0.005:
                return ""
            return (f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
                    f'border-radius:20px;padding:2px 8px;font-size:0.65rem;font-weight:700;'
                    f'margin-right:3px">{label} {prob*100:.0f}%</span>')

        pills = ""
        if row["p_title"]    > 0.005: pills += pill(row["p_title"],    "#ffd600", "Title")
        if row["p_top4"]     > 0.02:  pills += pill(row["p_top4"],     "#00e5ff", "Top 4")
        if row["p_top6"]     > 0.02:  pills += pill(row["p_top6"],     "#7c4dff", "Top 6")
        if row["p_relegated"]> 0.02:  pills += pill(row["p_relegated"],"#ff4081", "Rel")

        rows_html += f"""<tr>
            <td style="color:#445;font-weight:700;width:2rem">{pos+1}</td>
            <td style="font-weight:700;color:#e8eaf0">{tb(team, 18)}</td>
            <td style="color:#556">{played}</td>
            <td style="font-weight:700;color:#7c4dff">{pts}</td>
            <td style="color:#8892a4">{mean_pts}</td>
            <td>{pills}</td>
        </tr>"""

    st.markdown(f"""<div style="overflow-x:auto">
    <table class="score-table" style="text-align:left;min-width:600px">
        <thead><tr>
            <th>#</th><th>Team</th><th>Played</th>
            <th style="color:#7c4dff">Pts Now</th>
            <th>Proj Pts</th><th>Scenarios</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Position probability heatmap ──────────────────────────────────────
    st.markdown('<p class="section-label">Position Probability Heatmap</p>', unsafe_allow_html=True)

    n_teams = len(sim_df)
    z = np.array([
        [sim_df.iloc[i][f"pos_{p+1}"] * 100 for p in range(n_teams)]
        for i in range(n_teams)
    ])
    team_labels = sim_df["Team"].tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(p+1) for p in range(n_teams)],
        y=team_labels,
        colorscale=[[0.0, "#0a0e1a"], [0.2, "#1a1040"], [0.5, "#3d1f8a"],
                    [0.75, "#6a35d9"], [1.0, "#ffd600"]],
        text=[[f"{v:.0f}%" if v >= 1 else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=9, color="#e8eaf0"),
        showscale=False,
        hovertemplate="<b>%{y}</b><br>Finish #%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **DARK,
        height=520,
        title=dict(text="Probability (%) of finishing in each position",
                   font=dict(size=12, color="#8892a4"), x=0.5),
        xaxis=dict(title="Final Position", side="top",
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Last Gameweek Results
# ─────────────────────────────────────────────────────────────────────────────

GW_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "data", "gameweek_history.json")


def _load_gw_history() -> list:
    if os.path.exists(GW_HISTORY_PATH):
        with open(GW_HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_gw_history(history: list):
    os.makedirs(os.path.dirname(GW_HISTORY_PATH), exist_ok=True)
    with open(GW_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def _gw_date_label(matches: list) -> str:
    dates = sorted(set(m["date"] for m in matches))
    d0 = pd.to_datetime(dates[0])
    d1 = pd.to_datetime(dates[-1])
    if d0.date() == d1.date():
        return d0.strftime("%-d %b %Y")
    return f"{d0.strftime('%-d %b')} – {d1.strftime('%-d %b %Y')}"


def _render_gw_records(records: list, date_label: str):
    """Render summary stats + match cards for a list of gameweek records."""
    n           = len(records)
    n_correct   = sum(r["correct"] for r in records)
    n_ou        = sum(r["ou_correct"] for r in records)
    total_goals = sum(r["total"] for r in records)
    exp_goals   = sum(r["exp_total"] for r in records)
    goals_err   = np.mean([abs(r["total"] - r["exp_total"]) for r in records])

    acc_col = "#00e676" if n_correct / n >= 0.5 else ("#ffd600" if n_correct / n >= 0.33 else "#ff4081")
    ou_col  = "#00e676" if n_ou / n >= 0.5 else "#ff4081"

    def _gw_stat(val, lbl, color="#e8eaf0"):
        return (f'<div class="gw-stat">'
                f'<div class="gw-stat-val" style="color:{color}">{val}</div>'
                f'<div class="gw-stat-lbl">{lbl}</div></div>')

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1: st.markdown(_gw_stat(f"{n_correct}/{n}", "CORRECT H/D/A", acc_col), unsafe_allow_html=True)
    with s2: st.markdown(_gw_stat(f"{n_correct/n*100:.0f}%", "ACCURACY"), unsafe_allow_html=True)
    with s3: st.markdown(_gw_stat(f"{n_ou}/{n}", "O/U 2.5 CORRECT", ou_col), unsafe_allow_html=True)
    with s4: st.markdown(_gw_stat(f"{total_goals}", "ACTUAL GOALS"), unsafe_allow_html=True)
    with s5: st.markdown(_gw_stat(f"{exp_goals:.1f}", "EXPECTED GOALS"), unsafe_allow_html=True)
    with s6: st.markdown(_gw_stat(f"±{goals_err:.1f}", "AVG GOALS OFF"), unsafe_allow_html=True)

    best  = max(records, key=lambda r: r["prob_actual"])
    upset = min(records, key=lambda r: r["prob_actual"])
    pred_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    hi1, hi2 = st.columns(2)
    with hi1:
        st.markdown(
            f'<div style="background:rgba(0,230,118,0.07);border:1px solid rgba(0,230,118,0.2);'
            f'border-radius:12px;padding:0.8rem 1.1rem;margin:1rem 0 0.3rem;font-size:0.82rem;color:#aab">'
            f'🎯 <b style="color:#00e676">Best call:</b> {tb(best["home"], 18)} vs {tb(best["away"], 18)} — '
            f'predicted {pred_labels[best["pred_ftr"]]} with '
            f'<b style="color:#e8eaf0">{best["prob_actual"]*100:.0f}% confidence</b> · '
            f'Actual: {best["hg"]}–{best["ag"]}</div>',
            unsafe_allow_html=True,
        )
    with hi2:
        st.markdown(
            f'<div style="background:rgba(255,64,129,0.07);border:1px solid rgba(255,64,129,0.2);'
            f'border-radius:12px;padding:0.8rem 1.1rem;margin:1rem 0 0.3rem;font-size:0.82rem;color:#aab">'
            f'😱 <b style="color:#ff4081">Biggest surprise:</b> {tb(upset["home"], 18)} vs {tb(upset["away"], 18)} — '
            f'only gave the actual result <b style="color:#e8eaf0">{upset["prob_actual"]*100:.0f}% probability</b> · '
            f'Actual: {upset["hg"]}–{upset["ag"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    mkt_colors  = {"H": "#3d6eff", "D": "#ffd600", "A": "#ff4081"}
    current_date = None
    for r in records:
        r_date = r.get("date", "")
        if r_date != current_date:
            current_date = r_date
            st.markdown(
                f'<p style="font-size:0.68rem;font-weight:700;color:#556;'
                f'text-transform:uppercase;letter-spacing:2px;margin:0.8rem 0 0.3rem">'
                f'{pd.to_datetime(r_date).strftime("%A %-d %B")}</p>',
                unsafe_allow_html=True,
            )

        card_cls  = "res-correct" if r["correct"] else "res-wrong"
        badge_cls = "res-win"     if r["correct"] else "res-loss"
        badge_txt = "✓ Correct"   if r["correct"] else "✗ Wrong"
        pred_col  = mkt_colors[r["pred_ftr"]]
        act_label = pred_labels[r["actual_ftr"]]

        bw_h = int(r["p_h"] * 100)
        bw_d = int(r["p_d"] * 100)
        bw_a = 100 - bw_h - bw_d

        xg_txt = ""
        if r.get("xg_h") is not None and r.get("xg_a") is not None:
            xg_txt = f'<span>xG: <b>{r["xg_h"]:.1f}–{r["xg_a"]:.1f}</b></span>'

        goal_diff = r["total"] - r["exp_total"]
        goal_diff_txt = f'+{goal_diff:.1f}' if goal_diff > 0 else f'{goal_diff:.1f}'

        st.markdown(f"""
        <div class="res-card {card_cls}">
            <div class="res-teams">
                <span class="res-team">{tb(r['home'], 18)}</span>
                <span class="res-score">{r['hg']}–{r['ag']}</span>
                <span class="res-team">{tb(r['away'], 18)}</span>
                <span class="res-badge {badge_cls}">{badge_txt}</span>
            </div>
            <div class="res-prob-bar">
                <div style="width:{bw_h}%;background:#3d6eff;display:flex;align-items:center;
                     justify-content:center;font-size:0.65rem;font-weight:800;color:#fff;min-width:18px">
                     {r['p_h']*100:.0f}%</div>
                <div style="width:{bw_d}%;background:#ffd600;display:flex;align-items:center;
                     justify-content:center;font-size:0.65rem;font-weight:800;color:#332200;min-width:18px">
                     {r['p_d']*100:.0f}%</div>
                <div style="width:{bw_a}%;background:#ff4081;display:flex;align-items:center;
                     justify-content:center;font-size:0.65rem;font-weight:800;color:#fff;min-width:18px">
                     {r['p_a']*100:.0f}%</div>
            </div>
            <div class="res-meta">
                <span>Predicted: <b style="color:{pred_col}">{pred_labels[r['pred_ftr']]}</b></span>
                <span>Actual: <b>{act_label}</b></span>
                <span>DC score: <b>{r['pred_score']}</b></span>
                <span>Exp goals: <b>{r['exp_total']:.1f}</b> · Actual: <b>{r['total']}</b>
                      <span style="color:{'#00e676' if abs(goal_diff)<0.5 else '#ffd600'}">({goal_diff_txt})</span></span>
                {xg_txt}
                <span>O/U 2.5: <b style="color:{'#00e676' if r['ou_correct'] else '#ff4081'}">
                {'✓' if r['ou_correct'] else '✗'}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)


def tab_results(df, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict):
    current_season = df["Season"].max()
    season_df = df[df["Season"] == current_season].copy()

    if season_df.empty:
        st.info("No results available yet for the current season.")
        return

    # Last gameweek = matches within 4 days of the most recent completed date
    latest_date = season_df["Date"].max()
    cutoff_date = latest_date - pd.Timedelta(days=4)
    last_gw = season_df[season_df["Date"] >= cutoff_date].sort_values("Date")

    if last_gw.empty:
        st.info("No recent results found.")
        return

    # ── Run predictions for the latest gameweek ───────────────────────────
    records = []
    for _, row in last_gw.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in teams or away not in teams:
            continue

        hs      = get_current_stats(df, home, elo_dict=elo_dict)
        as_     = get_current_stats(df, away, elo_dict=elo_dict)
        dc_pred, _, blended = full_predict(
            home, away, dc_r, dc_draw_r, xgb_m, feat_cols,
            draw_xgb_m, draw_fc, hs, as_,
        )

        p_h, p_d, p_a = blended["home_win"], blended["draw"], blended["away_win"]
        pred_ftr = "H" if p_h >= p_d and p_h >= p_a else ("A" if p_a >= p_h and p_a >= p_d else "D")
        actual_ftr = row["FTR"]
        correct = pred_ftr == actual_ftr

        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        total  = hg + ag
        lam_h  = dc_pred["lambda_home"]
        lam_a  = dc_pred["lambda_away"]
        exp_total = lam_h + lam_a

        p_o25 = dc_pred.get("over_25", 0.5)
        ou_correct = (total > 2) == (p_o25 > 0.5)

        top1 = dc_pred["top5"][0] if dc_pred.get("top5") else None
        pred_score = f"{top1[0]}-{top1[1]}" if top1 else f"{round(lam_h)}-{round(lam_a)}"

        xg_h = float(row["xg_h"]) if "xg_h" in row.index and pd.notna(row["xg_h"]) else None
        xg_a = float(row["xg_a"]) if "xg_a" in row.index and pd.notna(row["xg_a"]) else None

        records.append({
            "home": home, "away": away,
            "hg": hg, "ag": ag, "total": total,
            "date": row["Date"].strftime("%Y-%m-%d"),
            "actual_ftr": actual_ftr, "pred_ftr": pred_ftr,
            "correct": correct,
            "p_h": round(p_h, 4), "p_d": round(p_d, 4), "p_a": round(p_a, 4),
            "prob_actual": round({"H": p_h, "D": p_d, "A": p_a}[actual_ftr], 4),
            "exp_total": round(exp_total, 2),
            "pred_score": pred_score,
            "ou_correct": bool(ou_correct),
            "xg_h": xg_h, "xg_a": xg_a,
        })

    if not records:
        st.info("No results found for current season teams.")
        return

    # ── Save gameweek to history if new ───────────────────────────────────
    history = _load_gw_history()
    current_label = _gw_date_label(records)
    existing_labels = [gw["label"] for gw in history]
    if current_label not in existing_labels:
        history.insert(0, {"label": current_label, "matches": records})
        _save_gw_history(history)

    # ── Gameweek selector ─────────────────────────────────────────────────
    if len(history) > 1:
        options = [gw["label"] for gw in history]
        options[0] = options[0] + "  (latest)"
        sel = st.selectbox("Gameweek", options, index=0, label_visibility="collapsed")
        sel_idx = next(i for i, gw in enumerate(history) if gw["label"] in sel)
        display_records = history[sel_idx]["matches"]
        display_label   = history[sel_idx]["label"]
    else:
        display_records = records
        display_label   = current_label

    st.markdown(
        f'<p class="section-label" style="margin-top:0.5rem">'
        f'Gameweek · {display_label} · {len(display_records)} matches</p>',
        unsafe_allow_html=True,
    )

    _render_gw_records(display_records, display_label)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Mock Portfolio
# ─────────────────────────────────────────────────────────────────────────────

def _ev_badge_html(ev_val: float, min_ev: float) -> str:
    if ev_val >= 0.10:
        return f'<span class="ev-tag ev-strong">EV +{ev_val*100:.1f}%</span>'
    elif ev_val >= min_ev:
        return f'<span class="ev-tag ev-mild">EV +{ev_val*100:.1f}%</span>'
    elif ev_val >= 0:
        return f'<span class="ev-tag ev-neutral">EV +{ev_val*100:.1f}%</span>'
    else:
        return f'<span class="ev-tag ev-neg">EV {ev_val*100:.1f}%</span>'


def tab_portfolio(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict):
    # ── Auto-settle pending bets ─────────────────────────────────────────
    port = pf.load_portfolio()
    n_settled = pf.auto_settle(port, df)
    if n_settled > 0:
        pf.save_portfolio(port)

    stats    = pf.portfolio_stats(port)
    settings = port["settings"]
    min_ev        = float(settings.get("min_ev", 0.05))
    kelly_frac    = float(settings.get("kelly_fraction", 0.5))
    max_stake_pct = float(settings.get("max_stake_pct", 0.10))
    api_key       = settings.get("odds_api_key", "")

    # ── Settings expander ────────────────────────────────────────────────
    with st.expander("⚙️  Portfolio Settings", expanded=False):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            new_initial = st.number_input(
                "Starting Budget (£)", 100.0, 100_000.0,
                float(port["initial_bankroll"]), 100.0, key="port_budget",
            )
        with sc2:
            new_min_ev = st.slider("Min EV Threshold (%)", 1, 20, int(min_ev * 100), key="port_minev")
        with sc3:
            new_kelly = st.select_slider(
                "Kelly Fraction",
                options=[0.25, 0.5, 0.75, 1.0],
                value=kelly_frac,
                key="port_kelly",
                format_func=lambda x: f"{int(x*100)}%",
            )
        with sc4:
            new_api_key = st.text_input(
                "The Odds API Key (optional)",
                value=api_key, type="password", key="port_apikey",
                help="Free key from the-odds-api.com · 500 req/month · auto-fills bookmaker odds",
            )
        _usage = pf.get_api_usage()
        _usage_pct = _usage["count"] / _usage["cap"] * 100 if _usage["cap"] > 0 else 0
        _usage_col = "#00e676" if _usage_pct < 50 else ("#ffd600" if _usage_pct < 80 else "#ff4081")
        st.markdown(
            f'<div style="font-size:0.72rem;color:#445;margin-bottom:0.8rem">'
            f'💡 Get a free API key at <b>the-odds-api.com</b> to auto-fill live bookmaker odds. '
            f'&nbsp;·&nbsp; '
            f'<span style="color:{_usage_col};font-weight:700">'
            f'📊 {_usage["count"]}/{_usage["cap"]} API calls used this month '
            f'({_usage["remaining"]} remaining)</span> '
            f'&nbsp;·&nbsp; Cache: {pf._CACHE_HOURS}h'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Auto-bet row
        ab1, ab2, ab3 = st.columns([2, 2, 4])
        with ab1:
            new_auto_enabled = st.toggle(
                "🤖 Auto-Bet high EV",
                value=settings.get("auto_bet_enabled", False),
                key="port_auto_enabled",
                help="Automatically place Kelly bets when EV exceeds the threshold below",
            )
        with ab2:
            new_auto_thresh = st.slider(
                "Auto-Bet EV Threshold (%)", 1, 30,
                int(settings.get("auto_bet_threshold", 0.15) * 100),
                key="port_auto_thresh",
                help="Only auto-bet when model EV is this high — keeps it to genuinely strong signals",
            )
        with ab3:
            if new_auto_enabled:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#00e676;padding-top:0.6rem">'
                    f'✅ Auto-bet ON — will place Kelly bets when EV ≥ +{new_auto_thresh}% '
                    f'(requires Odds API key)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:0.75rem;color:#445;padding-top:0.6rem">'
                    'Auto-bet is off — all bets placed manually via the scanner below.</div>',
                    unsafe_allow_html=True,
                )
        # Market filter for auto-bet
        _all_markets = {"D": "Draw", "over25": "Over 2.5", "under25": "Under 2.5",
                        "H": "Home Win", "A": "Away Win"}
        _current_auto_mkts = settings.get("auto_markets", list(pf.PROFITABLE_MARKETS))
        st.markdown(
            '<div style="font-size:0.72rem;color:#445;margin:0.5rem 0 0.3rem">'
            '📊 <b>Auto-bet markets</b> — backtest shows Draw & O/U have edge, H/A lose money long-term</div>',
            unsafe_allow_html=True,
        )
        mk1, mk2, mk3, mk4, mk5 = st.columns(5)
        _new_auto_mkts = []
        for col, (code, label) in zip([mk1, mk2, mk3, mk4, mk5], _all_markets.items()):
            with col:
                checked = st.checkbox(
                    label, value=code in _current_auto_mkts, key=f"mkt_toggle_{code}",
                    help="Profitable" if code in pf.PROFITABLE_MARKETS else "Loses money in backtest",
                )
                if checked:
                    _new_auto_mkts.append(code)

        save_c, reset_c, _ = st.columns([1, 1, 4])
        with save_c:
            if st.button("💾 Save Settings", key="save_port_settings"):
                port["settings"]["min_ev"]              = new_min_ev / 100
                port["settings"]["kelly_fraction"]      = new_kelly
                port["settings"]["odds_api_key"]        = new_api_key
                port["settings"]["auto_bet_enabled"]    = new_auto_enabled
                port["settings"]["auto_bet_threshold"]  = new_auto_thresh / 100
                port["settings"]["auto_markets"]        = _new_auto_mkts
                if not any(b["status"] in ("won", "lost") for b in port["bets"]):
                    port["initial_bankroll"] = new_initial
                    port["bankroll"]         = new_initial
                pf.save_portfolio(port)
                st.success("Settings saved!")
                st.rerun()
        with reset_c:
            if st.button("🗑️ Reset Portfolio", key="reset_port", type="secondary"):
                st.session_state["_port_confirm_reset"] = True
        if st.session_state.get("_port_confirm_reset"):
            st.warning("⚠️ This will delete ALL bets and reset your bankroll. Are you sure?")
            cy, cn, _ = st.columns([1, 1, 4])
            with cy:
                if st.button("Yes, reset", key="port_yes"):
                    fresh = {
                        "initial_bankroll": new_initial,
                        "bankroll":         new_initial,
                        "bets":             [],
                        "settings": {
                            "min_ev":          new_min_ev / 100,
                            "kelly_fraction":  new_kelly,
                            "max_stake_pct":   0.10,
                            "odds_api_key":    new_api_key,
                        },
                    }
                    pf.save_portfolio(fresh)
                    st.session_state["_port_confirm_reset"] = False
                    st.rerun()
            with cn:
                if st.button("Cancel", key="port_no"):
                    st.session_state["_port_confirm_reset"] = False
                    st.rerun()

    if n_settled > 0:
        st.success(f"✅ Auto-settled {n_settled} bet{'s' if n_settled > 1 else ''}!")

    # ── Auto-bet (fires on every page load if enabled + API key set) ──────
    auto_enabled   = settings.get("auto_bet_enabled", False)
    auto_threshold = float(settings.get("auto_bet_threshold", 0.15))
    if auto_enabled and api_key.strip():
        _auto_odds = pf.fetch_live_odds(api_key)   # uses 4h cache — no extra API calls
        if _auto_odds:
            _auto_fixtures = cached_fixtures()
            _candidates: list[dict] = []
            for _fix in (_auto_fixtures or []):
                _h, _a = _fix["home"], _fix["away"]
                if _h not in teams or _a not in teams:
                    continue
                _api_o = _auto_odds.get((_h, _a), {})
                if len(_api_o) < 3:
                    continue
                _hs   = get_current_stats(df, _h, elo_dict=elo_dict)
                _as   = get_current_stats(df, _a, elo_dict=elo_dict)
                _dc, _, _res = full_predict(
                    _h, _a, dc_r, dc_draw_r, xgb_m, feat_cols,
                    draw_xgb_m, draw_fc, _hs, _as,
                )
                _ds   = _fix["date"].isoformat()
                _p_o25 = _dc.get("over_25", 0.5)
                for _mkt, _prob, _lbl in [
                    ("H",      _res["home_win"], f"Home Win ({_h})"),
                    ("D",      _res["draw"],     "Draw"),
                    ("A",      _res["away_win"], f"Away Win ({_a})"),
                    ("over25", _p_o25,           "Over 2.5 Goals"),
                    ("under25",1 - _p_o25,       "Under 2.5 Goals"),
                ]:
                    _candidates.append({
                        "home": _h, "away": _a, "date": _ds,
                        "market": _mkt, "selection": _lbl,
                        "model_prob": _prob,
                        "odds": _api_o.get(_mkt),
                        "ev": pf.compute_ev(_prob, _api_o[_mkt]) if _api_o.get(_mkt) else -1,
                    })
            _auto_placed = pf.auto_place_value_bets(port, _candidates, auto_threshold)
            if _auto_placed:
                pf.save_portfolio(port)
                st.success(
                    f"🤖 Auto-placed {len(_auto_placed)} bet{'s' if len(_auto_placed) > 1 else ''} "
                    f"with EV ≥ +{int(auto_threshold*100)}%! "
                    + " · ".join(f"{b['selection']} @ {b['odds']}" for b in _auto_placed)
                )
                stats    = pf.portfolio_stats(port)
                bankroll = stats["bankroll"]

            # Auto-accas disabled — backtest shows they consistently lose (-16% to -46% ROI).
            # Singles on Draw + Under 2.5 is where the model has proven edge.

    # ── P&L Hero ─────────────────────────────────────────────────────────
    profit   = stats["profit"]
    bankroll = stats["bankroll"]
    roi      = stats["roi"]
    pending_bets  = [b for b in port["bets"] if b["status"] == "pending"]
    pending_stake = sum(b["stake"] for b in pending_bets)

    if stats["n_settled"] == 0 and not pending_bets:
        hero_class = "pnl-neutral"
    elif profit >= 0:
        hero_class = "pnl-profit"
    else:
        hero_class = "pnl-loss"

    sign   = "+" if profit >= 0 else ""
    arrow  = "▲" if profit >= 0 else "▼"

    st.markdown(f"""
    <div class="pnl-hero {hero_class}">
        <div class="pnl-tag">📊 MOCK PORTFOLIO · PAPER BETS ONLY · NOT REAL MONEY</div>
        <div class="pnl-amount">{sign}£{abs(profit):,.2f}</div>
        <div class="pnl-subtitle">
            {arrow} {sign}{roi:.1f}% ROI &nbsp;·&nbsp;
            £{bankroll:,.2f} bankroll &nbsp;·&nbsp;
            {stats['n_pending']} pending (£{pending_stake:.0f} at risk)
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────
    def _stat(val, lbl, color="#e8eaf0"):
        return (f'<div class="pstat-card">'
                f'<div class="pstat-val" style="color:{color}">{val}</div>'
                f'<div class="pstat-lbl">{lbl}</div></div>')

    roi_col  = "#00e676" if roi >= 0 else "#ff4081"
    ev_col   = "#7c4dff"
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(_stat(f"£{bankroll:,.0f}", "BANKROLL"), unsafe_allow_html=True)
    with c2: st.markdown(_stat(str(stats["n_settled"]), "SETTLED"), unsafe_allow_html=True)
    with c3: st.markdown(_stat(f"{stats['win_rate']:.0f}%", "WIN RATE"), unsafe_allow_html=True)
    with c4: st.markdown(_stat(f"{sign}{roi:.1f}%", "ROI", roi_col), unsafe_allow_html=True)
    with c5: st.markdown(_stat(f"+{stats['avg_ev_pct']:.1f}%" if stats["avg_ev_pct"] >= 0 else f"{stats['avg_ev_pct']:.1f}%", "AVG MODEL EV", ev_col), unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin:1.2rem 0"></div>', unsafe_allow_html=True)

    # ── Chart + Pending bets ──────────────────────────────────────────────
    col_chart, col_pend = st.columns([6, 4])

    with col_chart:
        st.markdown('<p class="section-label">📈  BANKROLL HISTORY</p>', unsafe_allow_html=True)
        history = pf.bankroll_history(port)
        init_br = port["initial_bankroll"]
        if len(history) > 1:
            # Settled bets exist — show full bankroll history
            y_vals  = history["bankroll"].tolist()
            x_vals  = list(range(len(y_vals)))
            labels  = history["match"].tolist()
        else:
            # No settled bets yet — show starting bankroll + pending stakes
            y_vals = [init_br]
            labels = ["Start"]
            # Show each pending bet as a step down (stake leaving the bankroll)
            running = init_br
            for pb in sorted(pending_bets, key=lambda b: b.get("placed_at", "")):
                running -= pb["stake"]
                y_vals.append(round(running, 2))
                if pb.get("type") == "acca":
                    lbl = f"ACCA £{pb['stake']:.0f}"
                else:
                    lbl = f"{pb.get('selection', '?')} £{pb['stake']:.0f}"
                labels.append(lbl)
            # Add a "?" point for potential return range
            if pending_bets:
                # Best case: all pending bets win
                best_return = running + sum(
                    pb["stake"] * pb["odds"] for pb in pending_bets
                )
                y_vals.append(round(best_return, 2))
                labels.append("If all win")
            x_vals = list(range(len(y_vals)))

        line_col  = "#00e676" if y_vals[-1] >= init_br else "#ff4081"
        fill_col  = "rgba(0,230,118,0.07)" if y_vals[-1] >= init_br else "rgba(255,64,129,0.07)"
        fig_br = go.Figure()
        fig_br.add_hline(
            y=init_br, line_color="rgba(255,255,255,0.15)", line_dash="dot",
            annotation_text=f"Start £{init_br:,.0f}",
            annotation_font=dict(color="#556", size=10),
        )

        if len(history) > 1:
            # Solid line for settled history
            fig_br.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                line=dict(color=line_col, width=2.5),
                marker=dict(
                    size=[0] + [7] * (len(y_vals) - 1),
                    color=["#00e676" if v >= init_br else "#ff4081" for v in y_vals],
                ),
                fill="tonexty",
                fillcolor=fill_col,
                hovertemplate="<b>Bet %{x}</b><br>£%{y:,.2f}<extra></extra>",
            ))
        else:
            # Pending-only view — dashed line with labeled dots
            n_pts = len(y_vals)
            _mk_colors = ["#7c4dff"] + ["#ffd600"] * (n_pts - 2) + ["#00e676"] if n_pts > 2 else ["#7c4dff"] * n_pts
            _mk_sizes = [10] + [8] * (n_pts - 2) + [10] if n_pts > 2 else [10] * n_pts
            fig_br.add_trace(go.Scatter(
                x=x_vals[:-1] if n_pts > 2 else x_vals,
                y=y_vals[:-1] if n_pts > 2 else y_vals,
                mode="lines+markers+text",
                line=dict(color="#ffd600", width=2, dash="dot"),
                marker=dict(size=_mk_sizes[:-1] if n_pts > 2 else _mk_sizes,
                            color=_mk_colors[:-1] if n_pts > 2 else _mk_colors),
                text=labels[:-1] if n_pts > 2 else labels,
                textposition="top center",
                textfont=dict(size=9, color="#556"),
                hovertemplate="%{text}<br>£%{y:,.2f}<extra></extra>",
                name="Staked",
            ))
            if n_pts > 2:
                # Dotted green line to "if all win" point
                fig_br.add_trace(go.Scatter(
                    x=[x_vals[-2], x_vals[-1]],
                    y=[y_vals[-2], y_vals[-1]],
                    mode="lines+markers+text",
                    line=dict(color="#00e676", width=2, dash="dot"),
                    marker=dict(size=[0, 12], color=["#00e676", "#00e676"],
                                symbol=["circle", "star"]),
                    text=["", f"£{y_vals[-1]:,.0f}"],
                    textposition="top center",
                    textfont=dict(size=11, color="#00e676"),
                    hovertemplate="If all win<br>£%{y:,.2f}<extra></extra>",
                    name="Potential",
                ))

        fig_br.update_layout(
            **{k: v for k, v in DARK.items() if k != "margin"},
            height=260, showlegend=False,
            margin=dict(t=10, b=10, l=0, r=0),
            xaxis=dict(title=None, showgrid=False, showticklabels=False),
            yaxis=dict(title=None, gridcolor="rgba(255,255,255,0.04)", tickprefix="£"),
        )
        st.plotly_chart(fig_br, use_container_width=True, config={"displayModeBar": False})

    with col_pend:
        st.markdown('<p class="section-label">⏳  PENDING BETS</p>', unsafe_allow_html=True)
        if pending_bets:
            mkt_colors = {"H": "#3d6eff", "D": "#ffd600", "A": "#ff4081",
                          "over25": "#7c4dff", "under25": "#00e5ff"}
            for bet in pending_bets[-8:]:
                ev_pct = bet["ev"] * 100
                _pot_ret = round(bet['stake'] * bet['odds'], 2)
                _pot_profit = round(_pot_ret - bet['stake'], 2)
                if bet.get("type") == "acca":
                    legs_summary = " + ".join(
                        f"{lg.get('selection', '?')}" for lg in bet.get("legs", [])
                    )
                    match_label = " &amp; ".join(
                        f"{tb(lg['home'], 16)} vs {tb(lg['away'], 16)}" for lg in bet.get("legs", [])
                    )
                    mc = "#7c4dff"
                    st.markdown(f"""
                    <div class="pend-card">
                        <div class="pend-match">🎯 ACCA · {match_label}</div>
                        <div class="pend-sel" style="color:{mc}">{legs_summary} @ {bet['odds']}</div>
                        <div class="pend-meta">
                            <span>£{bet['stake']:.2f} stake</span>
                            <span style="color:#7c4dff">Model {bet.get('combined_model_prob', 0)*100:.1f}%</span>
                            <span style="color:#00e676">EV +{ev_pct:.1f}%</span>
                        </div>
                        <div class="pend-return">
                            <span class="pend-return-label">Returns</span>
                            <span class="pend-return-val">£{_pot_ret:,.2f}</span>
                            <span class="pend-return-profit">(+£{_pot_profit:,.2f} profit)</span>
                        </div>
                        <div class="pend-date">ID #{bet['id']}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    mc = mkt_colors.get(bet["market"], "#aab")
                    st.markdown(f"""
                    <div class="pend-card">
                        <div class="pend-match">{tb(bet['home'], 18)} vs {tb(bet['away'], 18)}</div>
                        <div class="pend-sel" style="color:{mc}">{bet['selection']} @ {bet['odds']}</div>
                        <div class="pend-meta">
                            <span>£{bet['stake']:.2f} stake</span>
                            <span style="color:#7c4dff">Model {bet['model_prob']*100:.1f}%</span>
                            <span style="color:#00e676">EV +{ev_pct:.1f}%</span>
                        </div>
                        <div class="pend-return">
                            <span class="pend-return-label">Returns</span>
                            <span class="pend-return-val">£{_pot_ret:,.2f}</span>
                            <span class="pend-return-profit">(+£{_pot_profit:,.2f} profit)</span>
                        </div>
                        <div class="pend-date">{(bet.get('date') or '')[:10]} · ID #{bet['id']}</div>
                    </div>""", unsafe_allow_html=True)
                if st.button("✕ Cancel", key=f"cancel_{bet['id']}", type="secondary"):
                    pf.remove_pending_bet(port, bet["id"])
                    pf.save_portfolio(port)
                    st.rerun()
        else:
            st.markdown(
                '<p style="color:#334;font-size:0.8rem;text-align:center;padding:2rem 0">'
                'No pending bets</p>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Value Scanner ────────────────────────────────────────────────────
    st.markdown(
        f'<p class="section-label">📡  VALUE BET SCANNER &nbsp;·&nbsp; '
        f'min EV +{int(min_ev*100)}% &nbsp;·&nbsp; {int(kelly_frac*100)}% Kelly</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style="background:rgba(124,77,255,0.07);border:1px solid rgba(124,77,255,0.2);
                border-radius:12px;padding:0.85rem 1.2rem;margin-bottom:1.2rem;
                font-size:0.82rem;color:#8892a4">
        💡 <b style="color:#ccd">How value betting works:</b> When our model's probability
        for an outcome exceeds the bookmaker's implied probability, we have
        <b style="color:#00e676">positive Expected Value</b> — the bet is theoretically
        profitable long-term. Kelly criterion sizes the stake proportionally to our edge.
        Enter odds from your bookmaker, or
        add <b style="color:#00e5ff">The Odds API key</b> in Settings to auto-fill live odds.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Fetching upcoming fixtures..."):
        fixtures = cached_fixtures()

    # Fetch live odds if API key set
    live_odds: dict = {}
    if api_key.strip():
        with st.spinner("Fetching live odds from The Odds API..."):
            live_odds = pf.fetch_live_odds(api_key)

    # ── Collect all fixture data first (needed for acca builder too) ─────────
    fixture_data: list[dict] = []
    if fixtures:
        for fix in fixtures:
            home, away = fix["home"], fix["away"]
            if home not in teams or away not in teams:
                continue
            hs      = get_current_stats(df, home, elo_dict=elo_dict)
            as_     = get_current_stats(df, away, elo_dict=elo_dict)
            dc_pred, _, result = full_predict(
                home, away, dc_r, dc_draw_r, xgb_m, feat_cols,
                draw_xgb_m, draw_fc, hs, as_,
            )
            fixture_data.append({
                "home":      home,
                "away":      away,
                "date":      fix["date"],
                "date_str":  fix["date"].isoformat(),
                "p_h":       result["home_win"],
                "p_d":       result["draw"],
                "p_a":       result["away_win"],
                "p_o25":     dc_pred.get("over_25", 0.5),
                "api_odds":  live_odds.get((home, away), {}),
            })

    if not fixture_data:
        st.info("No upcoming fixtures found. Check back closer to the weekend.")
    else:
        current_date = None
        for fd in fixture_data:
            home, away = fd["home"], fd["away"]
            p_h, p_d, p_a = fd["p_h"], fd["p_d"], fd["p_a"]
            p_o25      = fd["p_o25"]
            api_odds   = fd["api_odds"]
            has_api    = "H" in api_odds and "D" in api_odds and "A" in api_odds
            has_ou_api = "over25" in api_odds and "under25" in api_odds
            fix_key    = f"{home}_{away}".replace(" ", "").replace("'", "").replace("-", "")
            date_str   = fd["date_str"]

            if fd["date"] != current_date:
                current_date = fd["date"]
                st.markdown(
                    f'<p style="font-size:0.7rem;font-weight:700;color:#556;'
                    f'text-transform:uppercase;letter-spacing:2px;margin:1.2rem 0 0.3rem">'
                    f'{current_date.strftime("%A %-d %B")}</p>',
                    unsafe_allow_html=True,
                )

            # Best EV highlight
            value_html = ""
            if has_api:
                candidates_ev = [
                    ("H",      pf.compute_ev(p_h,         api_odds["H"]),       api_odds["H"],       p_h,         home),
                    ("D",      pf.compute_ev(p_d,         api_odds["D"]),       api_odds["D"],       p_d,         "Draw"),
                    ("A",      pf.compute_ev(p_a,         api_odds["A"]),       api_odds["A"],       p_a,         away),
                ]
                if has_ou_api:
                    candidates_ev += [
                        ("over25",  pf.compute_ev(p_o25,       api_odds["over25"]),  api_odds["over25"],  p_o25,       "Over 2.5"),
                        ("under25", pf.compute_ev(1-p_o25,     api_odds["under25"]), api_odds["under25"], 1-p_o25,     "Under 2.5"),
                    ]
                best = max(candidates_ev, key=lambda x: x[1])
                if best[1] >= min_ev:
                    _, ev_val, o_val, pr, label = best
                    kelly_rec = pf.kelly_stake_amount(pr, o_val, bankroll, kelly_frac, max_stake_pct)
                    _scan_ret = round(kelly_rec * o_val, 2)
                    value_html = (
                        f'<div class="scan-value-alert">'
                        f'💎 <b>Value:</b> {label} @ {o_val:.2f} &nbsp;'
                        f'<span class="ev-tag ev-strong">EV +{ev_val*100:.1f}%</span>'
                        f' &nbsp; Kelly: <b style="color:#ffd600">£{kelly_rec:.0f}</b>'
                        f' &nbsp; → &nbsp;<b style="color:#00e676;font-size:1rem">Returns £{_scan_ret:,.0f}</b>'
                        f'</div>'
                    )

            # Fixture header card — always show O/U model prediction
            ou_html = (
                f'<div class="scan-prob-item">'
                f'<div class="scan-pct" style="color:#7c4dff">{p_o25*100:.0f}%</div>'
                f'<div class="scan-lbl">OVER 2.5</div></div>'
            )
            st.markdown(f"""
            <div class="scan-card">
                <div class="scan-match">{tb(home)} <span style="color:#334">vs</span> {tb(away)}</div>
                <div class="scan-probs">
                    <div class="scan-prob-item">
                        <div class="scan-pct" style="color:#3d6eff">{p_h*100:.0f}%</div>
                        <div class="scan-lbl">HOME</div>
                    </div>
                    <div class="scan-prob-item">
                        <div class="scan-pct" style="color:#ffd600">{p_d*100:.0f}%</div>
                        <div class="scan-lbl">DRAW</div>
                    </div>
                    <div class="scan-prob-item">
                        <div class="scan-pct" style="color:#ff4081">{p_a*100:.0f}%</div>
                        <div class="scan-lbl">AWAY</div>
                    </div>
                    {ou_html}
                </div>
                <div class="scan-odds-source">
                    {'📡 Live odds' if has_api else '✍️ Enter odds manually'}
                </div>
            </div>
            {value_html}""", unsafe_allow_html=True)

            # Bet slip expander
            with st.expander(f"💰  Bet on {home} vs {away}", expanded=False):
                mkt_options = [
                    f"Home Win ({home})", "Draw", f"Away Win ({away})",
                    "Over 2.5 Goals", "Under 2.5 Goals",
                ]
                # Pre-select the best EV market
                _mkt_evs = [
                    ("H",      p_h,     api_odds.get("H", 2.0)),
                    ("D",      p_d,     api_odds.get("D", 3.0)),
                    ("A",      p_a,     api_odds.get("A", 3.0)),
                    ("over25", p_o25,   api_odds.get("over25", 2.0)),
                    ("under25",1-p_o25, api_odds.get("under25", 2.0)),
                ]
                _best_idx = max(range(len(_mkt_evs)),
                                key=lambda i: pf.compute_ev(_mkt_evs[i][1], _mkt_evs[i][2]))
                bc1, bc2, bc3, bc4 = st.columns([3, 2, 2, 2])
                with bc1:
                    mkt_sel = st.selectbox("Market", mkt_options, index=_best_idx, key=f"mkt_{fix_key}")
                    if "Home" in mkt_sel:
                        mkt_code, prob_sel, sel_label = "H",       p_h,       f"Home Win ({home})"
                    elif "Away" in mkt_sel:
                        mkt_code, prob_sel, sel_label = "A",       p_a,       f"Away Win ({away})"
                    elif "Over" in mkt_sel:
                        mkt_code, prob_sel, sel_label = "over25",  p_o25,     "Over 2.5 Goals"
                    elif "Under" in mkt_sel:
                        mkt_code, prob_sel, sel_label = "under25", 1-p_o25,   "Under 2.5 Goals"
                    else:
                        mkt_code, prob_sel, sel_label = "D",       p_d,       "Draw"
                with bc2:
                    _ou_key = "over25" if "Over" in mkt_sel else "under25"
                    _default_o = float(api_odds.get(mkt_code, api_odds.get(_ou_key, 2.0))) if api_odds else 2.0
                    odds_inp = st.number_input(
                        "Decimal Odds", 1.01, 200.0, _default_o, 0.05,
                        key=f"odds_{fix_key}", format="%.2f",
                    )
                with bc3:
                    ev_val    = pf.compute_ev(prob_sel, odds_inp)
                    kelly_rec = pf.kelly_stake_amount(prob_sel, odds_inp, bankroll,
                                                      kelly_frac, max_stake_pct)
                    stake_inp = st.number_input(
                        "Stake (£)", 1.0, max(float(bankroll), 1.0),
                        max(float(kelly_rec), 1.0), 1.0,
                        key=f"stake_{fix_key}", format="%.2f",
                    )
                with bc4:
                    ev_color = "#00e676" if ev_val >= min_ev else ("#ffd600" if ev_val >= 0 else "#ff4081")
                    st.markdown(f"""
                    <div style="padding-top:0.25rem">
                        <div style="font-size:0.62rem;color:#445;text-transform:uppercase;letter-spacing:1px">Model EV</div>
                        <div style="font-size:1.5rem;font-weight:800;color:{ev_color}">
                            {"+" if ev_val >= 0 else ""}{ev_val*100:.1f}%
                        </div>
                        <div style="font-size:0.68rem;color:#556">Kelly: £{kelly_rec:.0f}</div>
                        <div style="font-size:0.68rem;color:#556">Implied: {pf.implied_prob(odds_inp)*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                if ev_val < 0:
                    st.warning("⚠️ Negative EV — model says these odds aren't value.")
                elif 0 <= ev_val < min_ev:
                    st.info(f"ℹ️ Below your +{int(min_ev*100)}% threshold but still positive EV.")

                already = any(
                    b.get("type") != "acca"
                    and b["status"] == "pending"
                    and b["home"] == home and b["away"] == away
                    and b["market"] == mkt_code
                    for b in port["bets"]
                )
                if already:
                    st.warning("⚠️ Already have a pending bet on this market.")
                elif stake_inp > bankroll:
                    st.error(f"Insufficient bankroll — £{bankroll:.2f} available.")
                else:
                    pot_return = round(stake_inp * odds_inp, 2)
                    pot_profit = round(pot_return - stake_inp, 2)
                    st.markdown(
                        f'<div class="bet-return-box">'
                        f'<div>'
                        f'<div style="font-size:0.6rem;color:#556;text-transform:uppercase;letter-spacing:1px">Potential Return</div>'
                        f'<div class="bet-return-total">£{pot_return:,.2f}</div>'
                        f'</div>'
                        f'<div class="bet-return-detail">+£{pot_profit:,.2f} profit</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"✅  Place £{stake_inp:.0f} on {sel_label}",
                                 key=f"place_{fix_key}_{mkt_code}", type="primary"):
                        pf.place_bet(port, home, away, date_str, mkt_code, sel_label,
                                     prob_sel, odds_inp, stake_inp)
                        pf.save_portfolio(port)
                        st.success(f"🎯 Bet placed! £{stake_inp:.0f} on {sel_label} @ {odds_inp:.2f}")
                        st.rerun()

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Accumulator Builder ───────────────────────────────────────────────
    st.markdown('<p class="section-label">🎲  ACCUMULATOR BUILDER</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(124,77,255,0.07);border:1px solid rgba(124,77,255,0.2);
                border-radius:12px;padding:0.85rem 1.2rem;margin-bottom:1.2rem;
                font-size:0.82rem;color:#8892a4">
        💡 Accumulators multiply the odds of each leg — big returns if all legs win.
        The model finds combinations where <b style="color:#00e676">every individual leg has positive EV</b>,
        maximising the chance the combined bet also beats the bookmaker.
        Requires live odds (Odds API key) for best suggestions.
    </div>""", unsafe_allow_html=True)

    # Build candidates from fixture data
    acca_candidates: list[dict] = []
    for fd in fixture_data:
        api_odds = fd["api_odds"]
        if not ("H" in api_odds and "D" in api_odds and "A" in api_odds):
            continue
        for mkt, prob, sel in [
            ("H",      fd["p_h"],    f"Home Win ({fd['home']})"),
            ("D",      fd["p_d"],    "Draw"),
            ("A",      fd["p_a"],    f"Away Win ({fd['away']})"),
            ("over25", fd["p_o25"],  "Over 2.5"),
        ]:
            o = api_odds.get(mkt)
            if not o or o <= 1:
                continue
            acca_candidates.append({
                "home": fd["home"], "away": fd["away"],
                "date": fd["date_str"], "market": mkt, "selection": sel,
                "model_prob": prob, "odds": o,
                "ev": pf.compute_ev(prob, o),
            })

    suggestions = pf.build_acca_suggestions(acca_candidates, min_single_ev=0.0, max_legs=3) if acca_candidates else []

    if not suggestions:
        if not live_odds:
            st.markdown(
                '<p style="color:#445;font-size:0.82rem">Add an Odds API key in Settings to '
                'generate accumulator suggestions based on live bookmaker odds.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:#445;font-size:0.82rem">No positive-EV combinations found '
                'for upcoming fixtures.</p>',
                unsafe_allow_html=True,
            )
    else:
        doubles  = [s for s in suggestions if s["n_legs"] == 2][:5]
        trebles  = [s for s in suggestions if s["n_legs"] == 3][:5]

        for label, group in [("📗 Top Doubles", doubles), ("📘 Top Trebles", trebles)]:
            if not group:
                continue
            st.markdown(f'<p style="font-size:0.72rem;font-weight:700;color:#556;'
                        f'text-transform:uppercase;letter-spacing:2px;margin:0.8rem 0 0.4rem">'
                        f'{label}</p>', unsafe_allow_html=True)
            for i, sug in enumerate(group):
                ev_val  = sug["ev"]
                ev_col  = "#00e676" if ev_val >= min_ev else ("#ffd600" if ev_val >= 0 else "#ff4081")
                legs_txt = " &nbsp;✕&nbsp; ".join(
                    f'<b style="color:#ccd">{lg["selection"]}</b>'
                    f' <span style="color:#556">({tb(lg["home"], 16)} v {tb(lg["away"], 16)})</span>'
                    f' <span style="color:#aab">@ {lg["odds"]:.2f}</span>'
                    for lg in sug["legs"]
                )
                kelly_acca = pf.kelly_stake_amount(
                    sug["combined_prob"], sug["combined_odds"],
                    bankroll, kelly_frac, max_stake_pct
                )
                st.markdown(f"""
                <div class="scan-card" style="flex-direction:column;align-items:flex-start;gap:0.5rem">
                    <div style="font-size:0.82rem">{legs_txt}</div>
                    <div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap">
                        <span style="color:#aab">Combined @ <b style="color:#e8eaf0">{sug['combined_odds']:.2f}</b></span>
                        <span style="color:#7c4dff">Model: {sug['combined_prob']*100:.1f}%</span>
                        <span class="ev-tag {'ev-strong' if ev_val>=0.1 else ('ev-mild' if ev_val>=0 else 'ev-neg')}">
                            EV {'+'  if ev_val>=0 else ''}{ev_val*100:.1f}%</span>
                        <span style="color:#ffd600">Kelly: £{kelly_acca:.0f}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                acca_key = f"acca_{label.replace(' ','_')}_{i}"
                acca_s1, acca_s2, _ = st.columns([2, 2, 4])
                with acca_s1:
                    acca_stake = st.number_input(
                        "Stake (£)", 1.0, max(float(bankroll), 1.0),
                        max(float(kelly_acca), 1.0), 1.0,
                        key=f"s_{acca_key}", format="%.2f",
                    )
                with acca_s2:
                    pot_ret = round(acca_stake * sug["combined_odds"], 2)
                    pot_prof = round(pot_ret - acca_stake, 2)
                    st.markdown(
                        f'<div class="bet-return-box" style="margin-top:0.2rem">'
                        f'<div>'
                        f'<div style="font-size:0.55rem;color:#556;text-transform:uppercase;letter-spacing:1px">Returns</div>'
                        f'<div class="bet-return-total">£{pot_ret:,.2f}</div>'
                        f'</div>'
                        f'<div class="bet-return-detail">+£{pot_prof:,.2f} profit</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if acca_stake <= bankroll:
                    if st.button(f"✅  Place {sug['n_legs']}-fold Acca",
                                 key=f"b_{acca_key}", type="primary"):
                        pf.place_acca(port, sug["legs"], acca_stake)
                        pf.save_portfolio(port)
                        st.success(f"🎯 {sug['n_legs']}-fold acca placed @ {sug['combined_odds']:.2f}!")
                        st.rerun()

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Bet History ───────────────────────────────────────────────────────
    settled_hist = [b for b in port["bets"] if b["status"] in ("won", "lost")]
    if settled_hist:
        st.markdown('<p class="section-label">📋  BET HISTORY</p>', unsafe_allow_html=True)
        rows_hist = []
        for b in reversed(settled_hist):
            profit_b = b["profit"] or 0.0
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
                sel_label = b["selection"]
                date_label = (b.get("date") or "")[:10]
            rows_hist.append({
                "Date":       date_label,
                "Match":      match_label,
                "Selection":  sel_label,
                "Odds":       b["odds"],
                "Stake":      b["stake"],
                "EV":         f"+{b['ev']*100:.1f}%",
                "Result":     "✅ Won" if b["status"] == "won" else "❌ Lost",
                "P&L (£)":    profit_b,
            })
        hist_df = pd.DataFrame(rows_hist)
        st.dataframe(
            hist_df,
            use_container_width=True,
            hide_index=True,
            height=min(420, 60 + len(rows_hist) * 35),
            column_config={
                "Stake":    st.column_config.NumberColumn("Stake", format="£%.2f"),
                "P&L (£)":  st.column_config.NumberColumn("P&L", format="£%.2f"),
            },
        )

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Historical EV Backtest ────────────────────────────────────────────
    with st.expander("📜  Historical EV Backtest — could you beat the bookies on past data?", expanded=False):
        st.markdown("""
        <div style="font-size:0.82rem;color:#778;margin-bottom:1rem;line-height:1.6">
            Simulates what would have happened if you had placed Kelly-sized bets on every match
            where the <b style="color:#ccd">DC+XGB model</b> identified a value opportunity vs
            actual <b style="color:#ccd">Bet365 closing odds</b> from our historical CSV data.
            Training is strictly cut off before each test window — no data leakage.
            Results directly answer: <i>does our model have long-run edge over the bookmaker?</i>
        </div>""", unsafe_allow_html=True)

        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            hbt_weeks  = st.slider("Test Window (weeks)", 8, 52, 52, key="hbt_weeks")
        with hc2:
            hbt_min_ev = st.slider("Min EV (%)", 1, 20, 3, key="hbt_minev")
        with hc3:
            hbt_kelly  = st.select_slider(
                "Kelly Fraction", [0.25, 0.5, 0.75, 1.0], 1.0, key="hbt_kelly",
                format_func=lambda x: f"{int(x*100)}%",
            )
        with hc4:
            hbt_bankroll = st.number_input(
                "Starting Bankroll (£)", 100.0, 100000.0, 10000.0, 1000.0,
                key="hbt_bankroll", format="%.0f",
            )

        if st.button("🔄  Run Simulation", key="run_hbt", type="primary"):
            with st.spinner("Running EV simulation on historical match data..."):
                bt_df = backtest_models(df, df_features, test_weeks=hbt_weeks)
                _bt_markets = set(settings.get("auto_markets", list(pf.PROFITABLE_MARKETS)))
                log_df, summary = pf.ev_backtest_simulate(
                    bt_df, df,
                    min_ev_pct=float(hbt_min_ev),
                    kelly_frac=float(hbt_kelly),
                    max_stake_pct=0.10,
                    initial_bankroll=float(hbt_bankroll),
                    allowed_markets=_bt_markets,
                )
                st.session_state["_hbt_log"]     = log_df
                st.session_state["_hbt_summary"] = summary

        if "_hbt_summary" in st.session_state:
            summary = st.session_state["_hbt_summary"]
            log_df  = st.session_state["_hbt_log"]

            if "error" in summary:
                st.error(summary["error"])
            else:
                sim_profit = summary["profit"]
                sim_roi    = summary["roi"]
                sim_col    = "#00e676" if sim_profit >= 0 else "#ff4081"

                ms1, ms2, ms3, ms4, ms5 = st.columns(5)
                with ms1: st.metric("Final Bankroll", f"£{summary['final']:,.0f}",
                                    f"{'+'  if sim_profit >= 0 else ''}£{sim_profit:,.0f}")
                with ms2: st.metric("ROI", f"{'+'  if sim_roi >= 0 else ''}{sim_roi:.1f}%")
                with ms3: st.metric("Total Bets", str(summary["n_bets"]))
                with ms4: st.metric("Win Rate", f"{summary['win_rate']:.0f}%")
                with ms5: st.metric("Avg Odds", f"{summary['avg_odds']:.2f}")

                if not log_df.empty:
                    _init_br = summary["initial"]
                    x_h = list(range(len(log_df) + 1))
                    y_h = [_init_br] + log_df["Bankroll"].tolist()
                    fig_hbt = go.Figure()
                    fig_hbt.add_hline(
                        y=_init_br, line_color="rgba(255,255,255,0.15)", line_dash="dot",
                        annotation_text=f"£{_init_br:,.0f} start",
                        annotation_font=dict(color="#556", size=10),
                    )
                    fig_hbt.add_trace(go.Scatter(
                        x=x_h, y=y_h,
                        mode="lines",
                        line=dict(color=sim_col, width=2.5),
                        fill="tozeroy",
                        fillcolor=f"rgba(0,230,118,0.07)" if sim_profit >= 0 else "rgba(255,64,129,0.07)",
                        hovertemplate="Bet %{x}: £%{y:,.2f}<extra></extra>",
                        name="Bankroll",
                    ))
                    fig_hbt.update_layout(
                        **{k: v for k, v in DARK.items() if k != "margin"},
                        height=280, showlegend=False,
                        margin=dict(t=10, b=10, l=0, r=0),
                        xaxis=dict(title="Bet number", gridcolor="rgba(255,255,255,0.04)"),
                        yaxis=dict(title="Bankroll (£)", gridcolor="rgba(255,255,255,0.04)", tickprefix="£"),
                    )
                    st.plotly_chart(fig_hbt, use_container_width=True, config={"displayModeBar": False})

                    disp_cols = ["Date", "Match", "Market", "Model%", "Implied%",
                                 "EV", "Odds", "Stake", "Result", "Profit", "Bankroll"]
                    st.dataframe(
                        log_df[[c for c in disp_cols if c in log_df.columns]],
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, 60 + len(log_df) * 35),
                        column_config={
                            "Bankroll": st.column_config.NumberColumn("Bankroll", format="£%.2f"),
                            "Stake":    st.column_config.NumberColumn("Stake",    format="£%.2f"),
                            "Profit":   st.column_config.NumberColumn("P&L",      format="£%.2f"),
                        },
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading match data..."):
        try:
            df, df_features = cached_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    with st.spinner("Training models (Dixon-Coles MLE + XGBoost)..."):
        cache_key = f"{len(df)}_{df['Date'].max().date()}"
        poisson_r, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, elo_dict = cached_models(cache_key)

    teams     = get_current_teams(df)
    last_date = df["Date"].max().strftime("%d %b %Y")
    n_matches = len(df)
    rho       = dc_r.get("rho", 0.0)
    rho_draw  = dc_draw_r.get("rho", 0.0)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="pl-header">
        <div class="pl-title">⚽ Premier League Predictor</div>
        <div class="pl-subtitle">Dixon-Coles · Poisson · XGBoost · Draw Specialist</div>
        <div class="pl-meta">{n_matches:,} matches loaded · Data through {last_date}
        · DC ρ = {rho:.3f} · Draw ρ = {rho_draw:.3f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🔮  Predict Match",
        "📅  This Weekend",
        "📋  Last Gameweek",
        "📊  Backtesting",
        "🏆  Season Outlook",
        "💰  Mock Portfolio",
    ])

    with t1:
        tab_predict(df, df_features, poisson_r, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict)

    with t2:
        tab_weekend(df, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict)

    with t3:
        tab_results(df, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict)

    with t4:
        tab_backtest(len(df))

    with t5:
        tab_season(df, dc_r)

    with t6:
        tab_portfolio(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;font-size:0.65rem;color:#334;letter-spacing:2px">'
        'DATA: FOOTBALL-DATA.CO.UK + UNDERSTAT · MODEL: DIXON-COLES + POISSON + XGBOOST · FOR ENTERTAINMENT PURPOSES</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
