"""
Premier League Match Predictor — Streamlit App
Poisson · Dixon-Coles · XGBoost ensemble
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio as pf
import season_archive as sa
from data import (
    add_rolling_features,
    fetch_remaining_season_fixtures,
    fetch_upcoming_fixtures,
    get_current_elo,
    get_current_stats,
    get_current_table,
    preseason_table,
    table_is_stale_for,
    get_current_teams,
    get_head_to_head,
    get_team_form,
    load_data,
    team_match_counts,
)
from models import (
    MARKET_ODDS_CAPTURED,
    PROMOTED_PRIOR,
    market_odds_age_days,
    market_odds_are_stale,
    promoted_prior_for,
    seed_promoted_elo,
    promoted_elo_offsets,
    backtest_models,
    backtest_models_v2,
    blend,
    blend_dc,
    blend_draw_specialist,
    compute_backtest_summary,
    compute_dixon_coles_kn_ratings,
    compute_dixon_coles_ratings,
    seed_promoted_teams,
    compute_draw_dc_ratings,
    compute_poisson_ratings,
    predict_dixon_coles,
    predict_dixon_coles_kn,
    predict_draw_xgb,
    predict_poisson,
    predict_xgb,
    simulate_season,
    train_draw_xgb,
    train_xgb,
)

# ─────────────────────────────────────────────────────────────────────────────
# Premier League symbol (inlined as base64 data URI so it travels with the app)
# ─────────────────────────────────────────────────────────────────────────────
def _load_pl_symbol_data_uri() -> str:
    path = Path(__file__).parent / "assets" / "pl_symbol.jpg"
    if not path.exists():
        return ""
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


PL_SYMBOL_URI = _load_pl_symbol_data_uri()


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
    """Return an <img> tag for the team badge, or empty string if not found.

    Tagged with class="team-badge" so the universal halo CSS can apply.
    """
    url = _BADGE_URL.get(team, "")
    if not url:
        return ""
    return (
        f'<img class="team-badge" src="{url}" width="{size}" height="{size}" '
        f'style="vertical-align:middle;margin-right:4px" />'
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

/* ── Animations ────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes growFromLeft {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
@keyframes softPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(124,77,255,0.0); }
    50%      { box-shadow: 0 0 0 6px rgba(124,77,255,0.08); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

/* Fade-in for content cards — gentle, staggered by default */
.fixture-card, .pend-card, .metric-card, .res-card,
.score-table, .scan-card, .kpi-card {
    animation: fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
}

/* Probability bar grows in from the left */
.fixture-prob-bar { transform-origin: left center; }
.bar-home, .bar-draw, .bar-away, .res-prob-bar > div {
    transform-origin: left center;
    animation: growFromLeft 0.7s cubic-bezier(.22,.61,.36,1);
}

/* Team badges: subtle lift on card hover */
.fixture-card img, .res-card img, .pend-card img {
    transition: transform 0.28s cubic-bezier(.22,.61,.36,1),
                filter     0.28s ease;
}
.fixture-card:hover img,
.res-card:hover img,
.pend-card:hover img {
    transform: scale(1.12) rotate(-3deg);
    filter: drop-shadow(0 4px 8px rgba(124,77,255,0.25));
}

/* Cards themselves lift very slightly on hover */
.fixture-card, .pend-card, .res-card {
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.fixture-card:hover, .pend-card:hover, .res-card:hover {
    transform: translateY(-2px);
    border-color: rgba(124,77,255,0.25) !important;
}

/* Buttons: gentle press feedback */
.stButton > button {
    transition: transform 0.12s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(124,77,255,0.12);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Metric cards: soft pulse on mount */
.metric-card { animation: fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both,
                          softPulse 2.4s ease-in-out 0.5s; }

/* ── Hero strip (live stats banner above tabs) ─────────────────────── */
.hero-strip {
    display: grid;
    grid-template-columns: 1fr 1fr 2fr;
    gap: 0.8rem;
    margin: 1rem 0 1.5rem;
    animation: fadeInUp 0.55s cubic-bezier(.22,.61,.36,1) both;
}
.hero-stat {
    background: linear-gradient(135deg, rgba(124,77,255,0.05), rgba(0,229,255,0.03));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    transition: border-color 0.25s ease, transform 0.25s ease;
}
.hero-stat:hover {
    border-color: rgba(124,77,255,0.25);
    transform: translateY(-2px);
}
.hero-stat-label {
    font-size: 0.78rem; color: #b8c0d0; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 700; margin-bottom: 0.35rem;
}
.hero-stat-value {
    font-size: 1.6rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-stat-sub {
    font-size: 0.82rem; color: #8892a4; font-weight: 600;
}

/* ── Insight cards (Why this prediction) — larger, more visual weight ── */
.insight-card {
    background: linear-gradient(135deg, rgba(124,77,255,0.07), rgba(0,229,255,0.02));
    border: 1px solid rgba(124,77,255,0.18);
    border-left: 4px solid #7c4dff;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.85rem;
    display: flex; align-items: center; gap: 1.1rem;
    animation: fadeInUp 0.45s cubic-bezier(.22,.61,.36,1) both;
    transition: transform 0.22s ease, border-left-color 0.22s ease,
                box-shadow 0.22s ease;
}
.insight-card:hover {
    transform: translateX(3px);
    border-left-color: #00e5ff;
    box-shadow: 0 6px 20px rgba(124,77,255,0.20);
}
.insight-emoji {
    font-size: 2rem; flex-shrink: 0; line-height: 1;
    filter: drop-shadow(0 2px 6px rgba(124,77,255,0.4));
}
.insight-body { flex: 1; min-width: 0; }
.insight-head {
    font-size: 1.18rem; color: #e8eaf0; font-weight: 700; line-height: 1.4;
    letter-spacing: -0.1px;
}
.insight-sub {
    font-size: 0.94rem; color: #8892a4; margin-top: 0.4rem;
    line-height: 1.5;
}

/* ── Home landing screen ─────────────────────────────────────────── */
.home-hero {
    display: flex; flex-direction: column; align-items: center;
    padding: 3.5rem 1rem 1.5rem;
    animation: fadeInUp 0.7s cubic-bezier(.22,.61,.36,1) both;
}
.home-hero img {
    width: 180px; height: 180px;
    object-fit: cover; border-radius: 28px;
    box-shadow: 0 10px 40px rgba(124,77,255,0.18),
                0 0 0 1px rgba(255,255,255,0.06);
    transition: transform 0.4s cubic-bezier(.22,.61,.36,1);
}
.home-hero img:hover { transform: scale(1.04) rotate(-2deg); }
.home-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900; letter-spacing: -0.5px;
    text-align: center; margin-top: 1.6rem;
    background: linear-gradient(135deg, #7c4dff 0%, #00e5ff 55%, #00ff87 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.home-sub {
    font-size: 1.1rem; color: #a78bfa; letter-spacing: 2.5px;
    text-transform: uppercase; font-weight: 800;
    margin-top: 0.9rem; text-align: center; opacity: 0.85;
}

/* ── Home screen — Activity Feed ── */
.home-feed-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin: 1.4rem auto 1rem;
    max-width: 1200px;
    animation: fadeInUp 0.55s cubic-bezier(.22,.61,.36,1) both;
}
.hf-tile {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(124,77,255,0.18);
    border-radius: 16px; padding: 1.1rem 1.2rem;
    display: flex; flex-direction: column; gap: 0.4rem;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.hf-tile:hover {
    transform: translateY(-3px);
    border-color: rgba(124,77,255,0.45);
    box-shadow: 0 10px 30px rgba(124,77,255,0.15);
}
.hf-tile-icon { font-size: 1.7rem; line-height: 1; }
.hf-tile-lbl  {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px;
    text-transform: uppercase; color: #a78bfa;
}
.hf-tile-val  {
    font-size: 2.4rem; font-weight: 900; color: #e8eaf0;
    line-height: 1; letter-spacing: -1px; font-variant-numeric: tabular-nums;
}
.hf-tile-sub  { font-size: 0.86rem; color: #8892a4; line-height: 1.45; }

.hf-portfolios-row {
    display: flex; flex-direction: column; gap: 0.25rem;
    font-variant-numeric: tabular-nums;
}
.hf-port-line {
    font-size: 1.1rem; color: #cdd; font-weight: 800;
    display: flex; justify-content: space-between; gap: 0.5rem;
}
.hf-port-line b { color: #8892a4; font-weight: 700; }

/* Next fixture variant — different feel from numeric tiles */
.hf-next .hf-next-when {
    font-size: 1.55rem; font-weight: 900; color: #00e5ff; line-height: 1.05;
    letter-spacing: -0.4px; margin-top: 0.1rem;
}
.hf-next .hf-next-match {
    font-size: 0.94rem; color: #cdd; font-weight: 700; margin-top: 0.3rem;
    line-height: 1.35;
}

/* Recent events feed */
.hf-feed-title {
    font-size: 0.86rem; font-weight: 800; letter-spacing: 1.6px;
    text-transform: uppercase; color: #a78bfa;
    margin: 1.4rem 0 0.5rem; max-width: 1200px;
}
.hf-feed-list {
    display: flex; flex-direction: column; gap: 0.4rem;
    max-width: 1200px; margin: 0 auto 1rem;
}
.hf-evt {
    display: grid; grid-template-columns: 80px 1fr 90px;
    gap: 0.85rem; align-items: center;
    padding: 0.7rem 1rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid;
    border-radius: 10px;
    transition: transform 0.18s ease, border-color 0.18s ease;
}
.hf-evt:hover {
    transform: translateX(2px);
    border-color: rgba(124,77,255,0.4);
}
.hf-evt-place { border-left-color: #00e5ff;
                background: linear-gradient(90deg, rgba(0,229,255,0.06), transparent); }
.hf-evt-won   { border-left-color: #00e676;
                background: linear-gradient(90deg, rgba(0,230,118,0.06), transparent); }
.hf-evt-lost  { border-left-color: #ff4081;
                background: linear-gradient(90deg, rgba(255,64,129,0.06), transparent); }
.hf-evt-clv   { border-left-color: #a78bfa; }
.hf-evt-warn  { border-left-color: #ffd600;
                background: linear-gradient(90deg, rgba(255,214,0,0.06), transparent); }

.hf-evt-time { font-size: 0.86rem; color: #8892a4; font-weight: 700; }
.hf-evt-body { font-size: 0.96rem; color: #e8eaf0; line-height: 1.45; }
.hf-evt-body .hf-evt-match { color: #8892a4; font-weight: 600; margin-left: 0.4rem; }
.hf-evt-port-cell { text-align: right; }
.hf-evt-port {
    display: inline-block; padding: 0.2rem 0.55rem;
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    background: rgba(124,77,255,0.1); border: 1px solid rgba(124,77,255,0.3);
    color: #a5b4fc; border-radius: 999px;
}

@media (max-width: 1000px) {
    .home-feed-grid { grid-template-columns: repeat(2, 1fr); }
    .hf-evt { grid-template-columns: 70px 1fr; }
    .hf-evt-port-cell { display: none; }
}
@media (max-width: 600px) {
    .home-feed-grid { grid-template-columns: 1fr; }
}

/* Force st.button into tile shape — scoped by injecting CSS only on home view */
.home-tile-mode .stButton > button {
    min-height: 190px !important;
    font-size: 1.18rem !important;
    font-weight: 800 !important;
    white-space: pre-wrap !important;
    line-height: 1.55 !important;
    background: linear-gradient(135deg, rgba(124,77,255,0.04), rgba(0,229,255,0.02)) !important;
    border: 1px solid rgba(124,77,255,0.18) !important;
    border-radius: 22px !important;
    color: #e8eaf0 !important;
    padding: 1.6rem 1.3rem !important;
    transition: transform 0.3s cubic-bezier(.22,.61,.36,1),
                border-color 0.3s ease,
                background 0.3s ease,
                box-shadow 0.3s ease !important;
    animation: fadeInUp 0.55s cubic-bezier(.22,.61,.36,1) both;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
}
.home-tile-mode .stButton > button:hover {
    transform: translateY(-5px) !important;
    border-color: rgba(124,77,255,0.55) !important;
    background: linear-gradient(135deg, rgba(124,77,255,0.14), rgba(0,229,255,0.07)) !important;
    box-shadow: 0 16px 40px rgba(124,77,255,0.25) !important;
}

/* Compact top bar on non-home views */
.top-bar {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.4rem 0 1rem;
    margin-bottom: 0.3rem;
    animation: fadeInUp 0.4s cubic-bezier(.22,.61,.36,1) both;
}
.top-bar img {
    width: 72px; height: 72px;
    object-fit: cover; border-radius: 16px;
    box-shadow: 0 6px 22px rgba(124,77,255,0.28),
                0 0 0 1px rgba(255,255,255,0.06);
}
.top-bar-title {
    font-size: 2rem; font-weight: 900; color: #e8eaf0;
    letter-spacing: -0.6px; line-height: 1.1;
}
.top-bar-sub {
    font-size: 0.94rem; color: #8892a4; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 800; margin-top: 0.2rem;
}

/* ── Edge badge (value vs market) ─────────────────────────────────── */
.edge-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: linear-gradient(135deg, rgba(0,230,118,0.18), rgba(0,229,255,0.14));
    border: 1px solid rgba(0,230,118,0.45);
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.78rem; font-weight: 800; color: #00e676;
    letter-spacing: 1px;
    animation: softPulse 3s ease-in-out infinite;
}
.edge-badge.edge-neg {
    background: linear-gradient(135deg, rgba(255,64,129,0.15), rgba(124,77,255,0.08));
    border-color: rgba(255,64,129,0.35);
    color: #ff4081;
    animation: none;
}

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
    color: #b8c0d0 !important;
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
.pl-subtitle { font-size: 0.9rem; color: #b8c0d0; letter-spacing: 2px; text-transform: uppercase; font-weight: 500; }
.pl-meta { font-size: 0.88rem; color: #b8c0d0; margin-top: 0.6rem; line-height: 1.5; }

/* ── Divider ── */
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(124,77,255,0.4), transparent); margin: 2rem 0; }

/* ── Section label ── */
.section-label { font-size: 0.86rem; font-weight: 800; color: #a78bfa; text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 1.2rem; }

.vs-badge { text-align: center; font-size: 2rem; font-weight: 900;
            background: linear-gradient(135deg, #7c4dff, #00e5ff);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-top: 1.9rem; }

/* ── Probability cards ── */
.prob-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 1.8rem 1rem; text-align: center;
    position: relative; overflow: hidden;
}
.prob-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; }
.prob-card-home::before { background: linear-gradient(90deg, #6366f1, #22d3ee); }
.prob-card-draw::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.prob-card-away::before { background: linear-gradient(90deg, #e11d48, #fb7185); }

.prob-team { font-size: 0.95rem; font-weight: 800; color: #cdd; margin-bottom: 0.6rem;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob-pct  { font-size: 3.4rem; font-weight: 900; line-height: 1; margin-bottom: 0.3rem;
             letter-spacing: -1.5px; }
.prob-pct-home { color: #949bf7; }
.prob-pct-draw { color: #fbbf24; }
.prob-pct-away { color: #fb7185; }
.prob-label { font-size: 0.78rem; font-weight: 800; letter-spacing: 3px; text-transform: uppercase; color: #b8c0d0; }

/* ── Stacked probability bar — H / D / A with badges + bold % ── */
.prob-stack-bar {
    display: flex; width: 100%; height: 110px; border-radius: 18px;
    overflow: hidden; box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    margin: 0.6rem 0 1rem;
}
.prob-stack-seg {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0 1.1rem; min-width: 0; overflow: hidden;
    transform-origin: left center;
    animation: growFromLeft 0.7s cubic-bezier(.22,.61,.36,1);
}
.prob-stack-seg img {
    width: 64px; height: 64px; object-fit: contain;
    filter: drop-shadow(0 4px 10px rgba(0,0,0,0.6));
    flex-shrink: 0;
}
.seg-home {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #22d3ee 100%);
    justify-content: flex-start;
}
.seg-draw {
    background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
    justify-content: center;
}
.seg-away {
    background: linear-gradient(135deg, #fb7185 0%, #e11d48 50%, #9f1239 100%);
    justify-content: flex-end;
}
.seg-pct {
    font-size: 1.7rem; font-weight: 900; color: #fff;
    text-shadow: 0 2px 3px rgba(0,0,0,0.45); letter-spacing: -0.6px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap; line-height: 1;
}
.seg-pct-dark { /* on: #ffd600 */ color: #1a1d27; text-shadow: 0 1px 2px rgba(255,255,255,0.22); }

/* ── Model comparison cards ── */
.model-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 0.5rem;
}
.model-title { font-size: 0.78rem; font-weight: 800; color: #b8c0d0; text-transform: uppercase; letter-spacing: 2px; }

/* Accuracy badge in the corner of each model card */
.model-acc-badge {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 0.32rem 0.7rem; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    line-height: 1; flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}
.model-acc-num { font-size: 1.15rem; font-weight: 900; line-height: 1; }
.model-acc-lbl { font-size: 0.78rem; font-weight: 800; letter-spacing: 1.2px;
                 text-transform: uppercase; color: #b8c0d0; margin-top: 0.25rem; }
.model-acc-na  { color: #b8c0d0; }

/* ── Scoreline card ── */
.score-hero {
    background: linear-gradient(135deg, rgba(124,77,255,0.15), rgba(0,229,255,0.08));
    border: 1px solid rgba(124,77,255,0.35); border-radius: 24px; padding: 2.5rem 2rem; text-align: center;
}
.score-hero-label { font-size: 0.92rem; font-weight: 800; color: #8892a4; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 1rem; }
.score-digits { font-size: clamp(3.5rem, 8vw, 5.5rem); font-weight: 900; color: #e8eaf0; letter-spacing: -2px; line-height: 1; }
.score-dash { color: #9aa6ba; margin: 0 0.5rem; }
.score-prob { margin-top: 0.9rem; font-size: 1rem; color: #00e5ff; font-weight: 700; letter-spacing: 0.3px; }
.score-ci { margin-top: 1.2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; }
.ci-item { font-size: 0.92rem; color: #8892a4; line-height: 1.4; }
.ci-range { font-weight: 800; color: #cdd; }

/* ── Top scorelines table ── */
.score-table { width: 100%; border-collapse: collapse; }
.score-table th { font-size: 0.78rem; font-weight: 700; color: #b8c0d0; text-transform: uppercase;
                  letter-spacing: 2px; padding: 0.4rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.06); text-align: left; }
.score-table td { padding: 0.55rem 0.8rem; font-size: 0.85rem; color: #ccd; border-bottom: 1px solid rgba(255,255,255,0.04); }
.score-table tr:hover td { background: rgba(124,77,255,0.06); }
.score-table tr:first-child td { font-weight: 700; color: #e8eaf0; }
.badge-H { background: rgba(61,110,255,0.15); color: #7fa3ff; padding: 2px 8px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
.badge-D { background: rgba(255,214,0,0.15);  color: #ffd600; padding: 2px 8px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
.badge-A { background: rgba(255,64,129,0.15); color: #ff4081; padding: 2px 8px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }

/* ── Form badges ── */
.form-row { display: flex; gap: 6px; align-items: flex-start; flex-wrap: wrap; }
.form-item { display: flex; flex-direction: column; align-items: center; gap: 3px; }
.form-badge { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 800; }
.form-W { background: #00e676; color: #003; }
.form-D { background: #ffd600; color: #332200; }
.form-L { background: #ff4081; color: #2a0012; }
.form-score { font-size: 0.78rem; color: #b8c0d0; font-weight: 700; }
.form-opp   { font-size: 0.78rem;  color: #b8c0d0; max-width: 56px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }

/* ── Form card ── */
.form-card { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 1.4rem 1.6rem; }
.form-team-name { font-size: 1rem; font-weight: 700; color: #ccd; margin-bottom: 0.4rem; }
.form-stats { font-size: 0.82rem; color: #b8c0d0; margin-bottom: 1rem; }
.form-stats span { color: #8892a4; font-weight: 600; margin-right: 0.8rem; }

/* ── H2H section ── */
.h2h-summary { display: flex; justify-content: center; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.h2h-stat { text-align: center; }
.h2h-num { font-size: 2rem; font-weight: 900; }
.h2h-lbl { font-size: 0.78rem; color: #b8c0d0; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }

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
    font-size: 1.15rem; font-weight: 800; color: #e8eaf0;
    display: flex; align-items: center; gap: 0.5rem;
}
.fixture-team-home { justify-content: flex-end; text-align: right; }
.fixture-team-away { justify-content: flex-start; text-align: left; }
.fixture-prob-bar {
    display: flex; border-radius: 10px; overflow: hidden; height: 36px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.35);
}
.bar-home { background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #22d3ee 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.86rem; font-weight: 900; color: #fff; min-width: 26px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.4);
            font-variant-numeric: tabular-nums; letter-spacing: -0.2px; }
.bar-draw { background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.86rem; font-weight: 900; color: #1a1d27; min-width: 26px;
            text-shadow: 0 1px 2px rgba(255,255,255,0.18);
            font-variant-numeric: tabular-nums; letter-spacing: -0.2px; }
.bar-away { background: linear-gradient(135deg, #d1123c 0%, #b30f34 50%, #9f1239 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 0.86rem; font-weight: 900; color: #fff; min-width: 26px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.4);
            font-variant-numeric: tabular-nums; letter-spacing: -0.2px; }
/* Compact variants for the result-card and model-comparison strips */
.bar-home.bar-sm, .bar-draw.bar-sm, .bar-away.bar-sm {
    font-size: 0.78rem; min-width: 20px;
}
.fixture-footer {
    display: flex; justify-content: center; gap: 2rem;
    margin-top: 1rem; flex-wrap: wrap;
}
.fixture-xg { font-weight: 700; color: #a78bfa; }
.fixture-score { font-weight: 700; color: #aab; }

/* xG chip — highlighted box, hover-tooltip explains expected goals */
.fixture-xg-chip {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 1rem;
    background: linear-gradient(135deg, rgba(124,77,255,0.15), rgba(0,229,255,0.06));
    border: 1px solid rgba(124,77,255,0.35);
    border-radius: 999px;
    font-variant-numeric: tabular-nums;
    cursor: help;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.fixture-xg-chip:hover {
    border-color: rgba(124,77,255,0.6);
    background: linear-gradient(135deg, rgba(124,77,255,0.25), rgba(0,229,255,0.10));
}
.fixture-xg-chip .xg-label {
    font-size: 0.8rem; font-weight: 800; letter-spacing: 2px;
    color: #a78bfa; text-transform: uppercase;
}
.fixture-xg-chip .xg-home { font-size: 1.25rem; font-weight: 900; color: #949bf7; }
.fixture-xg-chip .xg-away { font-size: 1.25rem; font-weight: 900; color: #fb7185; }
.fixture-xg-chip .xg-dash { font-size: 1.15rem; font-weight: 700; color: #b8c0d0; }

/* Universal bright backdrop for team badges so dark/black crests (Tottenham,
   Newcastle, dark Liverpool variants) lift visibly against the dark UI.
   The badge sits ON TOP of a circular light halo — never blurred. */
.team-badge {
    background: radial-gradient(circle at 50% 50%,
                rgba(255,255,255,0.32) 0%,
                rgba(255,255,255,0.20) 45%,
                rgba(255,255,255,0.08) 75%,
                transparent 95%);
    border-radius: 50%;
    padding: 5px;
    box-sizing: border-box;
}
/* Slightly stronger halo on bigger badges (predict-tab picker, fixture cards) */
.team-badge[width="40"], .team-badge[width="44"],
.team-badge[width="80"], .team-badge[width="120"], .team-badge[width="200"] {
    background: radial-gradient(circle at 50% 50%,
                rgba(255,255,255,0.38) 0%,
                rgba(255,255,255,0.22) 50%,
                rgba(255,255,255,0.08) 80%,
                transparent 100%);
    padding: 7px;
}

/* ── Backtest table ── */
.bt-correct   { color: #00e676; font-weight: 700; }
.bt-incorrect { color: #ff4081; }
.metric-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem; text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 900; color: #e8eaf0;
                line-height: 1.05; letter-spacing: -0.7px;
                font-variant-numeric: tabular-nums; }
.metric-label { font-size: 0.86rem; color: #8892a4; font-weight: 800;
                text-transform: uppercase; letter-spacing: 1.6px; margin-top: 0.4rem; }
.metric-better { color: #00e676; }
.metric-worse  { color: #ff4081; }

.stSpinner > div { border-top-color: #a78bfa !important; }

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
.pnl-tag { font-size: 0.8rem; font-weight: 800; letter-spacing: 3px; text-transform: uppercase; color: #b8c0d0; margin-bottom: 0.8rem; }
.pnl-amount { font-size: clamp(2.8rem, 7vw, 5rem); font-weight: 900; line-height: 1; margin-bottom: 0.4rem; }

/* ── Portfolio journey panel — Started → Now → Potential ── */
.pnl-journey {
    display: grid;
    grid-template-columns: 1fr auto 1.2fr auto 1.4fr;
    gap: 1rem; align-items: stretch;
    margin: 0.8rem 0 1.5rem;
    animation: fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
}
.pj-step {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.1rem 1.3rem;
    display: flex; flex-direction: column; gap: 0.3rem;
    justify-content: center;
}
.pj-start  { border-left: 4px solid #8892a4; }
.pj-now    { border-left: 4px solid #7c4dff;
             background: linear-gradient(180deg, rgba(124,77,255,0.10), rgba(124,77,255,0.02)); }
.pj-future { border-left: 4px solid #00e5ff; padding: 0.7rem 0.8rem; }

.pj-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px;
    text-transform: uppercase; color: #8892a4;
}
.pj-val {
    font-size: 2.2rem; font-weight: 900; color: #e8eaf0;
    line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: -1px;
    margin-top: 0.15rem;
}
.pj-sub {
    font-size: 0.92rem; color: #8892a4; line-height: 1.4;
    margin-top: 0.3rem;
}

.pj-arrow {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    font-size: 2.2rem; font-weight: 900; line-height: 1;
    min-width: 80px;
}
.pj-arrow-sub {
    font-size: 0.92rem; font-weight: 800; line-height: 1.2;
    text-align: center; margin-top: 0.4rem;
    font-variant-numeric: tabular-nums;
}
.pj-arrow-future { min-width: 130px; }
.pj-future-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #00e5ff; text-align: center; line-height: 1.4;
}

.pj-future-row { display: flex; gap: 0.6rem; }
.pj-future-half {
    flex: 1; padding: 0.65rem 0.8rem; border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
.pj-future-win  { border-left: 3px solid #00e676;
                  background: linear-gradient(180deg, rgba(0,230,118,0.10), rgba(0,230,118,0.02)); }
.pj-future-lose { border-left: 3px solid #ff4081;
                  background: linear-gradient(180deg, rgba(255,64,129,0.08), rgba(255,64,129,0.02)); }
.pj-future-lbl-small {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; color: #8892a4;
}
.pj-future-val {
    font-size: 1.55rem; font-weight: 900; line-height: 1;
    margin-top: 0.25rem; font-variant-numeric: tabular-nums; letter-spacing: -0.5px;
}
.pj-future-win-val  { color: #00e676; }
.pj-future-lose-val { color: #ff4081; }

@media (max-width: 900px) {
    .pnl-journey { grid-template-columns: 1fr; }
    .pj-arrow { flex-direction: row; gap: 0.6rem; min-height: 0; padding: 0.5rem 0; }
    .pj-arrow-sub { margin-top: 0; }
}
.pnl-profit .pnl-amount { color: #00e676; }
.pnl-loss   .pnl-amount { color: #ff4081; }
.pnl-neutral .pnl-amount { color: #e8eaf0; }
.pnl-subtitle { font-size: 0.9rem; color: #b8c0d0; font-weight: 500; }

/* ── Portfolio: Stat cards ── */
.pstat-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1rem 1.1rem; text-align: center;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.pstat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(124,77,255,0.35);
    box-shadow: 0 8px 24px rgba(124,77,255,0.10);
}
.pstat-val { font-size: 1.85rem; font-weight: 900; color: #e8eaf0; line-height: 1.05;
             font-variant-numeric: tabular-nums; letter-spacing: -0.5px; }
.pstat-lbl { font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px; text-transform: uppercase; color: #b8c0d0; margin-top: 0.3rem; }

/* ── Portfolio: Pending bet cards ── */
.pend-card {
    background: rgba(255,214,0,0.05); border: 1px solid rgba(255,214,0,0.12);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
}
.pend-match  { font-size: 0.88rem; font-weight: 700; color: #e8eaf0; }
.pend-sel    { font-size: 0.78rem; font-weight: 700; margin-top: 0.2rem; }
.pend-meta   { display: flex; gap: 1rem; font-size: 0.82rem; color: #b8c0d0; margin-top: 0.35rem; flex-wrap: wrap; }
.pend-date   { font-size: 0.78rem; color: #9aa6ba; margin-top: 0.25rem; }
.pend-return {
    display: flex; align-items: center; gap: 0.6rem;
    margin-top: 0.45rem; padding: 0.5rem 0.75rem;
    background: rgba(0,230,118,0.06); border: 1px solid rgba(0,230,118,0.15);
    border-radius: 8px;
}
.pend-return-label { font-size: 0.78rem; color: #b8c0d0; text-transform: uppercase; letter-spacing: 1px; }
.pend-return-val   { font-size: 1.15rem; font-weight: 800; color: #00e676; }
.pend-return-profit { font-size: 0.88rem; color: #69f0ae; font-weight: 700; }

/* ── Pending bet cards v2 — bigger badges, bolder layout ── */
.pend-card-v2 {
    background: linear-gradient(180deg, rgba(255,214,0,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,214,0,0.2);
    border-left: 4px solid #ffd600;
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 0.6rem;
    transition: border-color 0.22s ease, transform 0.22s ease, box-shadow 0.22s ease;
    animation: fadeInUp 0.45s cubic-bezier(.22,.61,.36,1) both;
}
.pend-card-v2:hover {
    border-color: #ffd600;
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(255,214,0,0.15);
}
.pend-v2-top {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.85rem;
}
.pend-v2-date {
    font-size: 0.92rem; font-weight: 800; color: #8892a4;
    letter-spacing: 0.3px;
}
.pend-v2-badge {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.5px;
    padding: 0.25rem 0.7rem; border-radius: 999px;
    border: 1px solid; text-transform: uppercase;
}
.pend-v2-teams {
    display: grid; grid-template-columns: 1fr auto 1fr;
    gap: 1rem; align-items: center;
    padding: 0.6rem 0.4rem;
    background: rgba(0,0,0,0.18);
    border-radius: 12px;
    margin-bottom: 0.85rem;
}
.pend-v2-team {
    display: flex; flex-direction: column; align-items: center; gap: 0.4rem;
    text-align: center;
}
.pend-v2-team img { width: 56px; height: 56px; object-fit: contain;
                    filter: drop-shadow(0 4px 10px rgba(0,0,0,0.5)); }
.pend-v2-team span {
    font-size: 0.95rem; font-weight: 700; color: #e8eaf0;
    line-height: 1.2; max-width: 130px;
}
.pend-v2-vs {
    font-size: 1.4rem; font-weight: 900;
    background: linear-gradient(135deg, #7c4dff, #00e5ff);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pend-v2-pick {
    font-size: 1.35rem; font-weight: 900; text-align: center;
    margin-bottom: 0.85rem; letter-spacing: -0.3px;
    font-variant-numeric: tabular-nums;
}
.pend-v2-pick .pend-v2-at { color: #b8c0d0; font-weight: 700; }
.pend-v2-pick .pend-v2-odds { color: #00e5ff; }
.pend-v2-stats {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 0.6rem; margin-bottom: 0.85rem;
}
.pend-v2-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.55rem 0.6rem; text-align: center;
}
.pend-v2-stat-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.pend-v2-stat-val {
    font-size: 1.15rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; margin-top: 0.2rem;
    font-variant-numeric: tabular-nums; letter-spacing: -0.3px;
}
.pend-v2-model { color: #a78bfa; }
.pend-v2-ev    { color: #00e676; }
.pend-v2-return {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.85rem 1.1rem;
    background: linear-gradient(135deg, rgba(0,230,118,0.13), rgba(0,229,255,0.06));
    border: 1px solid rgba(0,230,118,0.3);
    border-radius: 12px;
    flex-wrap: wrap;
}
.pend-v2-return-block { flex: 1; min-width: 0; }
.pend-v2-return-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.pend-v2-return-val {
    font-size: 1.5rem; font-weight: 900; color: #e8eaf0;
    line-height: 1; margin-top: 0.2rem;
    font-variant-numeric: tabular-nums; letter-spacing: -0.5px;
}
.pend-v2-return-profit {
    font-size: 1.5rem; font-weight: 900; color: #00e676;
    line-height: 1; margin-top: 0.2rem;
    font-variant-numeric: tabular-nums; letter-spacing: -0.5px;
}
.pend-v2-return-pct {
    font-size: 0.84rem; font-weight: 700; color: #69f0ae;
    margin-top: 0.2rem;
}
.pend-v2-arrow {
    font-size: 1.5rem; font-weight: 900; color: #00e676;
    flex-shrink: 0;
}

.bet-return-box {
    background: linear-gradient(135deg, rgba(0,230,118,0.08), rgba(124,77,255,0.06));
    border: 1px solid rgba(0,230,118,0.2);
    border-radius: 12px; padding: 0.7rem 1rem; margin: 0.5rem 0;
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
}
.bet-return-total { font-size: 1.4rem; font-weight: 800; color: #00e676; }
.bet-return-detail { font-size: 0.78rem; color: #69f0ae; font-weight: 600; }

/* ── Bet History — custom card-style table (Mock Portfolio singles) ── */
/* Scrollable container — keeps the table from dominating the page */
.bh-scroll {
    max-height: 600px;
    overflow-y: auto;
    padding: 0.4rem 0.4rem 0.4rem 0;
    border-radius: 12px;
    background: rgba(0,0,0,0.18);
    border: 1px solid rgba(255,255,255,0.05);
}
.bh-scroll::-webkit-scrollbar          { width: 10px; }
.bh-scroll::-webkit-scrollbar-track    { background: rgba(255,255,255,0.02); border-radius: 8px; }
.bh-scroll::-webkit-scrollbar-thumb    { background: rgba(124,77,255,0.4); border-radius: 8px; }
.bh-scroll::-webkit-scrollbar-thumb:hover { background: rgba(124,77,255,0.6); }
.bh-table {
    display: flex; flex-direction: column; gap: 0.4rem;
    margin-bottom: 0.8rem;
    font-variant-numeric: tabular-nums;
}
/* Bigger team badges in the bet history rows */
.bh-team img.team-badge { width: 44px !important; height: 44px !important; }

/* ── CLV Hero panel — unified all-time + recent-trend layout ── */
.clv-hero {
    margin-bottom: 1rem;
    animation: fadeInUp 0.45s cubic-bezier(.22,.61,.36,1) both;
}
.clv-hero-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
}
.clv-hero-side {
    background: linear-gradient(180deg, rgba(124,77,255,0.06), rgba(124,77,255,0.01));
    border: 1px solid rgba(124,77,255,0.18);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    display: flex; flex-direction: column; gap: 0.85rem;
}
.clv-hero-recent {
    background: linear-gradient(180deg, rgba(0,229,255,0.06), rgba(0,229,255,0.01));
    border-color: rgba(0,229,255,0.2);
}
.clv-hero-side-lbl {
    font-size: 0.84rem; font-weight: 800; letter-spacing: 1.6px;
    text-transform: uppercase; color: #a78bfa;
}
.clv-hero-recent .clv-hero-side-lbl { color: #00e5ff; }
.clv-hero-row {
    display: flex; gap: 1rem; flex-wrap: wrap; align-items: stretch;
}
.clv-hero-block {
    flex: 1; min-width: 100px;
    display: flex; flex-direction: column; gap: 0.25rem;
}
.clv-hero-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.clv-hero-val {
    font-size: 1.85rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; font-variant-numeric: tabular-nums; letter-spacing: -0.5px;
}
.clv-hero-status {
    padding: 0.7rem 0.95rem; border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid;
    font-size: 0.94rem; line-height: 1.45;
}
.clv-hero-status b { font-size: 1rem; }
.clv-hero-note {
    font-size: 0.86rem; color: #8892a4; margin-top: 0.3rem;
}

/* Per-window mini blocks inside the Recent panel */
.clv-win {
    flex: 1; min-width: 90px;
    background: rgba(0,0,0,0.18);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    text-align: center;
}
.clv-win-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.clv-win-val {
    font-size: 1.4rem; font-weight: 900; color: #e8eaf0;
    margin-top: 0.2rem; line-height: 1;
    font-variant-numeric: tabular-nums; letter-spacing: -0.3px;
}
.clv-win-val.clv-win-na { color: #b8c0d0; }
.clv-win-pos {
    font-size: 0.78rem; color: #8892a4; margin-top: 0.2rem; font-weight: 600;
}

@media (max-width: 900px) {
    .clv-hero-grid { grid-template-columns: 1fr; }
}

/* ── CLV Guide (inside the "What do these numbers mean?" expander) ── */
.clv-guide {
    color: #e8eaf0; line-height: 1.6;
    font-size: 1rem;
    display: flex; flex-direction: column; gap: 1.1rem;
    padding: 0.4rem 0.2rem;
}

/* Hero header inside the guide */
.clv-guide-hero {
    display: flex; align-items: center; gap: 1.2rem;
    padding: 1.4rem 1.6rem;
    background: linear-gradient(135deg, rgba(124,77,255,0.18), rgba(0,229,255,0.06));
    border: 1px solid rgba(124,77,255,0.35);
    border-radius: 16px;
}
.clv-guide-hero-icon {
    font-size: 3.2rem; line-height: 1;
    filter: drop-shadow(0 4px 10px rgba(124,77,255,0.4));
}
.clv-guide-hero-title {
    font-size: 1.6rem; font-weight: 900; color: #e8eaf0;
    letter-spacing: -0.4px; line-height: 1.1;
    background: linear-gradient(135deg, #7c4dff, #00e5ff);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.clv-guide-hero-sub {
    font-size: 1.05rem; color: #cdd; margin-top: 0.3rem; line-height: 1.5;
}

/* Worked example block */
.clv-guide-example {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
.clv-guide-example-title {
    font-size: 0.92rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #ffd600; margin-bottom: 0.85rem;
}
.clv-guide-example-row {
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
    margin-bottom: 0.85rem;
}
.clv-guide-example-block {
    flex: 1; min-width: 130px;
    background: rgba(0,0,0,0.2); padding: 0.85rem 1rem;
    border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
}
.clv-guide-example-arrow {
    font-size: 1.6rem; font-weight: 900; color: #a78bfa;
    flex-shrink: 0;
}
.clv-guide-eg-lbl {
    font-size: 0.8rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.clv-guide-eg-val {
    font-size: 1.5rem; font-weight: 900; color: #e8eaf0;
    margin-top: 0.3rem; font-variant-numeric: tabular-nums;
    line-height: 1; letter-spacing: -0.3px;
}
.clv-guide-eg-took   { color: #00e5ff; }
.clv-guide-eg-close  { color: #8892a4; }
.clv-guide-eg-result { color: #00e676; }
.clv-guide-example-note {
    font-size: 0.96rem; color: #cdd; line-height: 1.5;
    padding: 0.7rem 0.95rem; border-radius: 8px;
    background: rgba(255,214,0,0.06); border-left: 3px solid #ffd600;
}

/* CLV vs ROI comparison */
.clv-guide-vs {
    display: flex; align-items: stretch; gap: 1rem;
}
.clv-guide-vs-card {
    flex: 1; padding: 1.2rem 1.4rem; border-radius: 14px;
    display: flex; flex-direction: column; gap: 0.5rem;
}
.clv-guide-roi {
    background: linear-gradient(135deg, rgba(255,64,129,0.10), rgba(255,64,129,0.02));
    border: 1px solid rgba(255,64,129,0.3);
}
.clv-guide-clv {
    background: linear-gradient(135deg, rgba(0,230,118,0.10), rgba(0,229,255,0.04));
    border: 1px solid rgba(0,230,118,0.4);
}
.clv-guide-vs-icon { font-size: 2rem; line-height: 1; }
.clv-guide-vs-title { font-size: 1.4rem; font-weight: 900; color: #e8eaf0; }
.clv-guide-vs-body  { font-size: 1rem; color: #cdd; line-height: 1.55; }
.clv-guide-vs-vs {
    font-size: 1.4rem; font-weight: 900;
    color: #a78bfa; align-self: center; padding: 0 0.4rem;
}

/* The four metrics — header + per-metric block */
.clv-guide-metric-title {
    font-size: 1.2rem; font-weight: 900; color: #e8eaf0;
    letter-spacing: -0.3px; margin-top: 0.4rem;
}
.clv-guide-metric {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 4px solid #7c4dff;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    display: flex; flex-direction: column; gap: 0.5rem;
}
.clv-guide-metric-name {
    font-size: 1.18rem; font-weight: 900; color: #e8eaf0;
    letter-spacing: -0.2px;
}
.clv-guide-metric-desc { font-size: 1rem; color: #cdd; line-height: 1.55; }

/* Threshold bands inside Median CLV */
.clv-guide-bands {
    display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.3rem;
}
.clv-guide-band {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0.55rem 0.85rem; border-radius: 8px;
    font-size: 0.96rem; font-weight: 700;
}
.clv-guide-band-tier {
    flex: 0 0 110px;
    font-size: 0.92rem; font-weight: 900;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.3px;
}
.clv-band-sharp { background: rgba(0,229,255,0.10);  border: 1px solid rgba(0,229,255,0.35); color: #00e5ff; }
.clv-band-good  { background: rgba(0,230,118,0.10);  border: 1px solid rgba(0,230,118,0.35); color: #00e676; }
.clv-band-warn  { background: rgba(255,214,0,0.10);  border: 1px solid rgba(255,214,0,0.35); color: #ffd600; }
.clv-band-bad   { background: rgba(255,64,129,0.10); border: 1px solid rgba(255,64,129,0.35); color: #ff4081; }

/* Target / what success looks like */
.clv-guide-target {
    background: linear-gradient(135deg, rgba(0,230,118,0.13), rgba(0,229,255,0.05));
    border: 1px solid rgba(0,230,118,0.4);
    border-left: 5px solid #00e676;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
.clv-guide-target-title {
    font-size: 1.18rem; font-weight: 900; color: #00e676; letter-spacing: -0.2px;
    margin-bottom: 0.4rem;
}
.clv-guide-target-body { font-size: 1rem; color: #cdd; line-height: 1.55; }

@media (max-width: 800px) {
    .clv-guide-vs { flex-direction: column; }
    .clv-guide-vs-vs { padding: 0.4rem 0; }
    .clv-guide-example-row { flex-direction: column; align-items: stretch; }
    .clv-guide-example-arrow { transform: rotate(90deg); align-self: center; }
}

/* ── Season simulation — Projected Final Table ── */
.sim-table {
    display: flex; flex-direction: column; gap: 0.45rem;
    margin-bottom: 1rem;
    font-variant-numeric: tabular-nums;
}
.sim-row {
    display: grid;
    grid-template-columns:
        70px        /* rank */
        1.6fr       /* team */
        46px        /* played */
        62px        /* pts now */
        72px        /* proj μ */
        82px        /* most likely */
        108px       /* elo */
        90px        /* form sparkline */
        2.0fr;      /* scenarios */
    gap: 0.7rem; align-items: center;
    padding: 0.85rem 1.1rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 4px solid transparent;
    border-radius: 12px;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    animation: fadeInUp 0.4s cubic-bezier(.22,.61,.36,1) both;
}
.sim-row:hover {
    transform: translateX(2px);
    box-shadow: 0 6px 20px rgba(124,77,255,0.10);
}
.sim-row.sim-header {
    background: transparent; border: none; padding: 0.4rem 1.1rem 0.2rem;
    font-size: 0.78rem; font-weight: 800; color: #a78bfa;
    letter-spacing: 1.6px;
}
.sim-row.sim-header:hover { transform: none; box-shadow: none; }

/* Zone stripes */
.sim-row.zone-title { border-left-color: #ffd600; background: linear-gradient(180deg, rgba(255,214,0,0.08), rgba(255,214,0,0.01)); }
.sim-row.zone-top4  { border-left-color: #00e5ff; background: linear-gradient(180deg, rgba(0,229,255,0.06), rgba(0,229,255,0.01)); }
.sim-row.zone-top6  { border-left-color: #a78bfa; background: linear-gradient(180deg, rgba(124,77,255,0.06), rgba(124,77,255,0.01)); }
.sim-row.zone-rel   { border-left-color: #ff4081; background: linear-gradient(180deg, rgba(255,64,129,0.06), rgba(255,64,129,0.01)); }

.sim-rank {
    font-size: 1.5rem; font-weight: 900; color: #cdd;
    line-height: 1.05; display: flex; align-items: baseline; gap: 0.35rem;
    letter-spacing: -0.5px;
}
.sim-zone-emoji { font-size: 1.05rem; line-height: 1; }

.sim-team {
    display: flex; align-items: center; gap: 0.7rem;
    min-width: 0;
}
.sim-team img.team-badge { width: 44px !important; height: 44px !important; }
.sim-team > span {
    font-size: 1.05rem; font-weight: 800; color: #e8eaf0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    letter-spacing: -0.2px;
}

.sim-played   { font-size: 1rem;   color: #8892a4; font-weight: 700; text-align: center; }
.sim-pts-now  { font-size: 1.4rem; font-weight: 900; color: #e8eaf0;
                text-align: center; line-height: 1.05; letter-spacing: -0.5px; }
.sim-pts-proj { font-size: 1.2rem; font-weight: 800; color: #a78bfa;
                text-align: center; line-height: 1.05; letter-spacing: -0.3px; }
.sim-pts-ml   { font-size: 1.2rem; font-weight: 800; color: #00e5ff;
                text-align: center; line-height: 1.05; letter-spacing: -0.3px;
                cursor: help; }

.sim-elo {
    display: flex; flex-direction: column; align-items: center; gap: 0.1rem;
}
.sim-elo-num   { font-size: 1.05rem; font-weight: 900; color: #e8eaf0;
                 letter-spacing: -0.3px; line-height: 1; }
.sim-elo-delta { font-size: 0.86rem; font-weight: 800; line-height: 1;
                 letter-spacing: -0.2px; }

.sim-spark { line-height: 0; text-align: center; }

.sim-pills {
    display: flex; flex-wrap: wrap; gap: 0.35rem;
}
.sim-pill {
    display: inline-block;
    padding: 0.32rem 0.7rem; border-radius: 999px;
    font-size: 0.86rem; font-weight: 700;
    letter-spacing: 0.2px; white-space: nowrap;
}
.sim-pill b { font-weight: 900; }
.sim-pill-none { color: #b8c0d0; font-size: 0.92rem; }

@media (max-width: 1100px) {
    .sim-row { grid-template-columns: repeat(2, 1fr); padding: 0.85rem; gap: 0.5rem; }
    .sim-row.sim-header { display: none; }
    .sim-team { grid-column: span 2; }
}

/* ── Season simulator math sanity banner ── */
.sim-math-banner {
    display: flex; align-items: center; gap: 1.6rem;
    padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(0,230,118,0.08), rgba(124,77,255,0.04));
    border: 1px solid rgba(0,230,118,0.25);
    border-left: 5px solid;
    border-radius: 14px;
    flex-wrap: wrap;
}
.sim-math-icon {
    font-size: 2rem; line-height: 1; flex-shrink: 0;
    filter: drop-shadow(0 3px 8px rgba(124,77,255,0.5));
}
.sim-math-block {
    display: flex; flex-direction: column; gap: 0.2rem;
    min-width: 140px;
}
.sim-math-block.sim-math-verdict { flex: 1; min-width: 200px; }
.sim-math-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.sim-math-val {
    font-size: 1.6rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; font-variant-numeric: tabular-nums;
    letter-spacing: -0.4px;
}
.sim-math-sub {
    font-size: 0.86rem; color: #8892a4; line-height: 1.4;
}
.bh-row {
    display: grid;
    grid-template-columns:
        56px        /* # */
        110px       /* date */
        1.4fr       /* home */
        1.4fr       /* away */
        1.5fr       /* selection */
        82px        /* odds */
        1.6fr       /* why we bet */
        110px       /* stake */
        110px       /* result */
        140px;      /* P&L */
    gap: 0.7rem; align-items: center;
    padding: 0.85rem 1.1rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    transition: border-color 0.22s ease, transform 0.22s ease, box-shadow 0.22s ease;
    animation: fadeInUp 0.4s cubic-bezier(.22,.61,.36,1) both;
}
.bh-row:hover {
    border-color: rgba(124,77,255,0.35);
    transform: translateX(2px);
    box-shadow: 0 6px 20px rgba(124,77,255,0.10);
}
.bh-row.bh-header {
    background: transparent; border: none; padding: 0.4rem 1.1rem 0.2rem;
    font-size: 0.78rem; font-weight: 800; color: #a78bfa;
    letter-spacing: 1.6px;
}
.bh-row.bh-header:hover { transform: none; box-shadow: none; }
.bh-row.bh-row-highlight {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.5),
                0 8px 28px rgba(0,229,255,0.18);
    background: linear-gradient(180deg, rgba(0,229,255,0.06), rgba(255,255,255,0.01));
}

.bh-num   { font-size: 1rem; font-weight: 800; color: #a78bfa;
            letter-spacing: -0.3px; }
.bh-date  { font-size: 0.92rem; color: #cdd; font-weight: 700;
            letter-spacing: 0.2px; }
.bh-team  { display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.95rem; font-weight: 700; color: #e8eaf0; min-width: 0; }
.bh-team img { flex-shrink: 0; }
.bh-team > :last-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bh-sel   { font-size: 0.95rem; font-weight: 700; color: #ffd600; }
.bh-odds  { font-size: 1.1rem; font-weight: 900; color: #00e5ff;
            letter-spacing: -0.3px; }

/* Why we bet — model vs bookie probability stack */
.bh-edge {
    display: flex; flex-direction: column; gap: 0.15rem;
    padding: 0.4rem 0.65rem;
    background: rgba(124,77,255,0.06);
    border: 1px solid rgba(124,77,255,0.18);
    border-radius: 8px;
    cursor: help;
}
.bh-edge-row {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 0.5rem;
}
.bh-edge-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; color: #8892a4;
}
.bh-edge-num {
    font-size: 0.95rem; font-weight: 900; color: #cdd;
}
.bh-edge-gap {
    font-size: 0.78rem; font-weight: 800; margin-top: 0.15rem;
    text-align: center; letter-spacing: 0.3px;
    border-top: 1px dashed rgba(255,255,255,0.06); padding-top: 0.2rem;
}

.bh-stake { font-size: 1.05rem; font-weight: 800; color: #e8eaf0;
            text-align: right; }

/* Result chip */
.bh-result {
    display: inline-block; padding: 0.32rem 0.7rem;
    font-size: 0.86rem; font-weight: 800;
    border-radius: 999px; letter-spacing: 0.3px;
}
.bh-result.bh-won  { background: rgba(0,230,118,0.15); color: #00e676;
                     border: 1px solid rgba(0,230,118,0.35); }
.bh-result.bh-lost { background: rgba(255,64,129,0.15); color: #ff4081;
                     border: 1px solid rgba(255,64,129,0.35); }

/* P&L chip — large & loud */
.bh-pnl {
    display: inline-block; padding: 0.45rem 0.85rem;
    font-size: 1.1rem; font-weight: 900;
    border-radius: 10px; letter-spacing: -0.3px;
}
.bh-pnl.bh-pnl-pos {
    color: #00e676;
    background: linear-gradient(135deg, rgba(0,230,118,0.18), rgba(0,229,255,0.06));
    border: 1px solid rgba(0,230,118,0.4);
}
.bh-pnl.bh-pnl-neg {
    color: #ff4081;
    background: linear-gradient(135deg, rgba(255,64,129,0.16), rgba(159,18,57,0.05));
    border: 1px solid rgba(255,64,129,0.35);
}
.bh-pnl.bh-pnl-flat {
    color: #8892a4; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.bh-cell-result, .bh-cell-pnl { text-align: right; }

@media (max-width: 1100px) {
    .bh-row { grid-template-columns: repeat(2, 1fr); padding: 0.85rem; gap: 0.5rem; }
    .bh-row.bh-header { display: none; }
}

/* ── Portfolio: Value scanner cards (legacy class kept for accumulator builder) ── */
.scan-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
}
.scan-match { font-size: 0.95rem; font-weight: 700; color: #e8eaf0; flex: 1; min-width: 180px; }
.scan-probs { display: flex; gap: 1.2rem; align-items: center; }
.scan-prob-item { text-align: center; }
.scan-pct  { font-size: 1.3rem; font-weight: 800; line-height: 1; }
.scan-lbl  { font-size: 0.84rem; font-weight: 800; letter-spacing: 1.8px; text-transform: uppercase; color: #b8c0d0; }
.scan-odds-source { font-size: 0.86rem; color: #8892a4; }

/* ── New scanner fixture card — fixture-tab style, big badges + gradient bar ── */
.scan-fixture-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 0.3rem;
    animation: fadeInUp 0.4s cubic-bezier(.22,.61,.36,1) both;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.scan-fixture-card:hover {
    border-color: rgba(124,77,255,0.25);
    box-shadow: 0 6px 24px rgba(124,77,255,0.10);
}
.scan-fixture-row {
    display: grid; grid-template-columns: 1fr 2.4fr 1fr;
    gap: 1.1rem; align-items: center;
}
.scan-fixture-row .fixture-team-name { font-size: 1.15rem; }

/* O/U 2.5 secondary bar */
.scan-ou-row {
    display: flex; align-items: center; gap: 1rem;
    margin-top: 0.85rem; padding-top: 0.85rem;
    border-top: 1px dashed rgba(255,255,255,0.06);
}
.scan-ou-lbl {
    font-size: 0.84rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #a78bfa; min-width: 130px;
}
.scan-ou-bar {
    flex: 1; display: flex; height: 26px; border-radius: 8px;
    overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.scan-ou-over  { background: linear-gradient(135deg, #7c4dff, #00e5ff);
                 display: flex; align-items: center; justify-content: center;
                 font-size: 0.84rem; font-weight: 900; color: #fff;
                 font-variant-numeric: tabular-nums; }
.scan-ou-under { background: linear-gradient(135deg, #1a1d27, #2d3148);
                 display: flex; align-items: center; justify-content: center;
                 font-size: 0.84rem; font-weight: 900; color: #cdd;
                 font-variant-numeric: tabular-nums; }
.scan-src-chip {
    font-size: 0.8rem; font-weight: 700; padding: 0.3rem 0.7rem;
    border-radius: 999px; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08); color: #8892a4;
    white-space: nowrap;
}
.scan-src-live {
    background: rgba(0,230,118,0.10); border-color: rgba(0,230,118,0.3);
    color: #00e676;
}

/* ── BIG, OBVIOUS value banner (above the bet expander) ── */
.scan-value-banner {
    display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
    padding: 1.1rem 1.5rem; margin: 0.6rem 0 0.3rem;
    background: linear-gradient(135deg, rgba(0,230,118,0.16), rgba(0,229,255,0.08));
    border: 1.5px solid rgba(0,230,118,0.45);
    border-left: 5px solid #00e676;
    border-radius: 14px;
    box-shadow: 0 6px 24px rgba(0,230,118,0.18);
    animation: fadeInUp 0.45s cubic-bezier(.22,.61,.36,1) both;
}
.scan-value-banner .svb-icon {
    font-size: 2.4rem; line-height: 1; flex-shrink: 0;
    filter: drop-shadow(0 3px 8px rgba(0,230,118,0.5));
}
.scan-value-banner .svb-block {
    display: flex; flex-direction: column; gap: 0.2rem; min-width: 110px;
}
.scan-value-banner .svb-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px;
    text-transform: uppercase; color: #8892a4;
}
.scan-value-banner .svb-val {
    font-size: 1.35rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; font-variant-numeric: tabular-nums;
    letter-spacing: -0.3px;
}
.scan-value-banner .svb-edge  { color: #00e676; }
.scan-value-banner .svb-stake { color: #ffd600; }
.scan-value-banner .svb-win   { color: #00e5ff; }

/* ── EV tags ── */
.ev-tag { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.5px; }
.ev-strong  { background: rgba(0,230,118,0.18); color: #00e676; }
.ev-mild    { background: rgba(105,240,174,0.15); color: #69f0ae; }
.ev-neutral { background: rgba(255,255,255,0.06); color: #b8c0d0; }
.ev-neg     { background: rgba(255,64,129,0.12); color: #ff4081; }

/* ── Last Gameweek result cards ── */
.res-card {
    border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 0.9rem;
    border-left: 4px solid; display: flex; flex-direction: column; gap: 0.65rem;
}
.res-correct { background: rgba(0,230,118,0.04); border-color: rgba(0,230,118,0.4); }
.res-wrong   { background: rgba(255,64,129,0.04); border-color: rgba(255,64,129,0.3); }
.res-teams   { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; }
.res-team    { font-size: 1.1rem; font-weight: 700; color: #e8eaf0; display: inline-flex; align-items: center; gap: 0.5rem; }
.res-score   { font-size: 2.4rem; font-weight: 900; color: #e8eaf0; letter-spacing: -2px;
               padding: 0 0.6rem;
               background: linear-gradient(135deg, #fff 0%, #aab 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text; }
.res-badge   { font-size: 0.78rem; font-weight: 800; padding: 4px 12px; border-radius: 20px;
               margin-left: auto; letter-spacing: 0.4px; }
.res-win  { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
.res-loss { background: rgba(255,64,129,0.15); color: #ff4081; border: 1px solid rgba(255,64,129,0.3); }

/* Predicted vs Actual headline row — bigger, clearer */
.res-headline {
    display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: center;
    margin-top: 0.2rem;
}
.res-headline-item {
    display: inline-flex; flex-direction: column; gap: 0.15rem;
}
.res-headline-label {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.8px;
    text-transform: uppercase; color: #b8c0d0;
}
.res-headline-value {
    font-size: 1.3rem; font-weight: 800; color: #e8eaf0;
    font-variant-numeric: tabular-nums; line-height: 1.1;
}

/* Goals-comparison chip — explicit Total Expected vs Total Actual */
.res-goals {
    display: inline-flex; align-items: center; gap: 0.7rem;
    padding: 0.45rem 0.95rem;
    background: linear-gradient(135deg, rgba(124,77,255,0.13), rgba(0,229,255,0.06));
    border: 1px solid rgba(124,77,255,0.3);
    border-radius: 12px;
    font-variant-numeric: tabular-nums;
}
.res-goals-block { display: inline-flex; flex-direction: column; gap: 0.1rem; }
.res-goals-lbl   { font-size: 0.78rem; letter-spacing: 1.4px; font-weight: 800;
                   color: #a78bfa; text-transform: uppercase; }
.res-goals-val   { font-size: 1.25rem; font-weight: 900; color: #e8eaf0; line-height: 1.05; }
.res-goals-vs    { font-size: 0.88rem; font-weight: 800; color: #b8c0d0; }

.res-meta  { font-size: 0.84rem; color: #b8c0d0; display: flex; gap: 1rem; flex-wrap: wrap;
             align-items: center; }
.res-meta b { color: #8892a4; }

/* Per-team xG chip in the meta row */
.res-xg-chip {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.4rem 0.95rem;
    background: linear-gradient(135deg, rgba(124,77,255,0.13), rgba(0,229,255,0.06));
    border: 1px solid rgba(124,77,255,0.3);
    border-radius: 12px;
    font-variant-numeric: tabular-nums;
    cursor: help;
}
.res-xg-chip .res-xg-lbl  { font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px;
                            color: #a78bfa; text-transform: uppercase; }
.res-xg-chip .res-xg-val  { font-size: 1.25rem; font-weight: 900; line-height: 1; }
.res-xg-chip .res-xg-h    { color: #949bf7; }
.res-xg-chip .res-xg-a    { color: #fb7185; }
.res-xg-chip .res-xg-dash { color: #b8c0d0; font-weight: 700; }
.res-prob-bar { display: flex; border-radius: 8px; overflow: hidden; height: 24px;
                margin-top: 0.3rem; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }

/* ── Summary stat chips ── */
.gw-stat { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
           border-radius: 14px; padding: 0.9rem 1rem; text-align: center; }

/* ── Team Deep Dive page ── */
.td-hero {
    display: flex; align-items: center; gap: 1.6rem;
    padding: 1.6rem 1.8rem;
    background: linear-gradient(135deg, rgba(124,77,255,0.10), rgba(0,229,255,0.04));
    border: 1px solid rgba(124,77,255,0.3);
    border-radius: 18px;
    margin: 0.8rem 0 1.2rem;
    animation: fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
}
.td-hero-badge img.team-badge {
    width: 130px !important; height: 130px !important;
    filter: drop-shadow(0 8px 24px rgba(0,0,0,0.5));
}
.td-hero-body { flex: 1; min-width: 0; }
.td-hero-name {
    font-size: 2.5rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; letter-spacing: -1px; margin-bottom: 0.85rem;
}
.td-hero-stats {
    display: flex; flex-wrap: wrap; gap: 1.4rem; align-items: center;
}
.td-stat {
    display: flex; flex-direction: column; gap: 0.15rem;
}
.td-stat-lbl {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #8892a4;
}
.td-stat-val {
    font-size: 1.7rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; letter-spacing: -0.5px;
    font-variant-numeric: tabular-nums;
}
.td-stat-spark { line-height: 0; margin-top: 0.15rem; }

/* Last-5 result chips */
.td-results-row {
    display: flex; gap: 0.7rem; flex-wrap: wrap;
    margin-bottom: 1.4rem;
}
.td-result-chip {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.7rem 0.95rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    flex: 1; min-width: 160px;
    transition: transform 0.18s ease, border-color 0.18s ease;
}
.td-result-chip:hover {
    transform: translateY(-2px);
    border-color: rgba(124,77,255,0.4);
}
.td-result-circle {
    width: 42px; height: 42px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    border: 2px solid;
    font-size: 1.05rem; font-weight: 900;
    flex-shrink: 0;
}
.td-result-meta { flex: 1; min-width: 0; }
.td-result-score {
    font-size: 1.1rem; font-weight: 900; color: #e8eaf0;
    font-variant-numeric: tabular-nums; letter-spacing: -0.3px;
}
.td-result-opp  { font-size: 0.84rem; color: #cdd; font-weight: 600;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.td-result-date { font-size: 0.78rem; color: #8892a4; }
.td-result-chip img.team-badge {
    width: 34px !important; height: 34px !important; flex-shrink: 0;
}

/* Upcoming fixture cards inside team deep-dive */
.td-fix-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
}
.td-fix-top {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.7rem;
}
.td-fix-date { font-size: 0.92rem; color: #8892a4; font-weight: 800; letter-spacing: 0.3px; }
.td-fix-verdict {
    font-size: 0.84rem; font-weight: 800; letter-spacing: 1.2px;
    padding: 0.3rem 0.8rem; border-radius: 999px;
    border: 1px solid; text-transform: uppercase;
}
.td-fix-row {
    display: grid; grid-template-columns: 1fr 2.4fr 1fr;
    gap: 1rem; align-items: center;
}

/* Portfolio summary stats row */
.td-portfolio-stats {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem; margin-bottom: 1.2rem;
}
.td-stat-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.td-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,77,255,0.10);
}
.td-stat-card .td-stat-lbl { color: #8892a4; }
.td-stat-card .td-stat-val { font-size: 1.85rem; margin-top: 0.3rem; }
.td-stat-card .td-stat-sub { font-size: 0.86rem; color: #8892a4; margin-top: 0.25rem;
                              line-height: 1.4; }

/* Per-bet rows */
.td-bet-table {
    display: flex; flex-direction: column; gap: 0.4rem;
    font-variant-numeric: tabular-nums;
}
.td-bet-row {
    display: grid;
    grid-template-columns: 90px 60px 1.4fr 1.5fr 70px 80px 100px 90px;
    gap: 0.7rem; align-items: center;
    padding: 0.7rem 1rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    font-size: 0.94rem;
    transition: transform 0.18s ease, border-color 0.18s ease;
}
.td-bet-row:hover {
    transform: translateX(2px);
    border-color: rgba(124,77,255,0.35);
}
.td-bet-row.td-bet-header {
    background: transparent; border: none; padding: 0.3rem 1rem 0.1rem;
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.4px;
    color: #a78bfa;
}
.td-bet-row.td-bet-header:hover { transform: none; }
.td-bet-date  { color: #cdd; font-weight: 700; }
.td-bet-port  { color: #a78bfa; font-weight: 800; font-size: 0.78rem;
                letter-spacing: 0.6px; text-transform: uppercase; }
.td-bet-match { color: #e8eaf0; font-weight: 700; }
.td-bet-sel   { color: #ffd600; font-weight: 700; }
.td-bet-odds  { color: #00e5ff; font-weight: 900; font-size: 1.05rem;
                text-align: center; }
.td-bet-stake { color: #e8eaf0; font-weight: 800; text-align: right; }
.td-bet-pl    { font-weight: 900; text-align: right;
                font-size: 1rem; letter-spacing: -0.2px; }
.td-bet-status {
    display: inline-block; padding: 0.25rem 0.6rem;
    font-size: 0.78rem; font-weight: 800; border-radius: 999px;
    letter-spacing: 0.3px;
}
.td-bet-status.td-won  { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.35); }
.td-bet-status.td-lost { background: rgba(255,64,129,0.15); color: #ff4081; border: 1px solid rgba(255,64,129,0.35); }
.td-bet-status.td-pend { background: rgba(255,214,0,0.10);  color: #ffd600; border: 1px solid rgba(255,214,0,0.35); }

@media (max-width: 1100px) {
    .td-bet-row { grid-template-columns: repeat(2, 1fr); padding: 0.7rem; }
    .td-bet-row.td-bet-header { display: none; }
    .td-portfolio-stats { grid-template-columns: repeat(2, 1fr); }
    .td-fix-row { grid-template-columns: 1fr; gap: 0.5rem; }
}
@media (max-width: 700px) {
    .td-hero { flex-direction: column; text-align: center; }
    .td-hero-stats { justify-content: center; }
    .td-portfolio-stats { grid-template-columns: 1fr; }
    .td-results-row > .td-result-chip { flex: 1 1 100%; }
}

/* Gameweek recap header — 4-block story summary above the per-match cards */
.gw-recap {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin: 0.6rem 0 1.4rem;
    animation: fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
}
.gw-recap-block {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    display: flex; flex-direction: column; gap: 0.3rem;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.gw-recap-block:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,77,255,0.15);
}
.gw-recap-icon { font-size: 1.6rem; line-height: 1; }
.gw-recap-lbl  {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px;
    text-transform: uppercase; color: #8892a4;
}
.gw-recap-val  {
    font-size: 1.7rem; font-weight: 900; color: #e8eaf0;
    line-height: 1.05; letter-spacing: -0.5px; font-variant-numeric: tabular-nums;
    display: flex; align-items: center; gap: 0.3rem;
}
.gw-recap-sub  {
    font-size: 0.92rem; color: #cdd; line-height: 1.45; margin-top: 0.2rem;
}

@media (max-width: 1100px) {
    .gw-recap { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .gw-recap { grid-template-columns: 1fr; }
}
.gw-stat-val { font-size: 1.85rem; font-weight: 900; color: #e8eaf0; line-height: 1.05; }
.gw-stat-lbl { font-size: 0.78rem; font-weight: 800; letter-spacing: 1.6px; text-transform: uppercase; color: #b8c0d0; margin-top: 0.4rem; }

.scan-value-alert {
    background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.2);
    border-radius: 10px; padding: 0.65rem 1rem; margin-bottom: 0.6rem;
    font-size: 0.82rem; color: #aab;
}

/* ── Mobile pass — phone-width breakpoints across heavy panels ── */
@media (max-width: 900px) {
    /* Hero strip (live stats banner) — stack 3 columns to 1 */
    .hero-strip { grid-template-columns: 1fr; gap: 0.6rem; margin: 0.8rem 0 1.1rem; }
    .hero-stat { padding: 0.8rem 1rem; }
    .hero-stat-value { font-size: 1.4rem; }

    /* Fixture cards (Weekend tab) — tighten columns + padding */
    .fixture-card { padding: 1.1rem 1.2rem; }
    .fixture-teams { grid-template-columns: 1fr 1.6fr 1fr; gap: 0.6rem; }
    .fixture-team-name { font-size: 0.96rem; gap: 0.35rem; }

    /* Scanner fixture cards — same pattern */
    .scan-fixture-card { padding: 1rem 1.1rem; }
    .scan-fixture-row { grid-template-columns: 1fr 1.6fr 1fr; gap: 0.7rem; }
    .scan-fixture-row .fixture-team-name { font-size: 0.98rem; }

    /* Pending bet card v2 — smaller badges/typography */
    .pend-v2-team img { width: 48px; height: 48px; }
    .pend-v2-team span { font-size: 0.88rem; max-width: 110px; }
    .pend-v2-pick { font-size: 1.15rem; }
    .pend-v2-stat-val { font-size: 1rem; }

    /* PnL hero — slightly less padding */
    .pnl-hero { padding: 1.9rem 1.3rem; }

    /* Predict-tab insight cards */
    .insight-card { padding: 1.1rem 1.2rem; gap: 0.85rem; }
    .insight-head { font-size: 1.05rem; }
    .insight-sub  { font-size: 0.88rem; }
}

@media (max-width: 700px) {
    /* Fixture cards — stack name above bar */
    .fixture-teams { grid-template-columns: 1fr; gap: 0.55rem; text-align: center; }
    .fixture-team-home, .fixture-team-away { justify-content: center; text-align: center; }
    .fixture-prob-bar { height: 32px; }
    .scan-fixture-row { grid-template-columns: 1fr; gap: 0.55rem; text-align: center; }
    .scan-fixture-row .fixture-team-home,
    .scan-fixture-row .fixture-team-away { justify-content: center; text-align: center; }
    /* O/U row label can shorten */
    .scan-ou-lbl { min-width: 0; }
    .scan-ou-row { flex-wrap: wrap; gap: 0.6rem; }

    /* Pending v2 — tighter stat grid */
    .pend-v2-stats { gap: 0.4rem; }
    .pend-v2-stat { padding: 0.45rem 0.5rem; }
    .pend-v2-team img { width: 40px; height: 40px; }
    .pend-v2-vs { font-size: 1.1rem; }
    .pend-v2-return { padding: 0.7rem 0.9rem; gap: 0.7rem; }
    .pend-v2-return-val, .pend-v2-return-profit { font-size: 1.2rem; }

    /* Bet history rows — tighten 2-col stack */
    .bh-row { grid-template-columns: repeat(2, 1fr); gap: 0.4rem; padding: 0.7rem; }

    /* Sim table — collapse to single column on phone */
    .sim-row { grid-template-columns: 1fr; gap: 0.45rem; padding: 0.7rem 0.85rem; }
    .sim-team { grid-column: auto; }
    .sim-rank { font-size: 1.2rem; }

    /* PnL hero — shrink amount letter-spacing for narrow screens */
    .pnl-hero { padding: 1.6rem 1rem; }
    .pnl-amount { letter-spacing: -1.5px; }

    /* Insight cards — even tighter */
    .insight-card { padding: 1rem 1.1rem; gap: 0.7rem; }
    .insight-emoji { font-size: 1.6rem; }
    .insight-head { font-size: 0.98rem; }
    .insight-sub  { font-size: 0.85rem; }

    /* Section headers don't need huge top margin on phone */
    .stMarkdown h2 { font-size: 1.3rem; }
    .stMarkdown h3 { font-size: 1.1rem; }
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


@st.cache_resource(show_spinner=False)
def cached_promoted_seeding(df_hash: str, fixture_key: str, _dc_r, _dc_draw_r):
    """Seed promoted sides into the fitted ratings.

    `predict_dixon_coles` falls back to 0.0 for an unknown team, and 0.0 is
    league average rather than "unknown" — so a promoted side is silently
    modelled as a mid-table club. On the real 2026-27 opener that inflated the
    Arsenal v Coventry draw from 13.3% to 31.7%, against a market-implied 10.9%.

    Kept out of `cached_models` on purpose: the fixture list comes from a
    separate cached function, and nesting one Streamlit cache inside another
    couples two independent TTLs and their locks.

    Betting on seeded sides stays blocked by the no-history gate. This is so
    predictions read honestly, not so they can be staked.
    """
    teams = [t for pair in fixture_key.split("|") if pair for t in pair.split("~")]
    if not teams:
        return _dc_r, _dc_draw_r
    return (seed_promoted_teams(_dc_r, teams),
            seed_promoted_teams(_dc_draw_r, teams))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fixtures():
    return fetch_upcoming_fixtures(lookahead_days=30)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_top_weekend_pick(dc_hash: str):
    """Find the highest-confidence upcoming fixture in the next 7 days.

    Uses Dixon-Coles only (cheap — no XGB) so the hero strip loads fast.
    Returns a dict: {home, away, pick, prob, date} or None.
    """
    try:
        fixtures = fetch_upcoming_fixtures(lookahead_days=7)
    except Exception:
        return None
    if not fixtures:
        return None
    dc_r = st.session_state.get("_hero_dc_r")
    if dc_r is None:
        return None
    best = None
    for fix in fixtures:
        h, a = fix.get("home"), fix.get("away")
        if not h or not a:
            continue
        if h not in dc_r["team_idx"] or a not in dc_r["team_idx"]:
            continue
        try:
            pred = predict_dixon_coles(h, a, dc_r)
        except Exception:
            continue
        probs = [("home_win", pred["home_win"], h),
                 ("draw",     pred["draw"],     "Draw"),
                 ("away_win", pred["away_win"], a)]
        market, prob, label = max(probs, key=lambda x: x[1])
        if best is None or prob > best["prob"]:
            best = {
                "home": h, "away": a,
                "pick_market": market, "pick_label": label,
                "prob": prob,
                "date": fix.get("date"),
            }
    return best


def render_hero_strip(dc_r: dict) -> None:
    """Live stat banner above the tabs: portfolio + weekend top pick."""
    # Cache dc_r in session so cached_top_weekend_pick can see it
    st.session_state["_hero_dc_r"] = dc_r

    # ── Portfolio snapshot ─────────────────────────────────────────────
    try:
        port = pf.load_portfolio()
        br  = port.get("bankroll", 0.0)
        init = port.get("initial_bankroll", 10_000.0)
        pending = [b for b in port.get("bets", []) if b.get("status") == "pending"]
        delta = br - init
        roi   = (delta / init * 100.0) if init else 0.0
    except Exception:
        br, init, pending, delta, roi = 10_000.0, 10_000.0, [], 0.0, 0.0

    delta_sign  = "+" if delta >= 0 else "−"
    delta_color = "#00e676" if delta >= 0 else "#ff4081"
    roi_color   = delta_color

    # ── Top weekend pick ───────────────────────────────────────────────
    try:
        top = cached_top_weekend_pick(str(dc_r.get("home_adv", ""))[:8])
    except Exception:
        top = None

    if top:
        pick_color = {"home_win": "#3d6eff", "draw": "#ffd600",
                      "away_win": "#ff4081"}.get(top["pick_market"], "#7c4dff")
        pick_html = (
            f'<div class="hero-stat">'
            f'  <div class="hero-stat-label">🔥 TOP WEEKEND PICK</div>'
            f'  <div class="hero-stat-value" style="color:{pick_color};font-size:1.05rem">'
            f'    {tb(top["home"], 20)} <span style="color:#b8c0d0">vs</span> {tb(top["away"], 20)}'
            f'  </div>'
            f'  <div class="hero-stat-sub">'
            f'    Model backs <b style="color:{pick_color}">{top["pick_label"]}</b> '
            f'    at <b style="color:#e8eaf0">{top["prob"]*100:.0f}%</b> confidence'
            f'  </div>'
            f'</div>'
        )
    else:
        pick_html = (
            '<div class="hero-stat">'
            '  <div class="hero-stat-label">🔥 TOP WEEKEND PICK</div>'
            '  <div class="hero-stat-value" style="color:#b8c0d0;font-size:0.95rem">No fixtures found</div>'
            '  <div class="hero-stat-sub">Try again closer to matchday</div>'
            '</div>'
        )

    st.markdown(f"""
    <div class="hero-strip">
        <div class="hero-stat">
            <div class="hero-stat-label">💰 BANKROLL</div>
            <div class="hero-stat-value">£{br:,.0f}</div>
            <div class="hero-stat-sub" style="color:{delta_color}">
                {delta_sign}£{abs(delta):,.0f} &nbsp;·&nbsp; ROI {delta_sign}{abs(roi):.1f}%
            </div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-label">⏳ PENDING BETS</div>
            <div class="hero-stat-value">{len(pending)}</div>
            <div class="hero-stat-sub">
                £{sum(b.get('stake',0) for b in pending):,.0f} at stake
            </div>
        </div>
        {pick_html}
    </div>
    """, unsafe_allow_html=True)


def _prediction_insights(hs: dict, as_: dict, home_team: str, away_team: str) -> list[tuple[str, str, str]]:
    """Return up to 3 most differentiating factors driving the prediction.

    Each item: (emoji, headline HTML, sub-note).
    """
    insights = []

    # xG attack differential
    if "avg_xg" in hs and "avg_xg" in as_:
        hx, ax = hs["avg_xg"], as_["avg_xg"]
        diff = hx - ax
        leader = home_team if hx > ax else away_team
        leader_color = "#3d6eff" if hx > ax else "#ff4081"
        insights.append((abs(diff), "⚡",
            f'<b style="color:{leader_color}">{leader}</b> averaging '
            f'<b style="color:#e8eaf0">{max(hx,ax):.2f} xG</b> vs '
            f'<b style="color:#e8eaf0">{min(hx,ax):.2f}</b> over last 6 matches',
            f'xG advantage of {abs(diff):.2f} goals per game'))

    # xG defensive differential
    if "avg_xga" in hs and "avg_xga" in as_:
        hx, ax = hs["avg_xga"], as_["avg_xga"]
        diff = ax - hx  # lower xGA is better, so home leads if hx < ax
        leader = home_team if hx < ax else away_team
        leader_color = "#3d6eff" if hx < ax else "#ff4081"
        insights.append((abs(diff), "🛡️",
            f'<b style="color:{leader_color}">{leader}</b> defence conceding just '
            f'<b style="color:#e8eaf0">{min(hx,ax):.2f} xGA</b> vs '
            f'<b style="color:#e8eaf0">{max(hx,ax):.2f}</b>',
            f'{abs(diff):.2f} fewer quality chances conceded per game'))

    # Form (pts / game)
    if "avg_pts" in hs and "avg_pts" in as_:
        hp, ap = hs["avg_pts"], as_["avg_pts"]
        diff = hp - ap
        leader = home_team if hp > ap else away_team
        leader_color = "#3d6eff" if hp > ap else "#ff4081"
        insights.append((abs(diff) * 1.5, "📈",
            f'<b style="color:{leader_color}">{leader}</b> earning '
            f'<b style="color:#e8eaf0">{max(hp,ap):.1f} pts/game</b> vs '
            f'<b style="color:#e8eaf0">{min(hp,ap):.1f}</b> recently',
            f'Form advantage of {abs(diff):.2f} pts/game'))

    # Venue-specific: home at home vs away away
    if "home_venue_pts" in hs and "away_venue_pts" in as_:
        h_home = hs["home_venue_pts"]
        a_away = as_["away_venue_pts"]
        diff = h_home - a_away
        insights.append((abs(diff) * 1.2, "🏟️",
            f'<b style="color:#7fa3ff">{home_team}</b> averaging '
            f'<b style="color:#e8eaf0">{h_home:.1f} pts at home</b> · '
            f'<b style="color:#ff4081">{away_team}</b> '
            f'<b style="color:#e8eaf0">{a_away:.1f} pts away</b>',
            f'Venue-specific gap of {abs(diff):.2f} pts/game'))

    # Elo gap
    if "elo" in hs and "elo" in as_:
        he, ae = hs["elo"], as_["elo"]
        diff = he - ae
        if abs(diff) > 20:  # only show if the gap is meaningful
            leader = home_team if he > ae else away_team
            leader_color = "#3d6eff" if he > ae else "#ff4081"
            insights.append((abs(diff) / 30.0, "📊",
                f'<b style="color:{leader_color}">{leader}</b> carries a '
                f'<b style="color:#e8eaf0">{abs(diff):.0f}-point Elo advantage</b> '
                f'({he:.0f} vs {ae:.0f})',
                'Historical-strength edge based on results quality'))

    # Sort by differentiator magnitude, take top 3
    insights.sort(key=lambda x: x[0], reverse=True)
    return [(emoji, headline, sub) for _, emoji, headline, sub in insights[:3]]


@st.cache_data(show_spinner=False)
def cached_backtest(df_hash: int, test_weeks: int):
    df, df_features = cached_data()
    bt = backtest_models(df, df_features, test_weeks=test_weeks)
    summary = compute_backtest_summary(bt)
    return bt, summary


@st.cache_resource(show_spinner=False)
def cached_calibrators(df_hash: int):
    """Fit per-market isotonic calibrators from a full season of backtest.

    LIVE-BETTING USE ONLY. The window is genuinely in the past relative to any
    new live bet, so this is leak-free for live sizing — but it must NOT be
    handed to a backtest evaluating an overlapping window. Backtests use
    `cached_honest_calibrators` instead.

    Forty weeks, not ten. Ten weeks is 80 matches pre-season, and isotonic on 80
    points is a three-step staircase: on 2026-08-09 it mapped six of nine
    opening fixtures to exactly 31.2% and sent a raw 17.3% to the 0.005 floor,
    leaving one bet on the board. `MIN_CALIBRATION_SAMPLES` now refuses such a
    fit outright, so a short window here means no calibration at all.
    """
    bt, _ = cached_backtest(df_hash, test_weeks=40)
    return pf.fit_calibrators_from_backtest(bt)


@st.cache_resource(show_spinner=False)
def cached_honest_calibrators(df_hash: int, exclude_weeks: int):
    """Leak-free calibrators for backtesting: fit on data strictly BEFORE the
    eval window.

    Truncates the dataset to exclude the most recent `exclude_weeks` weeks
    (the backtest's eval window), then fits the isotonic calibrators from a
    10-week backtest on what remains — same honest pattern as
    scripts/honest_random_search.py `_build_honest_calibrator`. Returns {} if
    there isn't enough prior data to fit.
    """
    df, df_features = cached_data()
    cutoff = df["Date"].max() - pd.Timedelta(weeks=int(exclude_weeks))
    pre = df[df["Date"] <= cutoff].copy()
    pre_ft = df_features[df_features["Date"] <= cutoff].copy()
    if len(pre) < 200:
        return {}
    try:
        bt_pre = backtest_models(pre, pre_ft, test_weeks=10)
        return pf.fit_calibrators_from_backtest(bt_pre)
    except Exception:
        return {}


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


def full_predict_v2(home, away, dc_kn_r, dc_draw_r, xgb_m, feat_cols,
                    draw_xgb_m, draw_fc, hs, as_):
    """Research-track prediction (Mock Portfolio Two): swaps in Dixon-Coles
    with Karlis-Ntzoufras diagonal inflation. XGB blend + draw specialist
    weighting are otherwise unchanged so the only difference vs `full_predict`
    is the underlying DC variant.
    """
    kn_pred  = predict_dixon_coles_kn(home, away, dc_kn_r)
    xgb_p    = predict_xgb(xgb_m, feat_cols, hs, as_)
    kn_blend = blend_dc(kn_pred, xgb_p)
    draw_xgb_prob = predict_draw_xgb(draw_xgb_m, draw_fc, hs, as_)
    final = blend_draw_specialist(kn_blend, dc_draw_r, draw_xgb_prob, home, away)
    return kn_pred, kn_blend, final


@st.cache_resource(show_spinner=False)
def cached_dc_kn(df_hash: str):
    """Fit and cache Dixon-Coles + K-N diagonal inflation ratings.
    Separate cache from the main DC fit so Mock Two's model recomputes
    independently when needed.
    """
    df, _ = cached_data()
    return compute_dixon_coles_kn_ratings(df)


@st.cache_resource(show_spinner=False)
def cached_bin_variances(df_hash: int, test_weeks: int = 10):
    """Per-bin Var(p̂) for each market, computed from the most recent backtest.
    Used by Mock Two's uncertainty-shrunk Kelly to size stakes by calibration
    confidence in each probability bucket.

    LIVE-BETTING USE ONLY — same leak rule as `cached_calibrators`. Backtests
    must use `cached_honest_bin_variances` so the variance estimates never see
    the eval window's outcomes.
    """
    bt, _ = cached_backtest(df_hash, test_weeks)
    return {
        "H":       pf.compute_per_bin_variance(bt, "H", n_bins=8),
        "D":       pf.compute_per_bin_variance(bt, "D", n_bins=8),
        "A":       pf.compute_per_bin_variance(bt, "A", n_bins=8),
    }


@st.cache_resource(show_spinner=False)
def cached_honest_bin_variances(df_hash: int, exclude_weeks: int):
    """Leak-free per-bin Var(p̂) for backtesting: computed from a backtest on
    data strictly BEFORE the eval window (mirrors `cached_honest_calibrators`).
    Returns {} when there isn't enough prior data — the v2 simulator then
    falls back to its fixed moderate-variance default rather than leaking
    (passing None would trigger its in-sample fit on the eval window).
    """
    df, df_features = cached_data()
    cutoff = df["Date"].max() - pd.Timedelta(weeks=int(exclude_weeks))
    pre = df[df["Date"] <= cutoff].copy()
    pre_ft = df_features[df_features["Date"] <= cutoff].copy()
    if len(pre) < 200:
        return {}
    try:
        bt_pre = backtest_models(pre, pre_ft, test_weeks=10)
        return {
            "H": pf.compute_per_bin_variance(bt_pre, "H", n_bins=8),
            "D": pf.compute_per_bin_variance(bt_pre, "D", n_bins=8),
            "A": pf.compute_per_bin_variance(bt_pre, "A", n_bins=8),
        }
    except Exception:
        return {}


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


def win_prob_bar_html(final, home_team, away_team) -> str:
    """HTML stacked probability bar — badges + bold % text in vibrant gradient segments.

    Replaces the Plotly version: bigger, bolder, badge-led, smoother animation.
    """
    h = round(final["home_win"] * 100, 1)
    d = round(final["draw"]     * 100, 1)
    a = round(final["away_win"] * 100, 1)
    h_url = _BADGE_URL.get(home_team, "")
    a_url = _BADGE_URL.get(away_team, "")

    # Bigger badges (64px) need a bit more segment width to fit cleanly
    h_badge = (f'<img class="team-badge" src="{h_url}" alt="{home_team}" />' if (h_url and h >= 12) else "")
    a_badge = (f'<img class="team-badge" src="{a_url}" alt="{away_team}" />' if (a_url and a >= 12) else "")

    return (
        f'<div class="prob-stack-bar">'
        f'  <div class="prob-stack-seg seg-home" style="flex:{h}">'
        f'    {h_badge}<span class="seg-pct">{h}%</span>'
        f'  </div>'
        f'  <div class="prob-stack-seg seg-draw" style="flex:{d}">'
        f'    <span class="seg-pct seg-pct-dark">DRAW</span>'
        f'    <span class="seg-pct seg-pct-dark">{d}%</span>'
        f'  </div>'
        f'  <div class="prob-stack-seg seg-away" style="flex:{a}">'
        f'    <span class="seg-pct">{a}%</span>{a_badge}'
        f'  </div>'
        f'</div>'
    )


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


def _compute_edge_badge(result: dict, odds: dict | None) -> str:
    """Return an HTML edge badge showing best model-vs-market value, or ''."""
    if not odds:
        return ""
    try:
        oh = float(odds.get("H", 0.0))
        od = float(odds.get("D", 0.0))
        oa = float(odds.get("A", 0.0))
    except (TypeError, ValueError):
        return ""
    if min(oh, od, oa) <= 1.0:
        return ""

    # De-vig implied probabilities
    imp_h, imp_d, imp_a = 1.0 / oh, 1.0 / od, 1.0 / oa
    margin = imp_h + imp_d + imp_a
    if margin <= 0:
        return ""
    mkt_h, mkt_d, mkt_a = imp_h / margin, imp_d / margin, imp_a / margin

    # Edge per market
    candidates = [
        ("Home Win", result["home_win"] - mkt_h, result["home_win"], mkt_h, "#3d6eff"),
        ("Draw",     result["draw"]     - mkt_d, result["draw"],     mkt_d, "#ffd600"),
        ("Away Win", result["away_win"] - mkt_a, result["away_win"], mkt_a, "#ff4081"),
    ]
    label, edge, model_p, mkt_p, color = max(candidates, key=lambda x: x[1])

    if edge <= 0.01:   # under 1% not worth highlighting
        return ""

    cls = "edge-badge" if edge > 0 else "edge-badge edge-neg"
    sign = "+" if edge > 0 else ""
    return (
        f'<span class="{cls}" title="Model {model_p*100:.0f}% · Market {mkt_p*100:.0f}%">'
        f'💎 {label} &nbsp;{sign}{edge*100:.1f}% edge'
        f'</span>'
    )


def fixture_card_html(home_team, away_team, result, lam_h, lam_a, odds: dict | None = None):
    """Render a fixture card — informational view (no betting edge).

    The Weekend tab is for "what does the model think will happen this week",
    distinct from the Mock Portfolio's betting flow. So we skip the edge badge
    (that lives in the portfolio scanner) and keep this card clean and visual.
    `odds` parameter retained for backwards compatibility but ignored here.
    """
    h_pct = round(result["home_win"] * 100, 1)
    d_pct = round(result["draw"]     * 100, 1)
    a_pct = round(result["away_win"] * 100, 1)
    return f"""
    <div class="fixture-card">
        <div class="fixture-teams">
            <div class="fixture-team-name fixture-team-home">{tb(home_team, 44)}</div>
            <div class="fixture-prob-bar">
                <div class="bar-home" style="width:{h_pct}%">{h_pct}%</div>
                <div class="bar-draw" style="width:{d_pct}%">{d_pct}%</div>
                <div class="bar-away" style="width:{a_pct}%">{a_pct}%</div>
            </div>
            <div class="fixture-team-name fixture-team-away">{tb(away_team, 44)}</div>
        </div>
        <div class="fixture-footer">
            <span class="fixture-xg-chip"
                  title="Expected Goals — the model's predicted goals for each team based on Dixon-Coles attack/defence ratings. {home_team}: {lam_h:.2f} · {away_team}: {lam_a:.2f}">
                <span class="xg-label">xG</span>
                <span class="xg-home">{lam_h:.1f}</span>
                <span class="xg-dash">–</span>
                <span class="xg-away">{lam_a:.1f}</span>
            </span>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Match Predictor
# ─────────────────────────────────────────────────────────────────────────────
def _big_badge_card(team: str, label: str, accent: str) -> str:
    """Big-badge card for the team picker. 200px badge + team name + corner label."""
    url = _BADGE_URL.get(team, "")
    badge_html = (f'<img class="team-badge" width="200" src="{url}" alt="{team}" '
                  f'style="width:200px;height:200px;object-fit:contain;'
                  f'filter:drop-shadow(0 8px 24px rgba(0,0,0,0.5));'
                  f'transition:transform 0.25s ease">'
                  if url else
                  '<div style="width:200px;height:200px;display:flex;align-items:center;'
                  'justify-content:center;color:#b8c0d0;font-size:0.9rem">no badge</div>')
    return (
        f'<div style="position:relative;text-align:center;padding:1.2rem 0.5rem 0.8rem;'
        f'background:linear-gradient(180deg,rgba({accent},0.06),transparent);'
        f'border-radius:14px;'
        f'border:1px solid rgba({accent},0.18)">'
        f'<div style="position:absolute;top:0.6rem;left:0.9rem;'
        f'font-size:0.8rem;letter-spacing:2.5px;color:rgb({accent});'
        f'font-weight:700;text-transform:uppercase">{label}</div>'
        f'{badge_html}'
        f'<div style="font-size:1.2rem;font-weight:700;color:#e8eaf0;'
        f'margin-top:0.6rem;letter-spacing:-0.2px">{team}</div>'
        f'</div>'
    )


def tab_predict(df, df_features, poisson_r, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict):
    # ── Big-badge team picker ──────────────────────────────────────────
    col_l, col_vs, col_r = st.columns([5, 1, 5])

    # Read current selections first so the badges above match what's selected
    _default_home = "Arsenal" if "Arsenal" in teams else teams[0]
    _default_away = "Chelsea" if "Chelsea" in teams else (teams[1] if len(teams) > 1 else teams[0])
    _home_pick = st.session_state.get("pred_home", _default_home)
    _away_pick = st.session_state.get("pred_away", _default_away)
    if _home_pick not in teams: _home_pick = _default_home
    if _away_pick not in teams: _away_pick = _default_away

    with col_l:
        st.markdown(_big_badge_card(_home_pick, "🏠  HOME", "61, 110, 255"),
                    unsafe_allow_html=True)
    with col_vs:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:280px"><div class="vs-badge" style="font-size:1.6rem">VS</div></div>',
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(_big_badge_card(_away_pick, "✈️  AWAY", "255, 64, 129"),
                    unsafe_allow_html=True)

    # ── Dropdowns underneath — secondary "change team" controls ─────────
    sl_l, _, sl_r = st.columns([5, 1, 5])
    with sl_l:
        home_team = st.selectbox("Change home team", teams,
            index=teams.index(_home_pick),
            key="pred_home", label_visibility="collapsed")
    with sl_r:
        away_team = st.selectbox("Change away team", teams,
            index=teams.index(_away_pick),
            key="pred_away", label_visibility="collapsed")

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

    # Pull accuracy scores from the cached 10-week walk-forward backtest
    try:
        _, _bt_summary = cached_backtest(len(df), test_weeks=10)
    except Exception:
        _bt_summary = {}
    _acc_pois = _bt_summary.get("pois_accuracy")
    _acc_dcb  = _bt_summary.get("dcb_accuracy")
    _acc_full = _bt_summary.get("dc_accuracy")

    def _acc_badge(acc: float | None) -> str:
        if acc is None:
            return ('<span class="model-acc-badge model-acc-na">'
                    'accuracy n/a</span>')
        col = "#00e676" if acc >= 55 else ("#ffd600" if acc >= 50 else "#ff4081")
        return (f'<span class="model-acc-badge" style="color:{col};'
                f'border-color:{col}55;background:rgba({_hex_to_rgb(col)},0.10)">'
                f'<span class="model-acc-num">{acc:.1f}%</span>'
                f'<span class="model-acc-lbl">10-wk accuracy</span></span>')

    cA, cB, cC = st.columns(3)
    for col, title, result, tag_color, acc in [
        (cA, "Poisson + XGB",                p_blend,  "#7c4dff", _acc_pois),
        (cB, "DC + XGB",                     dc_blend, "#00e5ff", _acc_dcb),
        (cC, "DC + XGB + Draw Specialist",   final,    "#00e676", _acc_full),
    ]:
        with col:
            h_p = round(result["home_win"] * 100, 1)
            d_p = round(result["draw"]     * 100, 1)
            a_p = round(result["away_win"] * 100, 1)
            st.markdown(f"""
            <div class="model-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.6rem;margin-bottom:0.6rem">
                    <div class="model-title" style="color:{tag_color};margin:0">{title}</div>
                    {_acc_badge(acc)}
                </div>
                <div class="fixture-prob-bar" style="height:34px;margin-bottom:0.6rem">
                    <div class="bar-home bar-sm" style="width:{h_p}%">{h_p}%</div>
                    <div class="bar-draw bar-sm" style="width:{d_p}%">{d_p}%</div>
                    <div class="bar-away bar-sm" style="width:{a_p}%">{a_p}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Use DC+XGB+Draw Specialist as the primary for the detailed breakdown below
    # (final already set by full_predict above)
    dc_lam_h = dc_pred["lambda_home"]
    dc_lam_a = dc_pred["lambda_away"]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Stacked probability bar — primary outcome view (full ensemble) ───
    st.markdown('<p class="section-label" style="text-align:center">Dixon-Coles · Overall Probability</p>',
                unsafe_allow_html=True)
    st.markdown(win_prob_bar_html(final, home_team, away_team),
                unsafe_allow_html=True)

    # ── Why this prediction — key drivers from rolling stats ──────────────
    insights = _prediction_insights(hs, as_, home_team, away_team)
    if insights:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label" style="text-align:center;font-size:1rem">🧠  Why this prediction</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.95rem;color:#8892a4;text-align:center;margin-bottom:1.2rem">'
            'Top factors driving the model — ranked by how much they favour one side.</p>',
            unsafe_allow_html=True,
        )
        for emoji, headline, sub in insights:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-emoji">{emoji}</div>
                <div class="insight-body">
                    <div class="insight-head">{headline}</div>
                    <div class="insight-sub">{sub}</div>
                </div>
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
                <div style="font-size:0.88rem;color:#b8c0d0;text-transform:uppercase;letter-spacing:2px;
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
                <td style="color:#b8c0d0">{row['Date'].strftime('%d %b %Y')}</td>
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

    # Optional live odds for edge badges (uses portfolio's API key + 6h cache)
    live_odds_map: dict = {}
    try:
        port_settings = pf.load_portfolio().get("settings", {})
        api_key = pf.resolve_odds_api_key(port_settings.get("odds_api_key", ""))
        if api_key:
            live_odds_map = pf.fetch_live_odds(api_key)
    except Exception:
        live_odds_map = {}

    if not fixtures:
        if _render_offseason_card(df, "Fixtures will appear here once the new "
                                  "season's schedule is released.", "weekend"):
            return
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
            f'<p style="font-size:0.82rem;color:#b8c0d0;margin-bottom:0.3rem;">'
            f'Next gameweek · <b style="color:#8892a4">{date_range_str}</b>'
            f' · {len(fixtures)} matches</p>',
            unsafe_allow_html=True,
        )
        # Legend chips — vibrant gradients matching the bar colours, larger for visibility
        st.markdown(
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;align-items:center;'
            'margin-bottom:1.4rem;font-size:0.84rem">'
            '<span style="display:inline-flex;align-items:center;gap:0.45rem;'
            'padding:0.32rem 0.85rem;border-radius:999px;'
            'background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.4);'
            'color:#a5b4fc;font-weight:800;letter-spacing:0.3px">'
            '<span style="display:inline-block;width:14px;height:14px;border-radius:4px;'
            'background:linear-gradient(135deg,#4f46e5,#22d3ee)"></span>Home Win</span>'

            '<span style="display:inline-flex;align-items:center;gap:0.45rem;'
            'padding:0.32rem 0.85rem;border-radius:999px;'
            'background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.4);'
            'color:#fcd34d;font-weight:800;letter-spacing:0.3px">'
            '<span style="display:inline-block;width:14px;height:14px;border-radius:4px;'
            'background:linear-gradient(135deg,#f59e0b,#fbbf24)"></span>Draw</span>'

            '<span style="display:inline-flex;align-items:center;gap:0.45rem;'
            'padding:0.32rem 0.85rem;border-radius:999px;'
            'background:rgba(225,29,72,0.12);border:1px solid rgba(225,29,72,0.4);'
            'color:#fda4af;font-weight:800;letter-spacing:0.3px">'
            '<span style="display:inline-block;width:14px;height:14px;border-radius:4px;'
            'background:linear-gradient(135deg,#fb7185,#9f1239)"></span>Away Win</span>'
            '</div>',
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
                    f'<p style="font-size:0.78rem;font-weight:700;color:#b8c0d0;'
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

            fix_odds = live_odds_map.get((home, away))
            st.markdown(
                fixture_card_html(home, away, result,
                                  dc_pred["lambda_home"], dc_pred["lambda_away"],
                                  odds=fix_odds),
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
            st.markdown('<div style="text-align:center;padding-top:0.5rem;font-weight:800;color:#b8c0d0;font-size:1.2rem">vs</div>',
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
        '<p style="font-size:0.8rem;color:#b8c0d0;margin-bottom:1.5rem;">'
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
        '<p style="font-size:0.78rem;color:#b8c0d0;text-align:center;">'
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
            <td style="color:#b8c0d0;white-space:nowrap">{row['Date'].strftime('%d %b')}</td>
            <td style="font-weight:700">{row['Home']}</td>
            <td style="text-align:center;font-weight:900;color:#e8eaf0">{row['Score']}</td>
            <td style="font-weight:700">{row['Away']}</td>
            <td style="color:#aab">{row['Actual']}</td>
            <td>
                <span style="color:#b8c0d0;font-size:0.78rem">{pos_str}</span><br>
                <span class="{t_cls}">{t_icon} {row['Table_Pred']}</span>
            </td>
            <td>
                <span style="color:#b8c0d0;font-size:0.82rem">{row['Pois_H']}·{row['Pois_D']}·{row['Pois_A']}</span><br>
                <span class="{p_cls}">{p_icon} {row['Pois_Pred']}</span>
            </td>
            <td>
                <span style="color:#b8c0d0;font-size:0.82rem">{row['DC_H']}·{row['DC_D']}·{row['DC_A']}</span><br>
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
        '<p style="font-size:0.8rem;color:#b8c0d0;margin-bottom:1.2rem">'
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

    # ── Calibration metrics (ECE + Brier per model) ───────────────────────
    def _ece(sub: pd.DataFrame) -> float:
        """Expected Calibration Error — sample-weighted |pred − actual| across bins."""
        grp = (sub.groupby("bin", observed=True)
                  .agg(mean_pred=("pred", "mean"),
                       mean_act=("actual", "mean"),
                       count=("actual", "count")))
        if grp["count"].sum() == 0:
            return float("nan")
        gap = (grp["mean_pred"] - grp["mean_act"]).abs()
        return float((gap * grp["count"]).sum() / grp["count"].sum())

    ece_dc  = _ece(cal_df[cal_df["Model"] == "DC+XGB"])
    ece_p   = _ece(cal_df[cal_df["Model"] == "Poisson+XGB"])

    cal_cols = st.columns(4)
    cal_metrics = [
        (cal_cols[0], f"{ece_dc*100:.1f}%",  "DC+XGB · ECE ↓",       ece_dc < ece_p),
        (cal_cols[1], f"{db:.3f}",            "DC+XGB · Brier ↓",     db < pb),
        (cal_cols[2], f"{ece_p*100:.1f}%",   "Poisson+XGB · ECE ↓",  ece_p < ece_dc),
        (cal_cols[3], f"{pb:.3f}",            "Poisson+XGB · Brier ↓", pb < db),
    ]
    for col, val, label, good in cal_metrics:
        cls = "metric-better" if good is True else ("metric-worse" if good is False else "")
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.82rem;color:#b8c0d0;text-align:center;margin:0.6rem 0 1.2rem;">'
        'ECE = average gap between predicted probability and actual outcome frequency '
        '(weighted by bin sample count). Lower is better — 0% means perfectly calibrated.</p>',
        unsafe_allow_html=True,
    )

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
        '<p style="font-size:0.82rem;color:#b8c0d0;text-align:center">'
        'Solid lines = DC+XGB · Dashed = Poisson+XGB · '
        '🔵 Home Win &nbsp;🟡 Draw &nbsp;🔴 Away Win</p>',
        unsafe_allow_html=True,
    )

    # ── Diagnostics matrix — where does the draw model bleed? ─────────────
    if "_dc_d" in bt.columns and "_act_d" in bt.columns:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">🔎  DIAGNOSTICS MATRIX · DRAW MODEL RESIDUALS</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.82rem;color:#b8c0d0;margin-bottom:1rem">'
            'Where does the Draw model bleed? Slice the backtest residuals (empirical − predicted) '
            'on the Draw market by closing-odds bucket, model-prob bucket, |xG diff|, and team. '
            'Positive gap = model under-predicts draws in that slice. Brier per-bucket.</p>',
            unsafe_allow_html=True,
        )
        # Pull closing odds (PSD) and xG diff into the backtest frame via merge
        df_full, _ = cached_data()
        merge_cols = ["Date", "HomeTeam", "AwayTeam"]
        extra = ["PSD", "xg_h", "xg_a"]
        avail = [c for c in extra if c in df_full.columns]
        bt_diag = bt.merge(
            df_full[merge_cols + avail].rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}),
            on=["Date", "Home", "Away"], how="left",
        )
        bt_diag["resid"] = bt_diag["_act_d"] - bt_diag["_dc_d"]
        bt_diag["brier"] = (bt_diag["_dc_d"] - bt_diag["_act_d"]) ** 2
        if "xg_h" in bt_diag.columns and "xg_a" in bt_diag.columns:
            bt_diag["xg_diff_abs"] = (bt_diag["xg_h"] - bt_diag["xg_a"]).abs()
        diag_tabs = st.tabs(["By model prob", "By closing odds", "By |xG diff|", "Per team"])

        def _render_bucket(agg_df: pd.DataFrame, x_col: str, x_label: str,
                           bin_label_col: str | None = None):
            """Render a paired bar chart (predicted vs empirical) + per-bin Brier."""
            if agg_df.empty:
                st.markdown('<div style="color:#b8c0d0;font-size:0.82rem">'
                            'Not enough data in this slice.</div>', unsafe_allow_html=True)
                return
            x_vals = agg_df[bin_label_col if bin_label_col else x_col].astype(str).tolist()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_vals, y=agg_df["mean_pred"].tolist(),
                name="Model predicted",
                marker=dict(color="#7c4dff", line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>Predicted: %{y:.1%}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=x_vals, y=agg_df["empirical"].tolist(),
                name="Actual draw freq",
                marker=dict(color="#ffd600", line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>Empirical: %{y:.1%}<extra></extra>",
            ))
            fig.update_layout(
                **{k: v for k, v in DARK.items() if k != "margin"},
                height=320, barmode="group",
                xaxis=dict(title=x_label, gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Draw probability", tickformat=".0%",
                           gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(x=0.02, y=0.98, font=dict(size=11)),
                margin=dict(t=20, l=40, r=20, b=50),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            tbl = agg_df.copy()
            tbl["Mean pred"] = tbl["mean_pred"].apply(lambda v: f"{v*100:.1f}%")
            tbl["Empirical"] = tbl["empirical"].apply(lambda v: f"{v*100:.1f}%")
            tbl["Gap"]       = (tbl["empirical"] - tbl["mean_pred"]).apply(lambda v: f"{v*100:+.1f}%")
            tbl["Brier"]     = tbl["brier"].apply(lambda v: f"{v:.3f}")
            tbl["N"]         = tbl["n"].astype(int)
            display_col = bin_label_col if bin_label_col else x_col
            st.dataframe(
                tbl[[display_col, "N", "Mean pred", "Empirical", "Gap", "Brier"]]
                  .rename(columns={display_col: x_label}),
                use_container_width=True, hide_index=True,
            )

        # By model prob bucket
        with diag_tabs[0]:
            prob_edges  = [0.0, 0.18, 0.22, 0.26, 0.30, 0.34, 1.01]
            prob_labels = ["<18%", "18-22%", "22-26%", "26-30%", "30-34%", "34%+"]
            bt_diag["prob_bucket"] = pd.cut(bt_diag["_dc_d"], bins=prob_edges,
                                            labels=prob_labels, right=False)
            agg = (bt_diag.groupby("prob_bucket", observed=True)
                          .agg(n=("brier", "count"),
                               mean_pred=("_dc_d", "mean"),
                               empirical=("_act_d", "mean"),
                               brier=("brier", "mean"))
                          .reset_index())
            _render_bucket(agg, "prob_bucket", "Model Probability Bucket")

        # By closing-odds bucket (Pinnacle close)
        with diag_tabs[1]:
            if "PSD" in bt_diag.columns and bt_diag["PSD"].notna().sum() > 0:
                odds_edges  = [1.0, 2.8, 3.0, 3.2, 3.4, 3.6, 4.0, 999.0]
                odds_labels = ["<2.8", "2.8-3.0", "3.0-3.2", "3.2-3.4", "3.4-3.6", "3.6-4.0", "4.0+"]
                bt_diag["odds_bucket"] = pd.cut(bt_diag["PSD"], bins=odds_edges,
                                                labels=odds_labels, right=False)
                agg = (bt_diag.dropna(subset=["odds_bucket"])
                              .groupby("odds_bucket", observed=True)
                              .agg(n=("brier", "count"),
                                   mean_pred=("_dc_d", "mean"),
                                   empirical=("_act_d", "mean"),
                                   brier=("brier", "mean"))
                              .reset_index())
                _render_bucket(agg, "odds_bucket", "Pinnacle Close (Draw)")
                st.markdown(
                    '<p style="font-size:0.78rem;color:#b8c0d0;text-align:center;margin-top:0.4rem">'
                    'Practitioner literature flags the <b style="color:#ffd600">3.20-3.60</b> '
                    'closing-odds bucket as the historically highest-EV draw zone.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div style="color:#b8c0d0;font-size:0.82rem">'
                            'Pinnacle closing odds not available in this dataset.</div>',
                            unsafe_allow_html=True)

        # By |xG diff| bucket
        with diag_tabs[2]:
            if "xg_diff_abs" in bt_diag.columns and bt_diag["xg_diff_abs"].notna().sum() > 0:
                xg_edges  = [0.0, 0.3, 0.6, 1.0, 1.5, 99.0]
                xg_labels = ["<0.3 (very tight)", "0.3-0.6", "0.6-1.0", "1.0-1.5", "1.5+"]
                bt_diag["xg_bucket"] = pd.cut(bt_diag["xg_diff_abs"], bins=xg_edges,
                                              labels=xg_labels, right=False)
                agg = (bt_diag.dropna(subset=["xg_bucket"])
                              .groupby("xg_bucket", observed=True)
                              .agg(n=("brier", "count"),
                                   mean_pred=("_dc_d", "mean"),
                                   empirical=("_act_d", "mean"),
                                   brier=("brier", "mean"))
                              .reset_index())
                _render_bucket(agg, "xg_bucket", "Match |xG diff|")
                st.markdown(
                    '<p style="font-size:0.78rem;color:#b8c0d0;text-align:center;margin-top:0.4rem">'
                    'Tight xG matchups (<0.3) are the canonical draw signal — empirical draw rate '
                    'should exceed model prediction here if the model is under-weighting parity.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div style="color:#b8c0d0;font-size:0.82rem">'
                            'xG data not available.</div>', unsafe_allow_html=True)

        # Per-team — biggest residual outliers
        with diag_tabs[3]:
            home_resid = bt_diag.groupby("Home").agg(
                n=("brier", "count"),
                mean_pred=("_dc_d", "mean"),
                empirical=("_act_d", "mean"),
                brier=("brier", "mean"),
            ).reset_index().rename(columns={"Home": "Team"})
            away_resid = bt_diag.groupby("Away").agg(
                n=("brier", "count"),
                mean_pred=("_dc_d", "mean"),
                empirical=("_act_d", "mean"),
                brier=("brier", "mean"),
            ).reset_index().rename(columns={"Away": "Team"})
            comb = pd.concat([home_resid, away_resid], ignore_index=True)
            # Aggregate per team without using groupby.apply(include_groups=...)
            # which requires pandas 2.1+. Build the aggregate manually so it
            # works on older pandas as well.
            _team_rows = []
            for team, g in comb.groupby("Team"):
                n_total = int(g["n"].sum())
                if n_total == 0:
                    continue
                _team_rows.append({
                    "Team":      team,
                    "n":         n_total,
                    "mean_pred": float((g["mean_pred"] * g["n"]).sum() / n_total),
                    "empirical": float((g["empirical"] * g["n"]).sum() / n_total),
                    "brier":     float((g["brier"]     * g["n"]).sum() / n_total),
                })
            agg = pd.DataFrame(_team_rows)
            agg["gap"]      = agg["empirical"] - agg["mean_pred"]
            agg["abs_gap"]  = agg["gap"].abs()
            agg = agg.sort_values("abs_gap", ascending=False).head(12)
            tbl = agg.copy()
            tbl["Mean pred"]  = tbl["mean_pred"].apply(lambda v: f"{v*100:.1f}%")
            tbl["Empirical"]  = tbl["empirical"].apply(lambda v: f"{v*100:.1f}%")
            tbl["Gap"]        = tbl["gap"].apply(lambda v: f"{v*100:+.1f}%")
            tbl["Brier"]      = tbl["brier"].apply(lambda v: f"{v:.3f}")
            st.markdown(
                '<p style="font-size:0.82rem;color:#b8c0d0;margin-bottom:0.5rem">'
                'Top 12 teams by absolute calibration gap. Persistently under-predicted draw teams '
                'are candidates for a per-team residual bias term.</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                tbl[["Team", "n", "Mean pred", "Empirical", "Gap", "Brier"]]
                  .rename(columns={"n": "N"}),
                use_container_width=True, hide_index=True,
            )

    # ── Equaliser-vs-winner empirical study ────────────────────────────────
    # Eoin's hypothesis: when a team is trailing at half-time, they're more likely
    # to score an equaliser than a level team is to score a winner. Tests an
    # asymmetry that math models miss because of point-value incentives.
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">⚽  RESEARCH · LATE-GAME EQUALISER ASYMMETRY</p>',
                unsafe_allow_html=True)
    df_eq, _ = cached_data()
    if {"HTHG", "HTAG", "FTHG", "FTAG"}.issubset(df_eq.columns):
        eq = df_eq.dropna(subset=["HTHG", "HTAG"]).copy()
        eq["HTHG"] = eq["HTHG"].astype(int)
        eq["HTAG"] = eq["HTAG"].astype(int)
        eq["ht_diff"]   = eq["HTHG"] - eq["HTAG"]
        eq["ft_diff"]   = eq["FTHG"] - eq["FTAG"]
        eq["sh_h_goals"] = eq["FTHG"] - eq["HTHG"]
        eq["sh_a_goals"] = eq["FTAG"] - eq["HTAG"]
        eq["sh_total_goals"] = eq["sh_h_goals"] + eq["sh_a_goals"]
        # Buckets: was the match level at HT, was someone trailing by exactly 1, etc.
        level_at_ht = eq[eq["ht_diff"] == 0]
        trailing_one_h = eq[eq["ht_diff"] == -1]   # home trailing by 1
        trailing_one_a = eq[eq["ht_diff"] == 1]    # away trailing by 1
        # Outcomes from level-at-HT
        level_n = len(level_at_ht)
        if level_n > 0:
            level_winner_rate = float((level_at_ht["ft_diff"] != 0).mean())
            level_stay_rate   = float((level_at_ht["ft_diff"] == 0).mean())
            level_avg_sh_goals = float(level_at_ht["sh_total_goals"].mean())
        else:
            level_winner_rate = level_stay_rate = level_avg_sh_goals = 0.0
        # Outcomes from trailing-by-1 (combine both perspectives)
        trail_n = len(trailing_one_h) + len(trailing_one_a)
        if trail_n > 0:
            home_eq_rate = float((trailing_one_h["FTHG"] >= trailing_one_h["FTAG"]).mean()) if len(trailing_one_h) > 0 else 0.0
            away_eq_rate = float((trailing_one_a["FTAG"] >= trailing_one_a["FTHG"]).mean()) if len(trailing_one_a) > 0 else 0.0
            # "Equalise or take the lead" — i.e., they recovered at least to a draw
            home_at_least_lvl = float((trailing_one_h["ft_diff"] >= 0).mean()) if len(trailing_one_h) > 0 else 0.0
            away_at_least_lvl = float((trailing_one_a["ft_diff"] <= 0).mean()) if len(trailing_one_a) > 0 else 0.0
            recover_rate = (home_at_least_lvl * len(trailing_one_h) + away_at_least_lvl * len(trailing_one_a)) / trail_n
            # Pure equaliser (got to draw, did not win)
            home_eq_only = float(((trailing_one_h["ft_diff"] == 0)).mean()) if len(trailing_one_h) > 0 else 0.0
            away_eq_only = float(((trailing_one_a["ft_diff"] == 0)).mean()) if len(trailing_one_a) > 0 else 0.0
            equaliser_rate = (home_eq_only * len(trailing_one_h) + away_eq_only * len(trailing_one_a)) / trail_n
        else:
            recover_rate = equaliser_rate = 0.0
        # Render
        eq1, eq2, eq3 = st.columns(3)
        with eq1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#ffd600">{equaliser_rate*100:.1f}%</div>
                <div class="metric-label">P(EQUALISER) · 1-DOWN AT HT</div>
            </div>""", unsafe_allow_html=True)
        with eq2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#7fa3ff">{level_winner_rate*100:.1f}%</div>
                <div class="metric-label">P(LATE WINNER) · LEVEL AT HT</div>
            </div>""", unsafe_allow_html=True)
        with eq3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#a78bfa">{recover_rate*100:.1f}%</div>
                <div class="metric-label">P(RECOVERY ≥ DRAW) · 1-DOWN AT HT</div>
            </div>""", unsafe_allow_html=True)
        # Verdict
        if equaliser_rate > 0:
            ratio = equaliser_rate / max(level_winner_rate, 0.001)
            ratio_msg = (
                f"Trailing teams equalise {ratio:.2f}× as often as level teams produce a winner."
                if ratio > 1.0 else
                f"Trailing teams equalise {ratio:.2f}× the rate of late winners — hypothesis NOT supported."
            )
            verdict_col = "#00e676" if ratio > 1.05 else ("#ffd600" if ratio > 0.95 else "#ff4081")
            st.markdown(
                f'<div style="padding:0.7rem 1rem;margin-top:0.6rem;'
                f'background:rgba(124,77,255,0.06);border-left:3px solid {verdict_col};'
                f'border-radius:6px;font-size:0.84rem;color:#e8eaf0">'
                f'<b style="color:{verdict_col}">{ratio_msg}</b> '
                f'&nbsp;·&nbsp; <span style="color:#8892a4">Sample: '
                f'{trail_n} trailing-by-1 matches, {level_n} level-at-HT matches '
                f'across {df_eq["Season"].nunique()} seasons. '
                f'2nd-half avg goals when level: {level_avg_sh_goals:.2f}.</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p style="font-size:0.88rem;color:#b8c0d0;margin-top:0.6rem;text-align:center">'
            'Caveat: HT/FT data is coarser than minute-by-minute. A "winner from level at HT" '
            'and an "equaliser when 1 down at HT" both require a goal in the second half, but '
            'the stronger version of the hypothesis (last-30-minute goals only) needs minute-level '
            'event data we don\'t cache. This study answers the directional question only.</p>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Season Outlook (Monte Carlo simulator)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 12, show_spinner=False)
def cached_season_sim(dc_hash: str, n_sims: int = 10_000, param_noise: float = 0.12):
    df, _ = cached_data()
    table_df  = get_current_table(df)
    remaining = fetch_remaining_season_fixtures()
    # Pre-season the loaded data still ends in May, so `get_current_table` hands
    # back LAST season's finished standings: every club on Played 38, the
    # relegated three still present, and the promoted sides missing entirely.
    # Simulating 380 fixtures on top of that projects a second season onto
    # completed points for the wrong twenty clubs.
    if table_is_stale_for(table_df, remaining):
        table_df = preseason_table(remaining)
    summary, sim_pts, fixtures_used, sim_positions = simulate_season(
        remaining, table_df, st.session_state["_dc_r_"],
        n_sims=n_sims, param_noise=param_noise, return_samples=True,
    )
    return table_df, summary, len(remaining), sim_pts, fixtures_used, sim_positions


def tab_season(df: pd.DataFrame, dc_r: dict):
    st.markdown('<p class="section-label" style="margin-top:1rem">Season Outlook · Monte Carlo Simulation</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1rem;color:#8892a4;margin-bottom:1.5rem;line-height:1.55">'
        'Simulates the remaining season <b>10,000 times</b> using Dixon-Coles match probabilities '
        '(with parametric-bootstrap noise on attack/defence ratings) to estimate each team\'s '
        'final position distribution. Elo and 5-game form trend shown alongside as supporting context.</p>',
        unsafe_allow_html=True,
    )

    # ── Rating-uncertainty slider ─────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.78rem;color:#8892a4;margin-bottom:0.3rem;font-weight:600">'
        '⚙️  Rating uncertainty (σ)</p>',
        unsafe_allow_html=True,
    )
    param_noise = st.slider(
        label="Rating uncertainty (σ)",
        min_value=0.02, max_value=0.30, value=0.12, step=0.01,
        key="sim_noise",
        label_visibility="collapsed",
        help=(
            "Gaussian noise injected on each team's attack/defence log-rating per simulation. "
            "Higher σ → more race unpredictability (outsiders have realistic title chances). "
            "Lower σ → favourites dominate. "
            "Cross-check against bookmaker odds: if the leader's title probability looks too "
            "certain vs the market, raise σ; if too flat, lower it."
        ),
    )
    st.markdown(
        f'<p style="font-size:0.92rem;color:#8892a4;margin:-0.4rem 0 1rem 0;">'
        f'σ = <b style="color:#a78bfa;font-size:1.05rem">{param_noise:.2f}</b> '
        f'&nbsp;·&nbsp; <b>0.05</b> ≈ very confident favourites &nbsp;·&nbsp; '
        f'<b>0.12</b> ≈ balanced default &nbsp;·&nbsp; <b>0.25</b> ≈ wide open race</p>',
        unsafe_allow_html=True,
    )

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run_clicked = st.button("⚡  Run Season Simulation", key="run_sim")
    if run_clicked:
        st.session_state["_sim_has_run"] = True

    if not st.session_state.get("_sim_has_run", False):
        return

    # Store dc_r in session state so the cached function can access it
    st.session_state["_dc_r_"] = dc_r

    with st.spinner("Fetching remaining fixtures and simulating 10,000 seasons…"):
        try:
            table_df, sim_df, n_remaining, sim_pts, fixtures_used, sim_positions = cached_season_sim(
                dc_hash=str(dc_r.get("home_adv", 0))[:8],
                param_noise=param_noise,
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

    if sim_df.empty:
        if _render_offseason_card(df, "Monte Carlo forecasts return when the "
                                  "new season's fixtures are released.", "season"):
            return
        st.warning("No remaining fixtures found — season may already be complete.")
        return

    st.markdown(
        f'<p style="font-size:0.78rem;color:#b8c0d0;text-align:center;margin-bottom:1rem;">'
        f'{n_remaining} remaining fixtures · 10,000 simulations</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Summary table — card-style rows with badges + Elo form ────────────
    st.markdown('<p class="section-label">Projected Final Table</p>', unsafe_allow_html=True)

    cur_pts = table_df.set_index("Team")["Pts"].to_dict()
    cur_played = table_df.set_index("Team")["Played"].to_dict()

    # ── Math sanity banner ────────────────────────────────────────────
    # Each remaining match awards (3 − p_draw) expected pts in total
    # (3 if decisive, 2 if drawn). Sum across remaining fixtures gives
    # total points still up for grabs. Then compare to the simulator's
    # implied pts-to-be-distributed = sum of (proj − current) per team.
    _exp_remaining = 0.0
    for _fix in fixtures_used:
        try:
            _pred = predict_dixon_coles(_fix["home"], _fix["away"], dc_r)
            _exp_remaining += 3.0 - _pred["draw"]
        except Exception:
            _exp_remaining += 2.66  # safe fallback
    _proj_gain_total = sum(
        max(0, sim_df.set_index("Team").loc[t, "mean_pts"] - cur_pts.get(t, 0))
        for t in sim_df["Team"]
    )
    # Decisive-match outcomes give 3 pts, draws give 2. So absolute bounds:
    _max_pts = 3.0 * len(fixtures_used)   # all decisive
    _min_pts = 2.0 * len(fixtures_used)   # all drawn
    _check_ok = (_min_pts - 1) <= _proj_gain_total <= (_max_pts + 1)
    _check_col = "#00e676" if _check_ok else "#ffd600"
    st.markdown(
        f'<div class="sim-math-banner" style="border-left-color:{_check_col}">'
        f'<span class="sim-math-icon">🧮</span>'
        f'<div class="sim-math-block">'
        f'<div class="sim-math-lbl">POINTS LEFT UP FOR GRABS</div>'
        f'<div class="sim-math-val">{_min_pts:.0f} – {_max_pts:.0f}</div>'
        f'<div class="sim-math-sub">expected ≈ <b>{_exp_remaining:.0f}</b> '
        f'· {len(fixtures_used)} matches × 2-3 pts per match</div>'
        f'</div>'
        f'<div class="sim-math-block">'
        f'<div class="sim-math-lbl">SUM OF PROJECTED GAINS</div>'
        f'<div class="sim-math-val" style="color:{_check_col}">'
        f'+{_proj_gain_total:.0f}</div>'
        f'<div class="sim-math-sub">total points the simulator distributes</div>'
        f'</div>'
        f'<div class="sim-math-block sim-math-verdict">'
        f'<div class="sim-math-lbl" style="color:{_check_col}">'
        f'{"✅ MATHEMATICALLY CONSISTENT" if _check_ok else "⚠️ DIVERGENCE"}</div>'
        f'<div class="sim-math-sub">Projected points distribute within '
        f'the {_min_pts:.0f}–{_max_pts:.0f} bound of remaining fixtures.</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Pre-compute Elo + recent Elo trajectory per team (last 5 matches)
    from data import _compute_elo_series as _elo_fn
    _records, _current_elo = _elo_fn(df)
    _records["Date"] = pd.to_datetime(_records["Date"])
    elo_form_map = {}
    for _t in sim_df["Team"].tolist():
        _team_recs = _records[
            (_records["HomeTeam"] == _t) | (_records["AwayTeam"] == _t)
        ].sort_values("Date").tail(5)
        if len(_team_recs) >= 2:
            _series = []
            for _, _r in _team_recs.iterrows():
                _series.append(_r["home_elo"] if _r["HomeTeam"] == _t else _r["away_elo"])
            _series.append(_current_elo.get(_t, 1500))
            _delta_5 = _series[-1] - _series[0]
            elo_form_map[_t] = {
                "elo":   _current_elo.get(_t, 1500),
                "form":  _series,
                "delta": _delta_5,
            }
        else:
            elo_form_map[_t] = {"elo": _current_elo.get(_t, 1500), "form": [], "delta": 0}

    n_total = len(sim_df)
    _no_pill = '<span class="sim-pill-none">—</span>'

    # ── Most-Likely Points: pick the modal outcome of each remaining match
    # (whichever H/D/A has highest probability) and add the resulting points
    # for each team. This is a single deterministic scenario — useful as a
    # "path of least resistance" finish next to the probabilistic mean.
    most_likely_gain = {t: 0 for t in sim_df["Team"]}
    for _fix in fixtures_used:
        h, a = _fix["home"], _fix["away"]
        try:
            _pred = predict_dixon_coles(h, a, dc_r)
        except Exception:
            continue
        # Find the modal outcome
        outcomes = {
            "H": _pred["home_win"],
            "D": _pred["draw"],
            "A": _pred["away_win"],
        }
        modal = max(outcomes, key=outcomes.get)
        if modal == "H":
            most_likely_gain[h] = most_likely_gain.get(h, 0) + 3
        elif modal == "A":
            most_likely_gain[a] = most_likely_gain.get(a, 0) + 3
        else:  # draw
            most_likely_gain[h] = most_likely_gain.get(h, 0) + 1
            most_likely_gain[a] = most_likely_gain.get(a, 0) + 1

    rows_html_parts = []
    for pos, row in sim_df.iterrows():
        team    = row["Team"]
        pts     = cur_pts.get(team, 0)
        played  = cur_played.get(team, 0)
        mean_pts = round(row["mean_pts"], 1)
        rank    = pos + 1

        # Zone classification stripe colour
        if rank == 1:                               zone_cls, zone_lbl = "zone-title",  "👑"
        elif rank <= 4:                             zone_cls, zone_lbl = "zone-top4",   "🏆"
        elif rank <= 6:                             zone_cls, zone_lbl = "zone-top6",   "⭐"
        elif rank > n_total - 3:                    zone_cls, zone_lbl = "zone-rel",    "⬇"
        else:                                       zone_cls, zone_lbl = "zone-mid",    ""

        # Probability pills — only show when meaningful
        pills = ""
        def _pill(prob, color, label):
            return (f'<span class="sim-pill" style="color:{color};'
                    f'background:rgba({_hex_to_rgb(color)},0.10);'
                    f'border:1px solid rgba({_hex_to_rgb(color)},0.35)">'
                    f'{label} <b>{prob*100:.0f}%</b></span>')
        if row["p_title"]     > 0.005: pills += _pill(row["p_title"],     "#ffd600", "Title")
        if row["p_top4"]      > 0.02:  pills += _pill(row["p_top4"],      "#00e5ff", "Top 4")
        if row["p_top6"]      > 0.02:  pills += _pill(row["p_top6"],      "#7c4dff", "Top 6")
        if row["p_relegated"] > 0.02:  pills += _pill(row["p_relegated"], "#ff4081", "Rel")

        # Elo + form sparkline
        elo_data = elo_form_map.get(team, {})
        elo_now  = elo_data.get("elo", 1500)
        elo_d    = elo_data.get("delta", 0)
        spark_svg = _elo_sparkline_svg(elo_data.get("form", []), width=80, height=28) \
            if len(elo_data.get("form", [])) >= 2 else ""
        delta_col = "#00e676" if elo_d > 5 else ("#ff4081" if elo_d < -5 else "#8892a4")
        delta_arrow = "▲" if elo_d > 0 else ("▼" if elo_d < 0 else "→")

        # Team badge (bigger 44px)
        badge_url = _BADGE_URL.get(team, "")
        badge_html = (f'<img class="team-badge" src="{badge_url}" alt="{team}" />'
                      if badge_url else '')

        # Most-likely deterministic finish (current + modal-outcome wins/draws)
        ml_pts = pts + most_likely_gain.get(team, 0)

        rows_html_parts.append(
            f'<div class="sim-row {zone_cls}">'
              f'<div class="sim-rank">{rank}<span class="sim-zone-emoji">{zone_lbl}</span></div>'
              f'<div class="sim-team">{badge_html}<span>{team}</span></div>'
              f'<div class="sim-played">{played}</div>'
              f'<div class="sim-pts-now">{pts}</div>'
              f'<div class="sim-pts-proj">{mean_pts}</div>'
              f'<div class="sim-pts-ml" title="Most-likely deterministic finish — '
                f'each remaining match scored at its highest-probability outcome '
                f'(W/D/L), summed onto current points.">{ml_pts}</div>'
              f'<div class="sim-elo">'
                f'<div class="sim-elo-num">{elo_now:.0f}</div>'
                f'<div class="sim-elo-delta" style="color:{delta_col}">'
                  f'{delta_arrow} {abs(elo_d):.0f}</div>'
              f'</div>'
              f'<div class="sim-spark">{spark_svg}</div>'
              f'<div class="sim-pills">{pills or _no_pill}</div>'
            f'</div>'
        )

    header_html = (
        '<div class="sim-row sim-header">'
        '<div class="sim-rank">#</div>'
        '<div class="sim-team">TEAM</div>'
        '<div class="sim-played">P</div>'
        '<div class="sim-pts-now">PTS</div>'
        '<div class="sim-pts-proj" title="Mean projected points across 10,000 sims">PROJ μ</div>'
        '<div class="sim-pts-ml" title="Most likely deterministic finish — pick modal '
        'W/D/L outcome of every remaining match">MOST LIKELY</div>'
        '<div class="sim-elo">ELO · 5-GAME Δ</div>'
        '<div class="sim-spark">FORM</div>'
        '<div class="sim-pills">SCENARIOS</div>'
        '</div>'
    )

    st.markdown(
        '<div class="sim-table">' + header_html + ''.join(rows_html_parts) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Position probability heatmap ──────────────────────────────────────
    st.markdown('<p class="section-label">Position Probability Heatmap</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.92rem;color:#8892a4;margin-bottom:0.6rem">'
        'How often each team finishes at each rank across 10,000 simulations. '
        'Brighter cells = more likely outcome. Hover for exact % per cell.</p>',
        unsafe_allow_html=True,
    )

    n_teams = len(sim_df)
    z = np.array([
        [sim_df.iloc[i][f"pos_{p+1}"] * 100 for p in range(n_teams)]
        for i in range(n_teams)
    ])
    team_labels = sim_df["Team"].tolist()

    text_grid = [[f"<b>{v:.0f}%</b>" if v >= 1 else "" for v in row] for row in z]

    # Use numeric x positions so shapes/labels align cleanly with cells
    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(1, n_teams + 1)),  # 1, 2, …, 20 numeric
        y=list(range(n_teams)),          # numeric y so images line up
        colorscale=[
            [0.0,  "#1a1d27"],
            [0.10, "#2d3148"],
            [0.25, "#4f46e5"],
            [0.50, "#7c4dff"],
            [0.75, "#22d3ee"],
            [1.0,  "#a5f3fc"],
        ],
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#fff", family="Inter"),
        showscale=False,
        hovertemplate="<b>%{customdata}</b><br>Finish #%{x}: %{z:.1f}%<extra></extra>",
        customdata=[[t] * n_teams for t in team_labels],
        xgap=2, ygap=2,
    ))

    # ── European/relegation zones (per UEFA qualification rules) ────────
    # Full-column vertical bands so it reads as "this is the X race" at a glance.
    # Labels above the heatmap stay in a single row, short enough to avoid overlap.
    def _zone_band(c_start, c_end, color, label):
        # Full-height column band UNDERNEATH the cells (plus a brighter strip
        # at the very top + bottom so the zone is unmistakable). Cell content
        # still reads on top because layer="below".
        fig.add_shape(
            type="rect",
            x0=c_start - 0.5, x1=c_end + 0.5,
            y0=-0.5,           y1=n_teams - 0.5,
            xref="x", yref="y",
            fillcolor=color, opacity=0.28,
            line=dict(color=color, width=2),
            layer="below",
        )
        # Pill label centred above the band
        fig.add_annotation(
            x=(c_start + c_end) / 2.0, y=-1.3,
            text=f"<b>{label}</b>",
            showarrow=False,
            xref="x", yref="y",
            font=dict(size=12, color=color, family="Inter"),
            bgcolor=f"rgba({_hex_to_rgb(color)},0.18)",
            bordercolor=color, borderwidth=1.2, borderpad=5,
        )

    # Short labels — keeps Europa + Conference from running into each other
    ZONES = [
        (1, 1,                  "#ffd600", "👑 TITLE"),
        (2, 5,                  "#00e5ff", "🏆 CHAMPIONS LEAGUE"),
        (6, 7,                  "#fb923c", "🥈 EUROPA"),
        (8, 8,                  "#22c55e", "🥉 CONFERENCE"),
        (n_teams - 2, n_teams,  "#ff4081", "⬇ RELEGATION"),
    ]
    for c_start, c_end, col, lbl in ZONES:
        _zone_band(c_start, c_end, col, lbl)

    # ── Replace y-axis text with badge images via layout.images ─────────
    # Plotly heatmap doesn't natively support image y-tick labels, so we
    # overlay a badge for each team row, positioned just left of the plot.
    for i, team in enumerate(team_labels):
        url = _BADGE_URL.get(team, "")
        if not url:
            continue
        fig.add_layout_image(dict(
            source=url,
            xref="paper", yref="y",
            x=-0.005, y=i,  # just outside the left edge of the plot
            sizex=0.05, sizey=0.95,
            xanchor="right", yanchor="middle",
            layer="above",
        ))

    fig.update_layout(
        plot_bgcolor="#10131c",
        paper_bgcolor="#0e111a",
        font=dict(family="Inter", color="#cdd"),
        height=780,
        margin=dict(t=110, b=30, l=110, r=30),  # extra top room for zone labels
        title=dict(
            text="Probability (%) of finishing in each position · 10,000 simulated seasons",
            font=dict(size=15, color="#cdd", family="Inter"), x=0.5, y=0.98,
        ),
        xaxis=dict(
            title=dict(text="Final Position",
                       font=dict(size=14, color="#7c4dff", family="Inter")),
            side="top",
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(size=15, color="#cdd", family="Inter"),
            tickmode="array",
            tickvals=list(range(1, n_teams + 1)),
            ticktext=[str(p) for p in range(1, n_teams + 1)],
            range=[0.5, n_teams + 0.5],   # tight to the cell grid, no padding
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(255,255,255,0.05)",
            tickmode="array",
            tickvals=list(range(n_teams)),
            ticktext=[""] * n_teams,
            range=[n_teams - 0.5, -0.5],
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Team drill-down ───────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🔍  Team Drill-Down</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:#b8c0d0;margin-bottom:1rem">'
        'Pick a team to see their remaining fixtures, expected points per match, '
        'and the distribution of their final points across 10,000 simulated seasons.</p>',
        unsafe_allow_html=True,
    )

    teams_order = list(table_df["Team"])
    t_idx_map = {t: i for i, t in enumerate(teams_order)}

    default_team = sim_df.iloc[0]["Team"] if not sim_df.empty else teams_order[0]
    sel_team = st.selectbox(
        "Team", teams_order,
        index=teams_order.index(default_team) if default_team in teams_order else 0,
        key="sim_team_select",
    )

    if sel_team and sel_team in t_idx_map:
        ti = t_idx_map[sel_team]
        team_pts_samples = sim_pts[:, ti]
        team_pos_samples = sim_positions[:, ti]

        # Summary cards for selected team
        mean_pts = float(team_pts_samples.mean())
        p5_pts, p95_pts = np.percentile(team_pts_samples, [5, 95])
        mean_pos = float(team_pos_samples.mean())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{mean_pts:.1f}</div>
                <div class="metric-label">Mean Final Points</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{p5_pts:.0f} – {p95_pts:.0f}</div>
                <div class="metric-label">90% Points Range</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">#{mean_pos:.1f}</div>
                <div class="metric-label">Mean Finish Position</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            # Most-likely position
            pos_counts = np.bincount(team_pos_samples, minlength=len(teams_order) + 1)[1:]
            mode_pos = int(np.argmax(pos_counts)) + 1
            mode_pct = float(pos_counts[mode_pos - 1]) / len(team_pos_samples) * 100
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">#{mode_pos} <span style="font-size:0.82rem;color:#8892a4">({mode_pct:.0f}%)</span></div>
                <div class="metric-label">Most Likely Finish</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Points distribution histogram ────────────────────────────────
        hist_col, fix_col = st.columns([1, 1])

        with hist_col:
            st.markdown('<p class="section-label" style="font-size:0.78rem">Final Points Distribution</p>',
                        unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=team_pts_samples,
                nbinsx=30,
                marker=dict(
                    color="#7c4dff",
                    line=dict(color="#a5b4fc", width=1),
                ),
                opacity=0.92,
                hovertemplate="<b>%{x:.0f} pts</b><br>%{y} sims<extra></extra>",
            ))
            fig_hist.add_vline(x=mean_pts, line=dict(color="#00e5ff", width=3, dash="dash"),
                               annotation_text=f"<b>Mean {mean_pts:.0f}</b>",
                               annotation_position="top right",
                               annotation=dict(font=dict(color="#00e5ff", size=14, family="Inter"),
                                               bgcolor="rgba(0,229,255,0.10)",
                                               bordercolor="rgba(0,229,255,0.35)",
                                               borderpad=4))
            fig_hist.update_layout(
                plot_bgcolor="#10131c",
                paper_bgcolor="#0e111a",
                font=dict(family="Inter", color="#cdd"),
                height=360,
                margin=dict(t=32, b=40, l=50, r=20),
                xaxis=dict(
                    title=dict(text="Final Points",
                               font=dict(size=13, color="#7c4dff", family="Inter")),
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=14, color="#cdd", family="Inter"),
                    zeroline=False,
                ),
                yaxis=dict(
                    title=dict(text="Simulations",
                               font=dict(size=13, color="#7c4dff", family="Inter")),
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=14, color="#cdd", family="Inter"),
                    zeroline=False,
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        # ── Remaining fixtures table — where points are gained/lost ─────
        with fix_col:
            st.markdown('<p class="section-label" style="font-size:0.78rem">Remaining Fixtures — Expected Points</p>',
                        unsafe_allow_html=True)

            # Compute expected points per remaining fixture for this team
            team_fixtures = []
            for fix in fixtures_used:
                h, a = fix["home"], fix["away"]
                if sel_team not in (h, a):
                    continue
                pred = predict_dixon_coles(h, a, dc_r)
                if sel_team == h:
                    opp = a
                    venue = "H"
                    exp_pts = 3.0 * pred["home_win"] + 1.0 * pred["draw"]
                    p_win  = pred["home_win"]
                    p_draw = pred["draw"]
                    p_loss = pred["away_win"]
                else:
                    opp = h
                    venue = "A"
                    exp_pts = 3.0 * pred["away_win"] + 1.0 * pred["draw"]
                    p_win  = pred["away_win"]
                    p_draw = pred["draw"]
                    p_loss = pred["home_win"]
                team_fixtures.append({
                    "Opp": opp, "Venue": venue,
                    "exp_pts": exp_pts,
                    "p_win": p_win, "p_draw": p_draw, "p_loss": p_loss,
                })

            if not team_fixtures:
                st.info("No remaining fixtures for this team.")
            else:
                total_exp = sum(f["exp_pts"] for f in team_fixtures)
                st.markdown(
                    f'<p style="font-size:0.82rem;color:#b8c0d0;margin-bottom:0.5rem">'
                    f'{len(team_fixtures)} fixtures · expected {total_exp:.1f} points</p>',
                    unsafe_allow_html=True,
                )

                rows_html = ""
                for f in team_fixtures:
                    # Classify by expected points: favourite / even / underdog
                    if f["exp_pts"] >= 2.0:
                        tag_color, tag_text = "#00e676", "FAVOURITE"
                    elif f["exp_pts"] <= 1.0:
                        tag_color, tag_text = "#ff4081", "UNDERDOG"
                    else:
                        tag_color, tag_text = "#ffd600", "EVEN"
                    venue_color = "#3d6eff" if f["Venue"] == "H" else "#ff4081"
                    rows_html += f"""<tr>
                        <td style="color:{venue_color};font-weight:700;width:2rem">{f['Venue']}</td>
                        <td style="font-weight:700;color:#e8eaf0">{tb(f['Opp'], 18)}</td>
                        <td style="color:#b8c0d0;font-size:0.82rem">
                            W {f['p_win']*100:.0f}% · D {f['p_draw']*100:.0f}% · L {f['p_loss']*100:.0f}%
                        </td>
                        <td style="font-weight:900;color:#e8eaf0;text-align:right">{f['exp_pts']:.2f}</td>
                        <td><span style="background:{tag_color}22;color:{tag_color};
                            border:1px solid {tag_color}44;border-radius:10px;padding:1px 6px;
                            font-size:0.78rem;font-weight:700">{tag_text}</span></td>
                    </tr>"""

                st.markdown(f"""<div style="overflow-x:auto;max-height:320px;overflow-y:auto">
                <table class="score-table" style="text-align:left;min-width:100%;font-size:0.82rem">
                    <thead><tr>
                        <th>Venue</th><th>Opponent</th><th>W/D/L</th>
                        <th style="text-align:right">xPts</th><th></th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table></div>""", unsafe_allow_html=True)


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
            f'🎯 <b style="color:#00e676">Best call:</b> {tb(best["home"], 24)} vs {tb(best["away"], 24)} — '
            f'predicted {pred_labels[best["pred_ftr"]]} with '
            f'<b style="color:#e8eaf0">{best["prob_actual"]*100:.0f}% confidence</b> · '
            f'Actual: {best["hg"]}–{best["ag"]}</div>',
            unsafe_allow_html=True,
        )
    with hi2:
        st.markdown(
            f'<div style="background:rgba(255,64,129,0.07);border:1px solid rgba(255,64,129,0.2);'
            f'border-radius:12px;padding:0.8rem 1.1rem;margin:1rem 0 0.3rem;font-size:0.82rem;color:#aab">'
            f'😱 <b style="color:#ff4081">Biggest surprise:</b> {tb(upset["home"], 24)} vs {tb(upset["away"], 24)} — '
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
                f'<p style="font-size:0.78rem;font-weight:700;color:#b8c0d0;'
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

        # Per-team xG chip — matches the "Total Expected Goals" treatment
        xg_txt = ""
        if r.get("xg_h") is not None and r.get("xg_a") is not None:
            xg_txt = (
                '<span class="res-xg-chip" '
                f'title="Per-team Expected Goals based on shot quality from Understat. '
                f'{r["home"]}: {r["xg_h"]:.2f} · {r["away"]}: {r["xg_a"]:.2f}">'
                '<span class="res-xg-lbl">xG</span>'
                f'<span class="res-xg-val res-xg-h">{r["xg_h"]:.1f}</span>'
                '<span class="res-xg-dash">–</span>'
                f'<span class="res-xg-val res-xg-a">{r["xg_a"]:.1f}</span>'
                '</span>'
            )

        goal_diff = r["total"] - r["exp_total"]
        goal_diff_txt = f'+{goal_diff:.1f}' if goal_diff > 0 else f'{goal_diff:.1f}'

        # Goal-diff colour: green if within 0.5, amber if 0.5-1.0, red if more
        if abs(goal_diff) < 0.5:    diff_col = "#00e676"
        elif abs(goal_diff) < 1.0:  diff_col = "#ffd600"
        else:                       diff_col = "#ff4081"

        st.markdown(f"""
        <div class="res-card {card_cls}">
            <div class="res-teams">
                <span class="res-team">{tb(r['home'], 36)}</span>
                <span class="res-score">{r['hg']}–{r['ag']}</span>
                <span class="res-team">{tb(r['away'], 36)}</span>
                <span class="res-badge {badge_cls}">{badge_txt}</span>
            </div>
            <div class="res-prob-bar">
                <div class="bar-home bar-sm" style="width:{bw_h}%">{r['p_h']*100:.0f}%</div>
                <div class="bar-draw bar-sm" style="width:{bw_d}%">{r['p_d']*100:.0f}%</div>
                <div class="bar-away bar-sm" style="width:{bw_a}%">{r['p_a']*100:.0f}%</div>
            </div>
            <div class="res-headline">
                <div class="res-headline-item">
                    <span class="res-headline-label">Predicted</span>
                    <span class="res-headline-value" style="color:{pred_col}">
                        {pred_labels[r['pred_ftr']]}</span>
                </div>
                <div class="res-headline-item">
                    <span class="res-headline-label">Actual</span>
                    <span class="res-headline-value">{act_label}</span>
                </div>
                <div class="res-goals">
                    <div class="res-goals-block">
                        <span class="res-goals-lbl">Total Expected Goals</span>
                        <span class="res-goals-val">{r['exp_total']:.1f}</span>
                    </div>
                    <span class="res-goals-vs">vs</span>
                    <div class="res-goals-block">
                        <span class="res-goals-lbl">Total Actual Goals</span>
                        <span class="res-goals-val">{r['total']}</span>
                    </div>
                    <span style="color:{diff_col};font-weight:800;font-size:0.86rem;
                          font-variant-numeric:tabular-nums">({goal_diff_txt})</span>
                </div>
            </div>
            <div class="res-meta">
                {xg_txt}
                <span>O/U 2.5: <b style="color:{'#00e676' if r['ou_correct'] else '#ff4081'}">
                {'✓ correct' if r['ou_correct'] else '✗ wrong'}</b></span>
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

    # ── Model recap header — story summary above the per-match rows ───────
    _render_gw_recap(display_records)

    _render_gw_records(display_records, display_label)


def _render_gw_recap(records: list[dict]) -> None:
    """A 4-block summary banner above the gameweek result cards.
    Tells the model's story for the week: how often it called the right side,
    its biggest correct conviction, its biggest surprise, and average
    confidence on the outcome that actually happened."""
    if not records:
        return

    n_total   = len(records)
    n_correct = sum(1 for r in records if r["correct"])
    win_pct   = (n_correct / n_total * 100) if n_total else 0
    record_col = ("#00e676" if win_pct >= 50 else
                  "#ffd600" if win_pct >= 33 else "#ff4081")

    # Average model probability assigned to the outcome that actually occurred
    # — a quick calibration / confidence read for the week.
    avg_conf = sum(r["prob_actual"] for r in records) / n_total
    conf_col = ("#00e676" if avg_conf >= 0.45 else
                "#ffd600" if avg_conf >= 0.34 else "#ff4081")

    # Biggest hit — correct prediction with the highest model conviction
    correct_hits = [r for r in records if r["correct"]]
    biggest_hit  = max(correct_hits, key=lambda r: r["prob_actual"]) if correct_hits else None

    # Biggest miss — wrong prediction with the lowest probability assigned to
    # the actual result (i.e. outcome that surprised the model the most)
    wrong = [r for r in records if not r["correct"]]
    biggest_miss = min(wrong, key=lambda r: r["prob_actual"]) if wrong else None

    pred_lbl = {"H": "Home", "D": "Draw", "A": "Away"}

    def _block(icon, label, value_html, sub_html, color="#7c4dff"):
        return (
            f'<div class="gw-recap-block" style="border-left-color:{color}">'
              f'<div class="gw-recap-icon">{icon}</div>'
              f'<div class="gw-recap-lbl">{label}</div>'
              f'<div class="gw-recap-val">{value_html}</div>'
              f'<div class="gw-recap-sub">{sub_html}</div>'
            f'</div>'
        )

    # Build each block
    record_block = _block(
        "✅", "MODEL RECORD",
        f'<span style="color:{record_col}">{n_correct}<span style="color:#b8c0d0">/</span>{n_total}</span>',
        f'{win_pct:.0f}% correct on H/D/A this week',
        record_col,
    )

    if biggest_hit:
        h, a = biggest_hit["home"], biggest_hit["away"]
        _bh_actual = pred_lbl[biggest_hit["actual_ftr"]]
        hit_html = (
            f'<span style="font-size:1.18rem">{tb(h, 28)} <span style="color:#b8c0d0">vs</span> {tb(a, 28)}</span>'
        )
        sub_html = (f'<b style="color:#00e676">{_bh_actual} Win</b> ·  model had '
                    f'<b>{biggest_hit["prob_actual"]*100:.0f}%</b> on the outcome — and it landed')
        hit_block = _block("🎯", "BIGGEST HIT", hit_html, sub_html, "#00e676")
    else:
        hit_block = _block("🎯", "BIGGEST HIT",
                            "<span style='color:#b8c0d0'>—</span>",
                            "no correct calls this week", "#445")

    if biggest_miss:
        h, a = biggest_miss["home"], biggest_miss["away"]
        _miss_actual = pred_lbl[biggest_miss["actual_ftr"]]
        _miss_pred   = pred_lbl[biggest_miss["pred_ftr"]]
        miss_html = (
            f'<span style="font-size:1.18rem">{tb(h, 28)} <span style="color:#b8c0d0">vs</span> {tb(a, 28)}</span>'
        )
        sub_html = (f'model picked <b>{_miss_pred}</b>, actual was '
                    f'<b style="color:#ff4081">{_miss_actual}</b> · '
                    f'model only had <b>{biggest_miss["prob_actual"]*100:.0f}%</b> on it')
        miss_block = _block("💥", "BIGGEST MISS", miss_html, sub_html, "#ff4081")
    else:
        miss_block = _block("💥", "BIGGEST MISS",
                             "<span style='color:#b8c0d0'>—</span>",
                             "no surprises this week", "#445")

    conf_block = _block(
        "📊", "AVG CONFIDENCE",
        f'<span style="color:{conf_col}">{avg_conf*100:.0f}%</span>',
        f'mean prob the model assigned to the<br>actual outcome — higher = more calibrated',
        conf_col,
    )

    st.markdown(
        '<div class="gw-recap">'
        + record_block + hit_block + miss_block + conf_block +
        '</div>',
        unsafe_allow_html=True,
    )


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


def _hex_to_rgb(hex_str: str) -> str:
    """'#00e676' → '0,230,118'  (string for use in CSS rgba())."""
    hex_str = hex_str.lstrip("#")
    return f"{int(hex_str[0:2], 16)},{int(hex_str[2:4], 16)},{int(hex_str[4:6], 16)}"


def _section_header(title: str, subtitle: str = "") -> None:
    """Visual section header inside settings expanders — vibrant accent + small subtitle."""
    sub = (f'<div style="font-size:0.86rem;color:#b8c0d0;margin-top:0.1rem;'
           f'line-height:1.4">{subtitle}</div>' if subtitle else '')
    st.markdown(
        f'<div style="margin:1.1rem 0 0.6rem 0;padding-top:0.7rem;'
        f'border-top:1px solid #2d3148">'
        f'<div style="font-size:0.86rem;color:#a78bfa;font-weight:700;'
        f'letter-spacing:1px">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def _render_settings_chip(port: dict, label_prefix: str = "port") -> None:
    """One-line summary of the current live config — surfaces all key params
    above the expander so users see state at a glance without expanding.
    """
    s = port["settings"]
    auto_on = s.get("auto_bet_enabled", False)
    mkts    = s.get("auto_markets", [])
    mkt_str = "+".join(mkts) if mkts else "(none)"
    mp      = int(s.get("min_prob", 0.30) * 100)
    mev     = int(s.get("auto_bet_threshold", s.get("min_ev", 0.40)) * 100)
    kel     = int(s.get("kelly_fraction", 1.0) * 100)
    maxs    = int(s.get("max_stake_pct", 0.33) * 100)
    skip_l  = s.get("skip_late_season", False)
    skip_t  = s.get("skip_home_title_race", False)
    sim     = s.get("use_simultaneous_kelly", False)
    has_u25 = "under25" in (s.get("market_gates") or {})

    chip = lambda txt, color="#2d3148", bg="rgba(124,77,255,0.07)": (
        f'<span style="display:inline-block;padding:0.2rem 0.55rem;'
        f'margin:0.15rem 0.25rem 0.15rem 0;border-radius:14px;'
        f'background:{bg};border:1px solid {color};'
        f'font-size:0.84rem;color:#e8eaf0;letter-spacing:0.3px">{txt}</span>'
    )
    on_chip  = lambda txt: chip(txt, "#00e676", "rgba(0,230,118,0.10)")
    off_chip = lambda txt: chip(txt, "#445",    "rgba(0,0,0,0.18)")
    auto_chip = (chip("🤖 Auto-Bet ON", "#00e676", "rgba(0,230,118,0.14)")
                 if auto_on else chip("🤖 Auto-Bet OFF", "#445", "rgba(0,0,0,0.18)"))

    chips = [
        chip(f"📊 {mkt_str}", "#7c4dff", "rgba(124,77,255,0.10)"),
        chip(f"🎯 mp ≥ {mp}%"),
        chip(f"💎 EV ≥ {mev}%"),
        chip(f"💰 Kelly {kel}%"),
        chip(f"🚦 max {maxs}%"),
        on_chip("⏭ Skip Mar-May ✓") if skip_l else off_chip("⏭ Skip Mar-May ✗"),
        on_chip("⏭ Skip title-race ✓") if skip_t else off_chip("⏭ Skip title-race ✗"),
        on_chip("🔗 Sim corr ✓") if sim else off_chip("🔗 Sim corr ✗"),
        on_chip("🎚 U2.5 sep gates ✓") if has_u25 else off_chip("🎚 U2.5 sep gates ✗"),
        auto_chip,
    ]
    st.markdown(
        '<div style="margin:0.4rem 0 0.5rem 0;line-height:1.7">'
        + "".join(chips) + '</div>',
        unsafe_allow_html=True,
    )


# ── Preset configurations ─────────────────────────────────────────────────
# Three named presets for one-click switching. Each is a complete settings
# dict patch — keys not listed are left as-is on the existing portfolio.
SETTINGS_PRESETS = {
    "recommended": {
        "label": "🎯 Recommended", "color": "#7c4dff",
        "tooltip": ("WF-validated optimal · highest OOS profit across 60 random configs. "
                    "Draws + U2.5 with separate gates."),
        "patch": {
            "min_prob": 0.30, "auto_bet_threshold": 0.40, "min_ev": 0.40,
            "kelly_fraction": 1.00, "max_stake_pct": 0.33,
            "auto_markets": ["D", "under25"],
            "skip_late_season": True, "skip_home_title_race": False,
            "use_simultaneous_kelly": True, "use_calibrated_probs": True,
            "market_gates": {"under25": {"min_prob": 0.50, "min_ev": 0.05}},
        },
    },
    "aggressive": {
        "label": "🚀 Aggressive", "color": "#ff4081",
        "tooltip": ("Max upside, more variance. Looser draw gates, larger max stake, "
                    "adds Over 2.5. Higher in-sample profit; less proven OOS."),
        "patch": {
            "min_prob": 0.25, "auto_bet_threshold": 0.35, "min_ev": 0.35,
            "kelly_fraction": 1.00, "max_stake_pct": 0.50,
            "auto_markets": ["D", "under25", "over25"],
            "skip_late_season": True, "skip_home_title_race": False,
            "use_simultaneous_kelly": True, "use_calibrated_probs": True,
            "market_gates": {
                "under25": {"min_prob": 0.50, "min_ev": 0.04},
                "over25":  {"min_prob": 0.50, "min_ev": 0.05},
            },
        },
    },
    "conservative": {
        "label": "🛡️ Conservative", "color": "#00e676",
        "tooltip": ("Sharper edge, fewer bets, lower variance. Half-Kelly base, draws-only, "
                    "tighter max stake, both filters on."),
        "patch": {
            "min_prob": 0.32, "auto_bet_threshold": 0.45, "min_ev": 0.45,
            "kelly_fraction": 0.50, "max_stake_pct": 0.20,
            "auto_markets": ["D"],
            "skip_late_season": True, "skip_home_title_race": True,
            "use_simultaneous_kelly": True, "use_calibrated_probs": True,
            "market_gates": {},
        },
    },
}


def _render_preset_buttons(port: dict, key_prefix: str, save_fn) -> None:
    """One-click preset loader — swaps in a complete validated settings patch."""
    st.markdown(
        '<div style="font-size:0.86rem;color:#a78bfa;font-weight:700;'
        'letter-spacing:1px;margin-bottom:0.4rem">⚡  QUICK PRESETS</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(SETTINGS_PRESETS))
    for col, (preset_key, preset) in zip(cols, SETTINGS_PRESETS.items()):
        with col:
            if st.button(preset["label"],
                         key=f"{key_prefix}_preset_{preset_key}",
                         help=preset["tooltip"],
                         use_container_width=True):
                for k, v in preset["patch"].items():
                    port["settings"][k] = v
                if not port["settings"].get("market_gates"):
                    port["settings"].pop("market_gates", None)
                save_fn(port)
                st.success(f"✅ Applied preset: {preset['label']}")
                st.rerun()
    st.markdown(
        '<div style="font-size:0.84rem;color:#b8c0d0;margin:0.4rem 0 0.7rem 0">'
        'Click a preset to overwrite all settings below in one click. '
        'Bet history and bankroll are preserved.</div>',
        unsafe_allow_html=True,
    )


# ── Backtest presets — set the slider/checkbox session state in one click ──
BACKTEST_PRESETS = {
    "recommended": {
        "label": "🎯 Recommended", "tooltip": "WF-validated D+U2.5 config",
        "values": {
            "minev": 40, "minprob": 30, "kelly": 1.0, "maxstake": 33,
            "skip_late": True, "skip_title": False, "sim": True,
            "odds_src": "B365", "detect_src": "(same as place)",
            "u25_gates": True,
        },
    },
    "aggressive": {
        "label": "🚀 Aggressive", "tooltip": "Looser gates, max stake 50%, more bets",
        "values": {
            "minev": 35, "minprob": 25, "kelly": 1.0, "maxstake": 50,
            "skip_late": True, "skip_title": False, "sim": True,
            "odds_src": "B365", "detect_src": "(same as place)",
            "u25_gates": True,
        },
    },
    "conservative": {
        "label": "🛡️ Conservative", "tooltip": "Sharp edge, half-Kelly, draws-only, both filters",
        "values": {
            "minev": 45, "minprob": 32, "kelly": 0.5, "maxstake": 20,
            "skip_late": True, "skip_title": True, "sim": True,
            "odds_src": "B365", "detect_src": "(same as place)",
            "u25_gates": False,
        },
    },
    "research_ps_max": {
        "label": "🔬 Research (F_PS_Max)", "tooltip": "Detect at Pinnacle, place at Max — best CLV",
        "values": {
            "minev": 40, "minprob": 30, "kelly": 1.0, "maxstake": 33,
            "skip_late": True, "skip_title": False, "sim": True,
            "odds_src": "Max", "detect_src": "PS",
            "u25_gates": True,
        },
    },
}


def _render_backtest_presets(key_prefix: str) -> None:
    """One-click preset loader for backtest sliders — writes to session state then reruns."""
    st.markdown(
        '<div style="font-size:0.86rem;color:#a78bfa;font-weight:700;'
        'letter-spacing:1px;margin:0 0 0.4rem 0">⚡  QUICK PRESETS</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(BACKTEST_PRESETS))
    for col, (preset_key, preset) in zip(cols, BACKTEST_PRESETS.items()):
        with col:
            if st.button(preset["label"],
                         key=f"{key_prefix}_bt_preset_{preset_key}",
                         help=preset["tooltip"],
                         use_container_width=True):
                for slot, val in preset["values"].items():
                    st.session_state[f"{key_prefix}_{slot}"] = val
                st.rerun()
    st.markdown(
        '<div style="font-size:0.84rem;color:#b8c0d0;margin:0.4rem 0 0.7rem 0">'
        'Click a preset to set all sliders below in one click.</div>',
        unsafe_allow_html=True,
    )


def _render_clv_trend(port: dict, key_prefix: str = "") -> None:
    """Rolling CLV panel — last 5/10/20 medians + sparkline + edge-status banner.

    Surfaces the most actionable signal: is our edge alive RIGHT NOW, or has the
    market caught up? CLV stabilises in ~50 bets but the rolling window flags
    drift much earlier than aggregate medians do.
    """
    rolling = pf.clv_rolling(port, windows=(5, 10, 20))
    if rolling["alert"] == "insufficient":
        return  # not enough tagged bets to show anything useful

    st.markdown(
        '<div style="margin-top:0.9rem;font-size:0.86rem;letter-spacing:2px;'
        'color:#a78bfa;font-weight:700;text-transform:uppercase">'
        'CLV TREND · last N bets</div>',
        unsafe_allow_html=True,
    )

    col_specs = []
    for w in (5, 10, 20):
        wd = rolling["windows"].get(w)
        if wd is None:
            col_specs.append((f"—", f"LAST {w}", "#445", "—"))
            continue
        med = wd["median"] * 100
        col = "#00e676" if med >= 1.0 else ("#ffd600" if med >= 0 else "#ff4081")
        col_specs.append((f"{med:+.2f}%", f"LAST {w}", col, f"{wd['pct_pos']:.0f}% pos"))

    cols = st.columns(3)
    for c, (val, label, color, sub) in zip(cols, col_specs):
        with c:
            st.markdown(
                f'<div style="background:#1a1d27;border:1px solid #2d3148;'
                f'border-radius:8px;padding:0.75rem 1rem">'
                f'<div style="font-size:1.4rem;font-weight:700;color:{color};'
                f'line-height:1.1">{val}</div>'
                f'<div style="font-size:0.84rem;color:#8892a4;letter-spacing:2px;'
                f'margin-top:0.25rem">{label}  <span style="color:#b8c0d0">·  {sub}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Sparkline of all CLVs over time (rolling-mean smoothed for clarity)
    series = rolling["series"]
    if len(series) >= 5:
        clvs_pct = [c * 100 for _, c in series]
        # Rolling-5 mean for the sparkline so it's readable
        roll = []
        for i in range(len(clvs_pct)):
            start = max(0, i - 4)
            roll.append(sum(clvs_pct[start:i+1]) / (i - start + 1))
        fig = go.Figure()
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dot")
        fig.add_trace(go.Scatter(
            x=list(range(1, len(roll) + 1)), y=roll,
            mode="lines", line=dict(color="#7c4dff", width=2),
            fill="tozeroy", fillcolor="rgba(124,77,255,0.10)",
            hovertemplate="Bet %{x}: %{y:+.2f}% (5-bet rolling avg)<extra></extra>",
        ))
        fig.update_layout(
            **{k: v for k, v in DARK.items() if k != "margin"},
            height=140, margin=dict(t=8, b=8, l=0, r=0),
            xaxis=dict(title="bet # (chronological)", gridcolor="rgba(255,255,255,0.04)",
                       title_font=dict(size=10), tickfont=dict(size=9)),
            yaxis=dict(title="CLV % (rolling 5)", gridcolor="rgba(255,255,255,0.04)",
                       title_font=dict(size=10), tickfont=dict(size=9), zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"clv_spark_{key_prefix}_{len(series)}")

    # Edge-status banner
    alert = rolling["alert"]
    msg, color = {
        "ok":    ("✅  Edge looks alive — recent median CLV positive. Keep auto-bet running.",     "#00e676"),
        "watch": ("⚠️  Watch — recent CLV near zero. Edge is marginal; another 5-10 bets will tell.", "#ffd600"),
        "drift": ("🚨  Drift — recent median CLV is negative. Edge has narrowed or reversed; "
                  "consider pausing auto-bet and reviewing.",                                       "#ff4081"),
    }.get(alert, ("", "#445"))
    if msg:
        st.markdown(
            f'<div style="padding:0.6rem 1rem;margin-top:0.5rem;'
            f'background:rgba(124,77,255,0.04);border-left:3px solid {color};'
            f'border-radius:6px;font-size:0.78rem;color:#e8eaf0">{msg}</div>',
            unsafe_allow_html=True,
        )


def _elo_sparkline_svg(values: list[float], width: int = 110, height: int = 36,
                        color: str = "#7c4dff") -> str:
    """Tiny SVG sparkline for a list of Elo values (chronological).
    Returns an inline SVG string with a polyline + endpoint dot.
    """
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"><text x="4" y="{height//2+4}" fill="#445" font-size="10">no data</text></svg>'
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 1.0)
    # Pad y by 15% so endpoints don't sit on the edges
    pad = 6
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = int((i / (n - 1)) * (width - 2 * pad)) + pad
        y = int((1.0 - (v - vmin) / span) * (height - 2 * pad)) + pad
        pts.append(f"{x},{y}")
    polyline = " ".join(pts)
    last_x, last_y = pts[-1].split(",")
    rising = values[-1] >= values[0]
    line_col = "#00e676" if rising else "#ff4081"
    fill_col = "rgba(0,230,118,0.10)" if rising else "rgba(255,64,129,0.10)"
    # Build a closed area for fill underneath the line
    area_pts = polyline + f" {width-pad},{height-pad} {pad},{height-pad}"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline points="{area_pts}" fill="{fill_col}" stroke="none" />'
        f'<polyline points="{polyline}" fill="none" stroke="{line_col}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" />'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.5" fill="{line_col}" />'
        f'</svg>'
    )


def tab_elo(df: pd.DataFrame, teams: list) -> None:
    """Live Elo table — current rating + week delta + form % over last 5 + sparkline."""
    from data import _compute_elo_series

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    max_date = df["Date"].max()

    # Full Elo trajectory across all data
    records, current_elo = _compute_elo_series(df)
    records["Date"] = pd.to_datetime(records["Date"])

    # Snapshot: Elo as of one week ago (filter df to ≤ max_date − 7d, recompute)
    cutoff_week = max_date - pd.Timedelta(days=7)
    df_week_ago = df[df["Date"] <= cutoff_week]
    _, elo_week_ago = _compute_elo_series(df_week_ago) if not df_week_ago.empty else ({}, {})

    # Per-team: collect their last 5 PRE-MATCH Elos (chronological) + last 5 results
    rows = []
    for team in teams:
        team_records = records[
            (records["HomeTeam"] == team) | (records["AwayTeam"] == team)
        ].sort_values("Date")
        if len(team_records) == 0:
            continue
        elo_series = []
        for _, r in team_records.iterrows():
            elo_series.append(r["home_elo"] if r["HomeTeam"] == team else r["away_elo"])
        # Append the post-final-match Elo so the sparkline ends at "current"
        elo_series.append(current_elo.get(team, 1500.0))
        last_5 = elo_series[-6:]  # 5 pre-match + 1 post = 6 points total

        cur     = current_elo.get(team, 1500.0)
        wk_ago  = elo_week_ago.get(team, cur)
        delta_w = cur - wk_ago
        # 5-game form: % change from 5 matches ago to current
        five_ago = elo_series[-6] if len(elo_series) >= 6 else elo_series[0]
        form_pct = (cur - five_ago) / five_ago * 100 if five_ago else 0.0

        rows.append({
            "team":     team,
            "elo":      cur,
            "delta_w":  delta_w,
            "form_pct": form_pct,
            "spark":    last_5,
        })

    rows.sort(key=lambda r: r["elo"], reverse=True)
    if not rows:
        st.warning("No Elo data available — check the match dataset.")
        return

    # ── Header summary ──────────────────────────────────────────────────
    top, mid, bot = rows[0], rows[len(rows)//2], rows[-1]
    st.markdown(
        f'<div style="padding:1rem 1.3rem;margin-bottom:1.3rem;'
        f'border-radius:14px;'
        f'background:linear-gradient(135deg,rgba(124,77,255,0.10),rgba(0,229,255,0.04));'
        f'border:1px solid rgba(124,77,255,0.25)">'
        f'<div style="font-size:0.86rem;letter-spacing:3px;color:#a78bfa;'
        f'font-weight:700;text-transform:uppercase;margin-bottom:0.5rem">'
        f'📈  LIVE PREMIER LEAGUE ELO</div>'
        f'<div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:0.86rem;color:#e8eaf0">'
        f'<div><span style="color:#8892a4">League leader:</span> '
        f'<b style="color:#00e676">{top["team"]}</b> ({top["elo"]:.0f})</div>'
        f'<div><span style="color:#8892a4">Median:</span> '
        f'<b>{mid["team"]}</b> ({mid["elo"]:.0f})</div>'
        f'<div><span style="color:#8892a4">Bottom:</span> '
        f'<b style="color:#ff4081">{bot["team"]}</b> ({bot["elo"]:.0f})</div>'
        f'<div><span style="color:#8892a4">Spread:</span> '
        f'<b>{top["elo"]-bot["elo"]:.0f}</b> Elo points</div>'
        f'<div><span style="color:#8892a4">As of:</span> '
        f'<b>{max_date.date()}</b></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Table styling ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    .elo-row {
        display: grid;
        grid-template-columns: 40px 56px 1.6fr 1.1fr 1.1fr 1.1fr 130px;
        gap: 1rem;
        align-items: center;
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #1a1d27;
        transition: background-color 0.18s ease;
    }
    .elo-row:hover { background: rgba(124,77,255,0.06); }
    .elo-row.header {
        font-size: 0.78rem; letter-spacing: 2px; color: #b8c0d0;
        font-weight: 700; text-transform: uppercase;
        border-bottom: 1px solid #2d3148;
    }
    .elo-rank   { font-size: 1rem; font-weight: 700; color: #b8c0d0; text-align: right; }
    .elo-rank.top    { color: #00e676; }
    .elo-rank.bottom { color: #ff4081; }
    .elo-badge img   { width: 40px; height: 40px; object-fit: contain;
                       filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4)); transition: transform 0.18s; }
    .elo-row:hover .elo-badge img { transform: scale(1.12); }
    .elo-team        { font-size: 0.96rem; font-weight: 600; color: #e8eaf0; }
    .elo-value       { font-size: 1.1rem; font-weight: 700; color: #e8eaf0;
                       font-variant-numeric: tabular-nums; }
    .elo-delta       { font-size: 0.86rem; font-weight: 600;
                       font-variant-numeric: tabular-nums; }
    .elo-up   { color: #00e676; }
    .elo-down { color: #ff4081; }
    .elo-flat { color: #b8c0d0; }
    .elo-spark { line-height: 0; }

    @media (max-width: 900px) {
        .elo-row {
            grid-template-columns: 30px 44px 1fr 80px 70px;
            gap: 0.5rem; padding: 0.55rem 0.7rem;
        }
        .elo-row > :nth-child(6),
        .elo-row > :nth-child(7) { display: none; }
        .elo-row.header { font-size: 0.78rem; letter-spacing: 1.2px; }
        .elo-team { font-size: 0.86rem; }
        .elo-badge img { width: 32px; height: 32px; }
    }
    @media (max-width: 600px) {
        .elo-row {
            grid-template-columns: 26px 36px 1fr 64px;
            gap: 0.4rem; padding: 0.5rem 0.55rem;
        }
        .elo-row > :nth-child(5),
        .elo-row > :nth-child(6),
        .elo-row > :nth-child(7) { display: none; }
        .elo-team { font-size: 0.82rem; }
        .elo-badge img { width: 28px; height: 28px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header row ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="elo-row header">'
        '<div>#</div>'
        '<div></div>'
        '<div>Team</div>'
        '<div>Elo</div>'
        '<div>Δ Week</div>'
        '<div>Form (5g)</div>'
        '<div>Trajectory</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Body rows ───────────────────────────────────────────────────────
    n_rows = len(rows)
    for i, r in enumerate(rows):
        rank = i + 1
        rank_cls = "top" if rank <= 4 else ("bottom" if rank > n_rows - 3 else "")
        # Delta this week
        if r["delta_w"] >  0.5:  d_cls, d_arrow = "elo-up",   "▲"
        elif r["delta_w"] < -0.5: d_cls, d_arrow = "elo-down", "▼"
        else:                     d_cls, d_arrow = "elo-flat", "→"
        # Form
        if r["form_pct"] >  0.1:  f_cls, f_arrow = "elo-up",   "▲"
        elif r["form_pct"] < -0.1: f_cls, f_arrow = "elo-down", "▼"
        else:                     f_cls, f_arrow = "elo-flat", "→"
        spark = _elo_sparkline_svg(r["spark"])

        badge_url = _BADGE_URL.get(r["team"], "")
        badge_img = (f'<img class="team-badge" src="{badge_url}" alt="{r["team"]}">'
                     if badge_url else '<span style="color:#b8c0d0">—</span>')
        st.markdown(
            f'<div class="elo-row">'
            f'<div class="elo-rank {rank_cls}">{rank}</div>'
            f'<div class="elo-badge">{badge_img}</div>'
            f'<div class="elo-team">{r["team"]}</div>'
            f'<div class="elo-value">{r["elo"]:.0f}</div>'
            f'<div class="elo-delta {d_cls}">{d_arrow} {abs(r["delta_w"]):.0f}</div>'
            f'<div class="elo-delta {f_cls}">{f_arrow} {abs(r["form_pct"]):.2f}%</div>'
            f'<div class="elo-spark">{spark}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Footer note ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:1rem;font-size:0.86rem;color:#b8c0d0;line-height:1.6">'
        'Elo computed from full PL match history (2021-22 onwards). K-factor 32, '
        'home advantage 100 pts. Δ Week = current Elo − Elo seven days ago. '
        'Form (5g) = % change in Elo over the last 5 league matches.'
        '</div>',
        unsafe_allow_html=True,
    )


def tab_team_deepdive(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                      draw_xgb_m, draw_fc, teams, elo_dict):
    """All-data-on-one-team view. Picker → big hero → fixtures, results,
    Elo trajectory, portfolio bets involving the team, lifetime CLV."""
    from data import _compute_elo_series

    # ── Team picker ──────────────────────────────────────────────────────
    default = "Arsenal" if "Arsenal" in teams else teams[0]
    pick_col1, pick_col2 = st.columns([3, 1])
    with pick_col1:
        sel = st.selectbox("Team", teams,
                           index=teams.index(default) if default in teams else 0,
                           key="td_team", label_visibility="collapsed")
    if not sel:
        return

    # ── Pre-compute the building blocks ──────────────────────────────────
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Elo trajectory
    records, current_elo = _compute_elo_series(df)
    records["Date"] = pd.to_datetime(records["Date"])
    elo_now = current_elo.get(sel, 1500.0)
    team_recs = records[
        (records["HomeTeam"] == sel) | (records["AwayTeam"] == sel)
    ].sort_values("Date")
    elo_series = []
    for _, r in team_recs.iterrows():
        elo_series.append(r["home_elo"] if r["HomeTeam"] == sel else r["away_elo"])
    elo_series.append(elo_now)
    elo_form_5g = elo_series[-6:] if len(elo_series) >= 6 else elo_series
    delta_5g = (elo_form_5g[-1] - elo_form_5g[0]) if len(elo_form_5g) >= 2 else 0

    # Current league position + points
    try:
        table_df = get_current_table(df)
        team_row = table_df[table_df["Team"] == sel]
        if len(team_row):
            cur_pos    = int(team_row.index[0]) + 1
            cur_pts    = int(team_row.iloc[0]["Pts"])
            cur_played = int(team_row.iloc[0]["Played"])
            cur_gd     = int(team_row.iloc[0].get("GD", 0))
        else:
            cur_pos = cur_pts = cur_played = cur_gd = 0
    except Exception:
        cur_pos = cur_pts = cur_played = cur_gd = 0

    # Last 5 results
    last5 = team_recs.tail(5)
    last5_results = []
    for _, r in last5.iterrows():
        is_home = r["HomeTeam"] == sel
        opp = r["AwayTeam"] if is_home else r["HomeTeam"]
        # Find actual result
        actual = df[(df["Date"] == r["Date"]) & (df["HomeTeam"] == r["HomeTeam"]) &
                    (df["AwayTeam"] == r["AwayTeam"])]
        if len(actual) == 0:
            continue
        a = actual.iloc[0]
        ftr = a["FTR"]
        won = (is_home and ftr == "H") or (not is_home and ftr == "A")
        drew = ftr == "D"
        result_lbl = "W" if won else ("D" if drew else "L")
        score = f"{int(a['FTHG'])}-{int(a['FTAG'])}"
        last5_results.append({
            "date":  r["Date"].strftime("%d-%b-%y"),
            "opp":   opp,
            "venue": "H" if is_home else "A",
            "score": score,
            "result": result_lbl,
        })

    # Upcoming fixtures involving this team
    try:
        all_fixtures = fetch_upcoming_fixtures(lookahead_days=45)
    except Exception:
        all_fixtures = []
    team_fixtures = [f for f in all_fixtures if sel in (f.get("home"), f.get("away"))][:8]

    # Predictions for upcoming fixtures
    fixture_predictions = []
    for fix in team_fixtures:
        h, a = fix["home"], fix["away"]
        if h not in teams or a not in teams:
            continue
        try:
            hs  = get_current_stats(df, h, elo_dict=elo_dict)
            as_ = get_current_stats(df, a, elo_dict=elo_dict)
            _, _, blended = full_predict(h, a, dc_r, dc_draw_r, xgb_m, feat_cols,
                                          draw_xgb_m, draw_fc, hs, as_)
        except Exception:
            continue
        fixture_predictions.append({
            "date": fix["date"],
            "home": h, "away": a,
            "is_home": h == sel,
            "p_h": blended["home_win"],
            "p_d": blended["draw"],
            "p_a": blended["away_win"],
        })

    # Portfolio bets involving this team
    try:
        main_p = pf.load_portfolio()
        mt_p   = pf.load_portfolio_two()
    except Exception:
        main_p = {"bets": []}
        mt_p   = {"bets": []}

    def _team_bets(p_bets):
        return [b for b in p_bets
                if b.get("type") != "acca"
                and (b.get("home") == sel or b.get("away") == sel)]
    main_bets = _team_bets(main_p.get("bets", []))
    mt_bets   = _team_bets(mt_p.get("bets", []))
    all_bets  = [(b, "Main") for b in main_bets] + [(b, "Mock 2") for b in mt_bets]

    # Lifetime stats
    settled_bets = [(b, src) for b, src in all_bets if b["status"] in ("won", "lost")]
    n_won  = sum(1 for b, _ in settled_bets if b["status"] == "won")
    n_lost = len(settled_bets) - n_won
    total_pl = sum((b.get("profit") or 0) for b, _ in settled_bets)
    clv_vals = [b["clv"] for b, _ in settled_bets if b.get("clv") is not None]
    median_clv = (np.median(clv_vals) * 100) if clv_vals else None
    pending_bets_team = [(b, src) for b, src in all_bets if b["status"] == "pending"]

    # ── Hero card with badge + stats ─────────────────────────────────────
    badge_url = _BADGE_URL.get(sel, "")
    spark_svg = _elo_sparkline_svg(elo_form_5g, width=140, height=40) if len(elo_form_5g) >= 2 else ""
    delta_col = "#00e676" if delta_5g > 5 else ("#ff4081" if delta_5g < -5 else "#8892a4")
    delta_arrow = "▲" if delta_5g > 0 else ("▼" if delta_5g < 0 else "→")

    st.markdown(f"""
    <div class="td-hero">
      <div class="td-hero-badge">
        {f'<img class="team-badge" src="{badge_url}" />' if badge_url else ''}
      </div>
      <div class="td-hero-body">
        <div class="td-hero-name">{sel}</div>
        <div class="td-hero-stats">
          <div class="td-stat"><div class="td-stat-lbl">POSITION</div>
              <div class="td-stat-val">#{cur_pos}</div></div>
          <div class="td-stat"><div class="td-stat-lbl">POINTS</div>
              <div class="td-stat-val">{cur_pts}</div></div>
          <div class="td-stat"><div class="td-stat-lbl">PLAYED</div>
              <div class="td-stat-val">{cur_played}</div></div>
          <div class="td-stat"><div class="td-stat-lbl">GD</div>
              <div class="td-stat-val">{cur_gd:+d}</div></div>
          <div class="td-stat"><div class="td-stat-lbl">ELO</div>
              <div class="td-stat-val">{elo_now:.0f}</div></div>
          <div class="td-stat td-stat-form"><div class="td-stat-lbl">FORM (5G)</div>
              <div class="td-stat-val" style="color:{delta_col}">{delta_arrow} {abs(delta_5g):.0f}</div>
              <div class="td-stat-spark">{spark_svg}</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Last 5 results strip ─────────────────────────────────────────────
    if last5_results:
        st.markdown('<p class="section-label">📋 RECENT FORM (LAST 5)</p>',
                    unsafe_allow_html=True)
        chips = []
        for r in last5_results:
            res_col = "#00e676" if r["result"] == "W" else ("#ffd600" if r["result"] == "D" else "#ff4081")
            opp_url = _BADGE_URL.get(r["opp"], "")
            opp_badge = (f'<img class="team-badge" src="{opp_url}" '
                         f'style="width:34px;height:34px" />' if opp_url else "")
            chips.append(
                f'<div class="td-result-chip">'
                f'<div class="td-result-circle" style="background:rgba({_hex_to_rgb(res_col)},0.15);'
                f'border-color:{res_col};color:{res_col}">{r["result"]}</div>'
                f'<div class="td-result-meta">'
                f'<div class="td-result-score">{r["score"]}</div>'
                f'<div class="td-result-opp">{r["venue"]} vs {r["opp"]}</div>'
                f'<div class="td-result-date">{r["date"]}</div>'
                f'</div>'
                f'{opp_badge}'
                f'</div>'
            )
        st.markdown('<div class="td-results-row">' + ''.join(chips) + '</div>',
                    unsafe_allow_html=True)

    # ── Upcoming fixtures with predictions ───────────────────────────────
    if fixture_predictions:
        st.markdown('<p class="section-label">📅 NEXT FIXTURES · MODEL PREDICTIONS</p>',
                    unsafe_allow_html=True)
        for fp in fixture_predictions:
            h, a = fp["home"], fp["away"]
            h_pct = round(fp["p_h"] * 100, 1)
            d_pct = round(fp["p_d"] * 100, 1)
            a_pct = round(fp["p_a"] * 100, 1)
            try:
                date_str = fp["date"].strftime("%d-%b-%y")
            except Exception:
                date_str = ""

            # Model verdict for this team
            if fp["is_home"]:
                team_p = fp["p_h"]
                opp_p  = fp["p_a"]
            else:
                team_p = fp["p_a"]
                opp_p  = fp["p_h"]
            if team_p > 0.5:    verdict = "FAVOURED"; v_col = "#00e676"
            elif team_p > 0.35: verdict = "EVEN";     v_col = "#ffd600"
            else:               verdict = "UNDERDOG"; v_col = "#ff4081"

            st.markdown(f"""
            <div class="td-fix-card">
              <div class="td-fix-top">
                <span class="td-fix-date">📅 {date_str}</span>
                <span class="td-fix-verdict" style="color:{v_col};border-color:{v_col}55;background:rgba({_hex_to_rgb(v_col)},0.10)">{verdict} · {team_p*100:.0f}%</span>
              </div>
              <div class="td-fix-row">
                <div class="fixture-team-name fixture-team-home">{tb(h, 44)}</div>
                <div class="fixture-prob-bar" style="height:38px">
                  <div class="bar-home" style="width:{h_pct}%">{h_pct:.0f}%</div>
                  <div class="bar-draw" style="width:{d_pct}%">{d_pct:.0f}%</div>
                  <div class="bar-away" style="width:{a_pct}%">{a_pct:.0f}%</div>
                </div>
                <div class="fixture-team-name fixture-team-away">{tb(a, 44)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="padding:1rem;color:#8892a4;text-align:center">'
            'No upcoming fixtures found in the next 45 days.</div>',
            unsafe_allow_html=True,
        )

    # ── Portfolio history involving this team ────────────────────────────
    if all_bets:
        st.markdown('<p class="section-label">💰 PORTFOLIO BETS · ALL-TIME</p>',
                    unsafe_allow_html=True)
        # Stats row
        clv_col = "#00e676" if (median_clv or 0) >= 1 else (
                  "#ffd600" if (median_clv or 0) >= 0 else "#ff4081")
        pl_col  = "#00e676" if total_pl >= 0 else "#ff4081"
        clv_html = f'{median_clv:+.2f}%' if median_clv is not None else '—'
        st.markdown(f"""
        <div class="td-portfolio-stats">
          <div class="td-stat-card">
            <div class="td-stat-lbl">SETTLED</div>
            <div class="td-stat-val">{n_won}<span style="color:#b8c0d0">/{len(settled_bets)}</span></div>
            <div class="td-stat-sub">{n_won} won · {n_lost} lost</div>
          </div>
          <div class="td-stat-card">
            <div class="td-stat-lbl">P&amp;L</div>
            <div class="td-stat-val" style="color:{pl_col}">£{total_pl:+,.0f}</div>
            <div class="td-stat-sub">all bets, all portfolios</div>
          </div>
          <div class="td-stat-card">
            <div class="td-stat-lbl">MEDIAN CLV</div>
            <div class="td-stat-val" style="color:{clv_col}">{clv_html}</div>
            <div class="td-stat-sub">{len(clv_vals)} tagged bet{'s' if len(clv_vals) != 1 else ''}</div>
          </div>
          <div class="td-stat-card">
            <div class="td-stat-lbl">PENDING</div>
            <div class="td-stat-val">{len(pending_bets_team)}</div>
            <div class="td-stat-sub">£{sum(b["stake"] for b, _ in pending_bets_team):,.0f} at risk</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Bet list (newest first)
        bet_rows = []
        for b, src in sorted(all_bets,
                             key=lambda x: x[0].get("date") or x[0].get("placed_at") or "",
                             reverse=True):
            try:
                d = pd.to_datetime(b.get("date") or b.get("placed_at"))
                date_short = d.strftime("%d-%b-%y")
            except Exception:
                date_short = (b.get("date") or "")[:10]
            opp = b["away"] if b["home"] == sel else b["home"]
            venue = "H" if b["home"] == sel else "A"
            status = b["status"]
            if status == "won":      result_html = '<span class="td-bet-status td-won">✅ Won</span>'
            elif status == "lost":   result_html = '<span class="td-bet-status td-lost">❌ Lost</span>'
            else:                    result_html = '<span class="td-bet-status td-pend">⏳ Pending</span>'
            profit = b.get("profit") or 0
            if status == "pending":  pl_str = "—"
            else:
                pl_str = f'£{profit:+,.0f}'
            pl_color = "#00e676" if profit > 0 else ("#ff4081" if profit < 0 else "#8892a4")
            bet_rows.append(
                f'<div class="td-bet-row">'
                f'<div class="td-bet-date">{date_short}</div>'
                f'<div class="td-bet-port">{src}</div>'
                f'<div class="td-bet-match">{venue} vs {opp}</div>'
                f'<div class="td-bet-sel">{b.get("selection", "")}</div>'
                f'<div class="td-bet-odds">{b.get("odds", 0):.2f}</div>'
                f'<div class="td-bet-stake">£{b.get("stake", 0):,.0f}</div>'
                f'<div class="td-bet-result">{result_html}</div>'
                f'<div class="td-bet-pl" style="color:{pl_color}">{pl_str}</div>'
                f'</div>'
            )
        header = (
            '<div class="td-bet-row td-bet-header">'
            '<div>DATE</div><div>WHERE</div><div>OPPONENT</div>'
            '<div>SELECTION</div><div>ODDS</div><div>STAKE</div>'
            '<div>RESULT</div><div>P&amp;L</div></div>'
        )
        st.markdown(
            '<div class="td-bet-table">' + header + ''.join(bet_rows) + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding:1rem;color:#8892a4;text-align:center;font-size:0.96rem">'
            f'No portfolio bets involving {sel} yet.</div>',
            unsafe_allow_html=True,
        )


def tab_portfolio(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict):
    # ── Auto-settle pending bets ─────────────────────────────────────────
    port = pf.load_portfolio()
    n_settled = pf.auto_settle(port, df)
    # Backfill closing-line value for any settled bets that haven't been tagged.
    # Reads Pinnacle close (PSH/PSD/PSA) from the football-data CSVs — non-mutating
    # to bet stakes/profits, only adds `closing_odds` and `clv` fields.
    n_clv_added = pf.backfill_clv_for_settled_bets(port, df)
    if n_settled > 0 or n_clv_added > 0:
        pf.save_portfolio(port)

    stats    = pf.portfolio_stats(port)
    settings = port["settings"]
    min_ev        = float(settings.get("min_ev", 0.05))
    kelly_frac    = float(settings.get("kelly_fraction", 0.5))
    max_stake_pct = float(settings.get("max_stake_pct", 0.10))
    api_key       = pf.resolve_odds_api_key(settings.get("odds_api_key", ""))

    # Fit isotonic calibrators once per session (shares cache with Backtesting tab)
    try:
        with st.spinner("Fitting probability calibration…"):
            calibrators = cached_calibrators(len(df))
    except Exception:
        calibrators = {}

    # ── Main vs Mock Two at a glance — the A/B story, surfaced up front ──
    try:
        _ab_mt = pf.load_portfolio_two()
        _ab_mt_s = pf.portfolio_stats(_ab_mt)
        _ab_mn_clv = (pf.clv_summary(port).get("median_clv") or 0) * 100
        _ab_mt_clv = (pf.clv_summary(_ab_mt).get("median_clv") or 0) * 100

        def _ab_half(label, color, s, clv_v):
            pcol = "#00e676" if s["profit"] >= 0 else "#ff4081"
            return (
                f'<div style="flex:1;min-width:230px">'
                f'<div style="font-size:0.78rem;letter-spacing:1.8px;font-weight:800;'
                f'color:{color}">{label}</div>'
                f'<div style="font-size:1.25rem;font-weight:900;color:#e8eaf0">'
                f'£{s["bankroll"]:,.0f} '
                f'<span style="font-size:0.84rem;color:{pcol}">'
                f'{"+" if s["profit"] >= 0 else ""}£{s["profit"]:,.0f}</span></div>'
                f'<div style="font-size:0.78rem;color:#8892a4">'
                f'ROI {s["roi"]:+.1f}% · {s["n_settled"]} bets · '
                f'win {s["win_rate"]:.0f}% · CLV {clv_v:+.2f}%</div></div>'
            )

        _ab_c1, _ab_c2 = st.columns([8, 2])
        with _ab_c1:
            st.markdown(
                '<div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;'
                'padding:0.8rem 1.1rem;background:#11162a;border:1px solid #1c2440;'
                'border-radius:14px;margin-bottom:0.6rem;'
                'animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both">'
                + _ab_half("MAIN", "#3d6eff", stats, _ab_mn_clv)
                + '<div style="font-size:0.84rem;color:#b8c0d0;font-weight:900">vs</div>'
                + _ab_half("MOCK TWO", "#7c4dff", _ab_mt_s, _ab_mt_clv)
                + '</div>',
                unsafe_allow_html=True,
            )
        with _ab_c2:
            st.markdown('<div style="padding-top:0.9rem">', unsafe_allow_html=True)
            if st.button("Mock Two →", key="port_to_mt", use_container_width=True):
                st.session_state["_active_view"] = "portfolio2"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        pass

    # ── Status chip — current config at a glance ─────────────────────────
    _render_settings_chip(port, label_prefix="port")

    # ── Settings expander ────────────────────────────────────────────────
    with st.expander("⚙️  Portfolio Settings", expanded=False):

        # ── Quick presets ──────────────────────────────────────────────
        _render_preset_buttons(port, key_prefix="port", save_fn=pf.save_portfolio)

        # ── Section: Stake & Bankroll ──────────────────────────────────
        _section_header("💰  Stake & Bankroll", "How much you bet, sized by Kelly")
        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            new_initial = st.number_input(
                "Starting Budget (£)", 100.0, 100_000.0,
                float(port["initial_bankroll"]), 100.0, key="port_budget",
            )
        with sb2:
            new_kelly = st.select_slider(
                "Kelly Fraction",
                options=[0.25, 0.5, 0.75, 1.0],
                value=kelly_frac, key="port_kelly",
                format_func=lambda x: f"{int(x*100)}%",
            )
        with sb3:
            new_max_stake = st.slider(
                "Max Stake (% of bankroll)", 5, 50,
                int(settings.get("max_stake_pct", 0.33) * 100),
                key="port_maxstake",
                help="Hard cap on a single bet. WF-validated saturation: 33%.",
            )

        # ── Section: Edge Gates ────────────────────────────────────────
        _section_header("🎯  Edge Gates",
                        "Default thresholds — per-market overrides set in Markets section below")
        eg1, eg2, eg3 = st.columns([2, 2, 3])
        with eg1:
            new_min_ev = st.slider("Min EV (default, %)", 1, 60,
                int(min_ev * 100), key="port_minev")
        with eg2:
            new_min_prob = st.slider("Min Prob Gate (%)", 10, 80,
                int(settings.get("min_prob", 0.30) * 100), key="port_minprob",
                help="Reject long-shots even with +EV. WF-validated optimum: 30%.")
        with eg3:
            st.markdown(
                f'<div style="font-size:0.78rem;color:#8892a4;padding-top:0.5rem;line-height:1.5">'
                f'📋 A bet must clear: probability ≥ <b style="color:#a78bfa">{new_min_prob}%</b> '
                f'AND EV ≥ <b style="color:#a78bfa">+{new_min_ev}%</b> '
                f'(unless market_gates overrides for U2.5/etc.)</div>',
                unsafe_allow_html=True,
            )

        # ── Section: Markets ───────────────────────────────────────────
        _section_header("📊  Markets",
                        "Which markets auto-bet considers + per-market gates")
        _all_markets = {"D": "Draw", "over25": "Over 2.5", "under25": "Under 2.5",
                        "H": "Home Win", "A": "Away Win"}
        _current_auto_mkts = settings.get("auto_markets", list(pf.PROFITABLE_MARKETS))
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

        _existing_gates = settings.get("market_gates") or {}
        new_u25_gates = st.toggle(
            "🎚️ Use separate Under 2.5 gates (mp=50%, mev=5%)",
            value="under25" in _existing_gates,
            key="port_u25_gates",
            help=("Adds U2.5 to auto-bet with its own thresholds — raw U2.5 wins "
                  "~52% but its EV distribution is much tighter than draws; the "
                  "40% global EV gate would exclude all U2.5 candidates."),
        )

        # ── Section: Filters ───────────────────────────────────────────
        _section_header("🚫  Filters",
                        "Skip systematically losing patterns identified in walk-forward")
        f1, f2, _ = st.columns([1, 1.5, 2])
        with f1:
            new_skip_late = st.toggle(
                "Skip Mar-Apr",
                value=bool(settings.get("skip_late_season", True)),
                key="port_skip_late",
                help="0/7 wins across 2024-25 + 2025-26 in March-April. "
                     "(May was previously bundled in but is now allowed.)",
            )
        with f2:
            new_skip_title = st.toggle(
                "Skip home_title_race",
                value=bool(settings.get("skip_home_title_race", False)),
                key="port_skip_title",
                help="Title-chasing home teams almost never draw.",
            )

        # ── Section: Optional Phase 4 / multi-season filters (Main) ─────
        _section_header("🧠  Optional escape-hatch filters (defaults OFF)",
                        "Toggle ON if you want to trade peak profit for floor protection. "
                        "Current Main produces £85k 2025-26 peak with all OFF; "
                        "ELO floor 1500 caps that to £35k but never crashes.")
        _all_dows_m   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _all_months_m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        mf1, mf2 = st.columns(2)
        with mf1:
            new_main_banned_dows = st.multiselect(
                "Banned days of week",
                options=_all_dows_m,
                default=list(settings.get("main_banned_dows", [])),
                key="port_main_banned_dows",
                help="Mon+Fri were validated as losing on Mock Two but didn't "
                     "transfer cleanly to Main on multi-season; OFF by default.",
            )
            new_main_banned_months = st.multiselect(
                "Banned months",
                options=_all_months_m,
                default=list(settings.get("main_banned_months", [])),
                key="port_main_banned_months",
                help="Octoberban was overfit; left as opt-in.",
            )
        with mf2:
            _main_max_ev_cur = settings.get("main_max_ev_pct")
            new_main_max_ev_on = st.toggle(
                "Cap claimed EV (overconfidence guard)",
                value=_main_max_ev_cur is not None,
                key="port_main_max_ev_on",
                help="Phase 1 found high-EV bets calibrate worst.",
            )
            new_main_max_ev = (
                st.slider("Max EV cap (%)", 40, 200,
                          int((_main_max_ev_cur or 1.0) * 100), 5,
                          key="port_main_max_ev_val")
                if new_main_max_ev_on else None
            )

        # Main ELO-profile filter
        st.markdown(
            '<div style="margin-top:0.7rem;padding-top:0.5rem;'
            'border-top:1px dashed rgba(0,229,255,0.20);'
            'font-size:0.78rem;font-weight:800;letter-spacing:1.4px;'
            'text-transform:uppercase;color:#00e5ff">'
            '🎯 ELO-profile filter (Mock Two grid winner: min team ELO 1500)'
            '</div>', unsafe_allow_html=True,
        )
        mec1, mec2 = st.columns(2)
        with mec1:
            _mte_cur = settings.get("main_min_team_elo")
            new_main_min_te_on = st.toggle(
                "Min team ELO floor",
                value=_mte_cur is not None,
                key="port_main_min_te_on",
                help="Skip if either team's ELO is below this. Mock Two grid "
                     "winner: 1500. Caps Main's £85k peak at ~£35k but eliminates crashes.",
            )
            new_main_min_te = (
                st.slider("Min ELO", 1300, 1700, int(_mte_cur or 1500), 10,
                          key="port_main_min_te_val")
                if new_main_min_te_on else None
            )
            _mgmin_cur = settings.get("main_elo_gap_min")
            new_main_gap_min_on = st.toggle(
                "Min |ΔELO| (skip too-close)",
                value=_mgmin_cur is not None, key="port_main_gap_min_on",
            )
            new_main_gap_min = (
                st.slider("Min gap", 20, 200, int(_mgmin_cur or 80), 10,
                          key="port_main_gap_min_val")
                if new_main_gap_min_on else None
            )
        with mec2:
            _mxte_cur = settings.get("main_max_team_elo")
            new_main_max_te_on = st.toggle(
                "Max team ELO ceiling",
                value=_mxte_cur is not None, key="port_main_max_te_on",
            )
            new_main_max_te = (
                st.slider("Max ELO", 1700, 2100, int(_mxte_cur or 1900), 10,
                          key="port_main_max_te_val")
                if new_main_max_te_on else None
            )
            _mgmax_cur = settings.get("main_elo_gap_max")
            new_main_gap_max_on = st.toggle(
                "Max |ΔELO| (skip lopsided)",
                value=_mgmax_cur is not None, key="port_main_gap_max_on",
            )
            new_main_gap_max = (
                st.slider("Max gap", 100, 500, int(_mgmax_cur or 300), 20,
                          key="port_main_gap_max_val")
                if new_main_gap_max_on else None
            )

        # ── Section: Auto-Bet ──────────────────────────────────────────
        _section_header("🤖  Auto-Bet",
                        "Automatic placement when gates and market filter pass")
        ab1, ab2 = st.columns(2)
        with ab1:
            new_auto_enabled = st.toggle(
                "Auto-Bet enabled",
                value=settings.get("auto_bet_enabled", False),
                key="port_auto_enabled",
                help="Automatically place Kelly bets when all gates pass",
            )
        with ab2:
            new_auto_thresh = st.slider(
                "Auto-Bet EV Threshold (%)", 1, 60,
                int(settings.get("auto_bet_threshold", 0.40) * 100),
                key="port_auto_thresh",
                help="Only auto-bet when model EV is this high. Optimum: 40%.",
            )
        new_use_sim = st.toggle(
            "Sim-bet correction (reduce stakes when ≥2 bets settle same day)",
            value=bool(settings.get("use_simultaneous_kelly", True)),
            key="port_use_sim",
            help="Mock Two has it built-in; this enables on Main.",
        )
        if new_auto_enabled:
            st.markdown(
                f'<div style="padding:0.5rem 0.8rem;background:rgba(0,230,118,0.06);'
                f'border-left:3px solid #00e676;border-radius:6px;font-size:0.78rem;color:#e8eaf0">'
                f'✅ Auto-bet ON — will place Kelly bets when EV ≥ +{new_auto_thresh}%, '
                f'prob ≥ {new_min_prob}% (or per-market overrides), market in '
                f'{{{", ".join(_new_auto_mkts) or "(none)"}}}, plus filters.</div>',
                unsafe_allow_html=True,
            )

        # ── Section: Calibration ───────────────────────────────────────
        _section_header("📐  Calibration", "Probability adjustment from backtest fit")
        new_use_cal = st.toggle(
            "Apply isotonic probability calibration",
            value=settings.get("use_calibrated_probs", True),
            key="port_use_cal",
            help="Fit on backtest data — corrects systematic over/under-confidence.",
        )

        # ── Section: API ───────────────────────────────────────────────
        _section_header("🔌  Live Odds API",
                        "The Odds API key (free tier — 500 req/month)")
        api_c1, api_c2 = st.columns([2, 3])
        with api_c1:
            new_api_key = st.text_input(
                "API Key", value=api_key, type="password", key="port_apikey",
                help="Free key from the-odds-api.com",
                label_visibility="collapsed",
            )
        with api_c2:
            _usage = pf.get_api_usage()
            _usage_pct = _usage["count"] / _usage["cap"] * 100 if _usage["cap"] > 0 else 0
            _usage_col = "#00e676" if _usage_pct < 50 else ("#ffd600" if _usage_pct < 80 else "#ff4081")
            st.markdown(
                f'<div style="font-size:0.78rem;color:#8892a4;padding-top:0.5rem">'
                f'<span style="color:{_usage_col};font-weight:700">'
                f'📊 {_usage["count"]}/{_usage["cap"]} calls this month '
                f'({_usage["remaining"]} left)</span> · Cache: {pf._CACHE_HOURS}h</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
        save_c, reset_c, _ = st.columns([1, 1, 4])
        with save_c:
            if st.button("💾 Save Settings", key="save_port_settings"):
                port["settings"]["min_ev"]              = new_min_ev / 100
                port["settings"]["kelly_fraction"]      = new_kelly
                port["settings"]["odds_api_key"]        = new_api_key
                port["settings"]["auto_bet_enabled"]    = new_auto_enabled
                port["settings"]["auto_bet_threshold"]  = new_auto_thresh / 100
                port["settings"]["auto_markets"]        = _new_auto_mkts
                port["settings"]["min_prob"]            = new_min_prob / 100
                port["settings"]["use_calibrated_probs"] = new_use_cal
                port["settings"]["max_stake_pct"]       = new_max_stake / 100
                port["settings"]["skip_late_season"]    = new_skip_late
                port["settings"]["skip_home_title_race"] = new_skip_title
                port["settings"]["use_simultaneous_kelly"] = new_use_sim
                # Optional Main filters (default OFF — escape hatches)
                port["settings"]["main_banned_dows"]   = list(new_main_banned_dows)
                port["settings"]["main_banned_months"] = list(new_main_banned_months)
                port["settings"]["main_max_ev_pct"]    = (
                    float(new_main_max_ev) / 100 if new_main_max_ev is not None else None
                )
                port["settings"]["main_min_team_elo"]  = (
                    float(new_main_min_te) if new_main_min_te is not None else None
                )
                port["settings"]["main_max_team_elo"]  = (
                    float(new_main_max_te) if new_main_max_te is not None else None
                )
                port["settings"]["main_elo_gap_min"]   = (
                    float(new_main_gap_min) if new_main_gap_min is not None else None
                )
                port["settings"]["main_elo_gap_max"]   = (
                    float(new_main_gap_max) if new_main_gap_max is not None else None
                )
                # Per-market gates: when U2.5 is enabled, register its own gates
                # AND add it to the auto_markets list so it can clear the market filter.
                if new_u25_gates:
                    port["settings"]["market_gates"] = {
                        "under25": {"min_prob": 0.50, "min_ev": 0.05},
                    }
                    if "under25" not in _new_auto_mkts:
                        _new_auto_mkts.append("under25")
                    port["settings"]["auto_markets"] = _new_auto_mkts
                else:
                    port["settings"].pop("market_gates", None)
                # Only rebase an untouched portfolio. This used to check for
                # settled bets alone, so saving settings with money still on
                # the table reset the bankroll and silently refunded every
                # pending stake — observed 2026-08-09, £1,042.64 handed back
                # while the bet was still live.
                if not port["bets"]:
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
                _pinn  = _api_o.get("_pinnacle") if isinstance(_api_o, dict) else None
                for _mkt, _prob, _lbl in [
                    ("H",      _res["home_win"], f"Home Win ({_h})"),
                    ("D",      _res["draw"],     "Draw"),
                    ("A",      _res["away_win"], f"Away Win ({_a})"),
                    ("over25", _p_o25,           "Over 2.5 Goals"),
                    ("under25",1 - _p_o25,       "Under 2.5 Goals"),
                ]:
                    _place_odds  = _api_o.get(_mkt)
                    _detect_odds = _pinn.get(_mkt) if _pinn else None
                    _ev_ref = _detect_odds if (_detect_odds and _detect_odds > 1) else _place_odds
                    _candidates.append({
                        "home": _h, "away": _a, "date": _ds,
                        "market": _mkt, "selection": _lbl,
                        "model_prob": _prob,
                        "odds": _place_odds,
                        "detect_odds": _detect_odds,
                        "home_elo": elo_dict.get(_h),
                        "away_elo": elo_dict.get(_a),
                        "ev": pf.compute_ev(_prob, _ev_ref) if _ev_ref else -1,
                    })
            _auto_placed = pf.auto_place_value_bets(
                port, _candidates, auto_threshold, calibrators=calibrators,
            )
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

    # ── Story-driven journey: Started → Now → Potential ──────────────────
    initial = port["initial_bankroll"]
    wealth_now = bankroll + pending_stake  # cash + capital locked in pending
    # Best case: all pending win → bankroll + sum(stake * odds)
    best_case  = bankroll + sum(b["stake"] * b["odds"] for b in pending_bets)
    # Worst case: all pending lose → bankroll stays as-is (stakes already deducted)
    worst_case = bankroll
    profit_pct = (profit / initial * 100) if initial else 0
    delta_col  = "#00e676" if profit >= 0 else "#ff4081"
    delta_arrow = "▲" if profit >= 0 else "▼"

    st.markdown(f"""
    <div class="pnl-hero {hero_class}">
        <div class="pnl-tag">📊 MOCK PORTFOLIO · PAPER BETS ONLY · NOT REAL MONEY</div>
        <div class="pnl-amount">{sign}£{abs(profit):,.2f}</div>
        <div class="pnl-subtitle">
            {arrow} {sign}{roi:.1f}% ROI on settled bets
        </div>
    </div>

    <div class="pnl-journey">
      <div class="pj-step pj-start">
        <div class="pj-lbl">STARTED WITH</div>
        <div class="pj-val">£{initial:,.0f}</div>
        <div class="pj-sub">initial bankroll</div>
      </div>
      <div class="pj-arrow"><span style="color:{delta_col}">{delta_arrow}</span>
        <span class="pj-arrow-sub" style="color:{delta_col}">
          {sign}£{abs(profit):,.0f}<br>({sign}{profit_pct:.1f}%)
        </span>
      </div>
      <div class="pj-step pj-now">
        <div class="pj-lbl">CURRENT WEALTH</div>
        <div class="pj-val">£{wealth_now:,.0f}</div>
        <div class="pj-sub">
          £{bankroll:,.0f} cash &nbsp;+&nbsp;
          <span style="color:#ffd600">£{pending_stake:,.0f} pending</span>
        </div>
      </div>
      <div class="pj-arrow pj-arrow-future">
        <div class="pj-future-lbl">{stats['n_pending']} bet{'s' if stats['n_pending']!=1 else ''} could resolve to</div>
      </div>
      <div class="pj-step pj-future">
        <div class="pj-future-row">
          <div class="pj-future-half pj-future-win">
            <div class="pj-future-lbl-small">IF ALL WIN</div>
            <div class="pj-future-val pj-future-win-val">£{best_case:,.0f}</div>
          </div>
          <div class="pj-future-half pj-future-lose">
            <div class="pj-future-lbl-small">IF ALL LOSE</div>
            <div class="pj-future-val pj-future-lose-val">£{worst_case:,.0f}</div>
          </div>
        </div>
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

    # ── Bankroll chart (full width) ───────────────────────────────────────
    st.markdown('<p class="section-label">📈  BANKROLL HISTORY</p>', unsafe_allow_html=True)
    history = pf.bankroll_history(port)
    init_br = port["initial_bankroll"]
    if len(history) > 1:
        y_vals  = history["bankroll"].tolist()
        x_vals  = list(range(len(y_vals)))
        labels  = history["match"].tolist()
    else:
        # No settled bets yet — show starting bankroll + pending stakes
        y_vals = [init_br]
        labels = ["Start"]
        running = init_br
        for pb in sorted(pending_bets, key=lambda b: b.get("placed_at", "")):
            running -= pb["stake"]
            y_vals.append(round(running, 2))
            if pb.get("type") == "acca":
                lbl = f"ACCA £{pb['stake']:.0f}"
            else:
                lbl = f"{pb.get('selection', '?')} £{pb['stake']:.0f}"
            labels.append(lbl)
        if pending_bets:
            best_return = running + sum(
                pb["stake"] * pb["odds"] for pb in pending_bets
            )
            y_vals.append(round(best_return, 2))
            labels.append("If all win")
        x_vals = list(range(len(y_vals)))

    line_col = "#00e676" if y_vals[-1] >= init_br else "#ff4081"
    fig_br = go.Figure()

    if len(history) > 1:
        # ── Red/green fill zones around the initial-bankroll baseline ──
        # Insert interpolated points wherever the bankroll line CROSSES the
        # baseline. Without this, fills bleed past the actual line because
        # Plotly draws straight segments between data points.
        x_exp: list[float] = [x_vals[0]]
        y_exp: list[float] = [y_vals[0]]
        for i in range(1, len(y_vals)):
            y_prev, y_cur = y_vals[i - 1], y_vals[i]
            # Sign change relative to baseline → insert crossing point
            if (y_prev - init_br) * (y_cur - init_br) < 0:
                t = (init_br - y_prev) / (y_cur - y_prev)
                x_cross = x_vals[i - 1] + t * (x_vals[i] - x_vals[i - 1])
                x_exp.append(x_cross); y_exp.append(init_br)
            x_exp.append(x_vals[i]); y_exp.append(y_cur)

        y_up = [max(v, init_br) for v in y_exp]
        y_dn = [min(v, init_br) for v in y_exp]
        baseline = [init_br] * len(y_exp)

        # Baseline (invisible anchor for green fill)
        fig_br.add_trace(go.Scatter(
            x=x_exp, y=baseline,
            mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        # Green: from baseline up to bankroll curve where bankroll ≥ baseline
        fig_br.add_trace(go.Scatter(
            x=x_exp, y=y_up,
            mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor="rgba(0,230,118,0.22)",
            hoverinfo="skip", showlegend=False,
        ))
        # Baseline again (anchor for red fill)
        fig_br.add_trace(go.Scatter(
            x=x_exp, y=baseline,
            mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        # Red: from baseline down to bankroll curve where bankroll ≤ baseline
        fig_br.add_trace(go.Scatter(
            x=x_exp, y=y_dn,
            mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor="rgba(255,64,129,0.22)",
            hoverinfo="skip", showlegend=False,
        ))

        # Baseline line on top of fills
        fig_br.add_hline(
            y=init_br, line_color="rgba(255,255,255,0.30)", line_dash="dot",
            annotation_text=f"Start £{init_br:,.0f}",
            annotation_font=dict(color="#8892a4", size=14, family="Inter"),
            annotation_position="top left",
        )

        # Main bankroll line with direction arrows per bet
        # (triangle-up = win / bankroll rose, triangle-down = loss / fell)
        deltas = [0.0] + [y_vals[i] - y_vals[i - 1] for i in range(1, len(y_vals))]
        marker_symbols = ["circle"] + [
            "triangle-up" if d > 0 else ("triangle-down" if d < 0 else "circle-open")
            for d in deltas[1:]
        ]
        marker_colors = ["#8892a4"] + [
            "#00e676" if d > 0 else ("#ff4081" if d < 0 else "#8892a4")
            for d in deltas[1:]
        ]
        marker_sizes = [0] + [13 if d != 0 else 6 for d in deltas[1:]]
        # Hover label tells you won/lost and delta
        hover_texts = ["Start"] + [
            ("▲ WON "  + f"+£{d:,.2f}") if d > 0 else
            ("▼ LOST " + f"−£{abs(d):,.2f}") if d < 0 else
            "No change"
            for d in deltas[1:]
        ]

        # Soft halo line beneath the main line for a subtle glow effect
        fig_br.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            line=dict(color=f"rgba({_hex_to_rgb(line_col)},0.30)", width=10),
            hoverinfo="skip", showlegend=False,
        ))
        fig_br.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            line=dict(color=line_col, width=3.5, shape="linear"),
            marker=dict(
                symbol=marker_symbols,
                size=marker_sizes,
                color=marker_colors,
                line=dict(color="#0a0e1a", width=1.5),
            ),
            text=hover_texts,
            hovertemplate=("<b>Bet %{x}</b><br>%{text}<br>"
                           "<b style='font-size:14px'>Bankroll £%{y:,.2f}</b><extra></extra>"),
            hoverlabel=dict(
                bgcolor="#1a1d27",
                bordercolor=line_col,
                font=dict(size=14, family="Inter", color="#e8eaf0"),
            ),
            showlegend=False,
        ))

        # ── Peak / trough / current annotations ───────────────────────
        peak_idx   = int(np.argmax(y_vals))
        trough_idx = int(np.argmin(y_vals))
        # Peak marker (only annotate if it's not the start AND meaningfully above start)
        if peak_idx > 0 and y_vals[peak_idx] > init_br * 1.05:
            fig_br.add_annotation(
                x=x_vals[peak_idx], y=y_vals[peak_idx],
                text=f"<b>Peak</b><br>£{y_vals[peak_idx]:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor="#00e676",
                arrowsize=1.2, arrowwidth=1.5,
                ax=0, ay=-38,
                font=dict(size=12, color="#00e676", family="Inter"),
                bgcolor="rgba(0,230,118,0.10)",
                bordercolor="rgba(0,230,118,0.4)",
                borderpad=4, borderwidth=1,
            )
        if trough_idx > 0 and y_vals[trough_idx] < init_br * 0.95 and trough_idx != peak_idx:
            fig_br.add_annotation(
                x=x_vals[trough_idx], y=y_vals[trough_idx],
                text=f"<b>Low</b><br>£{y_vals[trough_idx]:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor="#ff4081",
                arrowsize=1.2, arrowwidth=1.5,
                ax=0, ay=38,
                font=dict(size=12, color="#ff4081", family="Inter"),
                bgcolor="rgba(255,64,129,0.10)",
                bordercolor="rgba(255,64,129,0.4)",
                borderpad=4, borderwidth=1,
            )
        # Current bankroll badge at the right edge
        fig_br.add_annotation(
            x=x_vals[-1], y=y_vals[-1],
            text=f"<b>NOW · £{y_vals[-1]:,.0f}</b>",
            showarrow=False, xshift=15,
            font=dict(size=14, color="#fff", family="Inter"),
            bgcolor=line_col,
            bordercolor=line_col,
            borderpad=8, borderwidth=2,
            xanchor="left",
        )

        # ── Pending bets projection ───────────────────────────────────
        # Three dotted forks extending from the LAST SETTLED bet point:
        #   - Best case (all pending win)        → green
        #   - Expected case (model-weighted)     → cyan
        #   - Worst case (all lose)              → pink
        #
        # Visually attaching to the last settled bet is more intuitive than
        # attaching to "now" (which already has pending stakes deducted, so
        # the curves would start lower than they should). The TERMINAL
        # values (final if-all-win / expected / if-all-lose) are identical
        # to the old math — just expressed as profit-from-last-settled
        # rather than gross-return-from-current-cash.
        if pending_bets:
            n_pending = len(pending_bets)

            pending_sorted = sorted(
                pending_bets,
                key=lambda b: (b.get("date") or "", b.get("placed_at") or ""),
            )

            # Baseline = bankroll BEFORE pending stakes were deducted.
            # = current cash (`bankroll`) + sum of pending stakes still at risk.
            total_pending_stake = sum(float(b["stake"]) for b in pending_sorted)
            last_settled_br = bankroll + total_pending_stake

            x_proj = list(range(x_vals[-1], x_vals[-1] + n_pending + 1))
            best_y  = [last_settled_br]
            exp_y   = [last_settled_br]
            worst_y = [last_settled_br]
            for b in pending_sorted:
                stake = float(b["stake"])
                odds  = float(b["odds"])
                p_win = float(b.get("model_prob") or 0.0)
                # Profit if bet wins  = stake * (odds - 1)
                # Loss if bet loses   = -stake
                profit_win  = stake * (odds - 1)
                profit_loss = -stake
                exp_change  = p_win * profit_win + (1.0 - p_win) * profit_loss
                best_y.append(round(best_y[-1]   + profit_win,  2))
                worst_y.append(round(worst_y[-1] + profit_loss, 2))
                exp_y.append (round(exp_y[-1]    + exp_change,  2))

            # Best case (all pending win)
            fig_br.add_trace(go.Scatter(
                x=x_proj, y=best_y,
                mode="lines+markers",
                line=dict(color="#00e676", width=2.2, dash="dot"),
                marker=dict(size=[0] + [8] * n_pending,
                            color="#00e676", symbol="diamond"),
                hovertemplate=("<b>Pending #%{x}</b><br>"
                               "If all wins from here → £%{y:,.2f}<extra></extra>"),
                name="If all pending WIN",
                showlegend=False,
            ))

            # Expected case (probability-weighted)
            fig_br.add_trace(go.Scatter(
                x=x_proj, y=exp_y,
                mode="lines+markers",
                line=dict(color="#00e5ff", width=2.5, dash="dash"),
                marker=dict(size=[0] + [8] * n_pending,
                            color="#00e5ff", symbol="circle"),
                hovertemplate=("<b>Pending #%{x}</b><br>"
                               "Expected (model-weighted) → £%{y:,.2f}<extra></extra>"),
                name="Expected (model-weighted)",
                showlegend=False,
            ))

            # Worst case (all lose) — flat at current bankroll
            fig_br.add_trace(go.Scatter(
                x=x_proj, y=worst_y,
                mode="lines+markers",
                line=dict(color="#ff4081", width=2.2, dash="dot"),
                marker=dict(size=[0] + [8] * n_pending,
                            color="#ff4081", symbol="x"),
                hovertemplate=("<b>Pending #%{x}</b><br>"
                               "If all losses from here → £%{y:,.2f}<extra></extra>"),
                name="If all pending LOSE",
                showlegend=False,
            ))

            # Endpoint annotations — small chips at the right end of each line
            fig_br.add_annotation(
                x=x_proj[-1], y=best_y[-1],
                text=f"<b>If all win<br>£{best_y[-1]:,.0f}</b>",
                showarrow=False, xshift=12,
                font=dict(size=12, color="#00e676", family="Inter"),
                bgcolor="rgba(0,230,118,0.10)",
                bordercolor="rgba(0,230,118,0.4)",
                borderpad=5, borderwidth=1,
                xanchor="left",
            )
            fig_br.add_annotation(
                x=x_proj[-1], y=exp_y[-1],
                text=f"<b>Expected<br>£{exp_y[-1]:,.0f}</b>",
                showarrow=False, xshift=12,
                font=dict(size=12, color="#00e5ff", family="Inter"),
                bgcolor="rgba(0,229,255,0.10)",
                bordercolor="rgba(0,229,255,0.4)",
                borderpad=5, borderwidth=1,
                xanchor="left",
            )
            fig_br.add_annotation(
                x=x_proj[-1], y=worst_y[-1],
                text=f"<b>If all lose<br>£{worst_y[-1]:,.0f}</b>",
                showarrow=False, xshift=12,
                font=dict(size=12, color="#ff4081", family="Inter"),
                bgcolor="rgba(255,64,129,0.10)",
                bordercolor="rgba(255,64,129,0.4)",
                borderpad=5, borderwidth=1,
                xanchor="left",
            )

            # Vertical separator between settled and pending
            fig_br.add_vline(
                x=x_vals[-1], line_color="rgba(255,255,255,0.15)",
                line_dash="dot",
            )
            fig_br.add_annotation(
                x=x_vals[-1], y=1.0, yref="paper",
                text="settled  →  pending",
                showarrow=False, yshift=-6,
                font=dict(size=11, color="#7c4dff", family="Inter"),
            )
    else:
        # Pending-only view — dashed line with labeled dots
        fig_br.add_hline(
            y=init_br, line_color="rgba(255,255,255,0.25)", line_dash="dot",
            annotation_text=f"Start £{init_br:,.0f}",
            annotation_font=dict(color="#8892a4", size=11),
            annotation_position="top left",
        )
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
            textfont=dict(size=10, color="#8892a4"),
            hovertemplate="%{text}<br>£%{y:,.2f}<extra></extra>",
            name="Staked",
        ))
        if n_pts > 2:
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
        height=520, showlegend=False,
        margin=dict(t=40, b=40, l=20, r=140),
        xaxis=dict(
            title=dict(text="BET NUMBER", font=dict(size=12, color="#7c4dff", family="Inter"),
                       standoff=18),
            showgrid=False, showticklabels=True,
            tickfont=dict(size=13, color="#8892a4", family="Inter"),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="BANKROLL", font=dict(size=12, color="#7c4dff", family="Inter"),
                       standoff=14),
            gridcolor="rgba(255,255,255,0.05)",
            tickprefix="£",
            tickfont=dict(size=14, color="#cdd", family="Inter"),
            zeroline=False,
            tickformat=",.0f",
        ),
    )
    st.plotly_chart(fig_br, use_container_width=True, config={"displayModeBar": False})

    # ── Pending bets — singles only; accas live in their own section below ──
    pending_singles = [b for b in pending_bets if b.get("type") != "acca"]
    pending_accas   = [b for b in pending_bets if b.get("type") == "acca"]

    st.markdown('<div class="divider" style="margin:1.5rem 0 1rem"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">⏳  PENDING BETS · Singles</p>', unsafe_allow_html=True)

    if pending_singles:
        mkt_colors = {"H": "#3d6eff", "D": "#ffd600", "A": "#ff4081",
                      "over25": "#7c4dff", "under25": "#00e5ff"}

        def _render_pending_card(bet: dict) -> None:
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
                        <span style="color:#a78bfa">Model {bet.get('combined_model_prob', 0)*100:.1f}%</span>
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
                # Short date format like bet history
                try:
                    _d = pd.to_datetime(bet.get("date") or bet.get("placed_at"))
                    _date_short = _d.strftime("%d-%b-%y")
                except Exception:
                    _date_short = (bet.get("date") or "")[:10]
                _model_p = bet.get("model_prob", 0) * 100
                _profit_pct = round((_pot_profit / max(bet['stake'], 0.01)) * 100, 0)
                st.markdown(f"""
                <div class="pend-card-v2">
                    <div class="pend-v2-top">
                        <span class="pend-v2-date">📅 {_date_short}</span>
                        <span class="pend-v2-badge" style="color:{mc};border-color:{mc}55;background:rgba({_hex_to_rgb(mc)},0.10)">{bet['market'].upper()}</span>
                    </div>
                    <div class="pend-v2-teams">
                        <div class="pend-v2-team"><img class="team-badge" src="{_BADGE_URL.get(bet['home'], '')}" width="56" /><span>{bet['home']}</span></div>
                        <div class="pend-v2-vs">VS</div>
                        <div class="pend-v2-team"><img class="team-badge" src="{_BADGE_URL.get(bet['away'], '')}" width="56" /><span>{bet['away']}</span></div>
                    </div>
                    <div class="pend-v2-pick" style="color:{mc}">
                        🎯 {bet['selection']} <span class="pend-v2-at">@</span> <span class="pend-v2-odds">{bet['odds']:.2f}</span>
                    </div>
                    <div class="pend-v2-stats">
                        <div class="pend-v2-stat">
                            <div class="pend-v2-stat-lbl">STAKE</div>
                            <div class="pend-v2-stat-val">£{bet['stake']:,.2f}</div>
                        </div>
                        <div class="pend-v2-stat">
                            <div class="pend-v2-stat-lbl">MODEL</div>
                            <div class="pend-v2-stat-val pend-v2-model">{_model_p:.1f}%</div>
                        </div>
                        <div class="pend-v2-stat">
                            <div class="pend-v2-stat-lbl">EV</div>
                            <div class="pend-v2-stat-val pend-v2-ev">+{ev_pct:.1f}%</div>
                        </div>
                    </div>
                    <div class="pend-v2-return">
                        <div class="pend-v2-return-block">
                            <div class="pend-v2-return-lbl">IF IT WINS</div>
                            <div class="pend-v2-return-val">£{_pot_ret:,.2f}</div>
                        </div>
                        <div class="pend-v2-arrow">→</div>
                        <div class="pend-v2-return-block">
                            <div class="pend-v2-return-lbl">PROFIT</div>
                            <div class="pend-v2-return-profit">+£{_pot_profit:,.2f}</div>
                            <div class="pend-v2-return-pct">+{_profit_pct:.0f}% on stake</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            if st.button("✕ Cancel", key=f"cancel_{bet['id']}", type="secondary"):
                pf.remove_pending_bet(port, bet["id"])
                pf.save_portfolio(port)
                st.rerun()

        # Most-recent 8 single bets, rendered two per row
        recent = pending_singles[-8:]
        for i in range(0, len(recent), 2):
            cols = st.columns(2)
            with cols[0]:
                _render_pending_card(recent[i])
            if i + 1 < len(recent):
                with cols[1]:
                    _render_pending_card(recent[i + 1])
    else:
        st.markdown(
            '<p style="color:#9aa6ba;font-size:0.9rem;text-align:center;padding:2.5rem 0">'
            'No pending single bets</p>',
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
                    f'<p style="font-size:0.82rem;font-weight:700;color:#b8c0d0;'
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
                        f'<div class="scan-value-banner">'
                        f'  <div class="svb-icon">💎</div>'
                        f'  <div class="svb-block">'
                        f'    <div class="svb-lbl">VALUE PICK</div>'
                        f'    <div class="svb-val">{label} @ {o_val:.2f}</div>'
                        f'  </div>'
                        f'  <div class="svb-block">'
                        f'    <div class="svb-lbl">EDGE vs MARKET</div>'
                        f'    <div class="svb-val svb-edge">+{ev_val*100:.1f}%</div>'
                        f'  </div>'
                        f'  <div class="svb-block">'
                        f'    <div class="svb-lbl">KELLY STAKE</div>'
                        f'    <div class="svb-val svb-stake">£{kelly_rec:,.0f}</div>'
                        f'  </div>'
                        f'  <div class="svb-block">'
                        f'    <div class="svb-lbl">IF IT WINS</div>'
                        f'    <div class="svb-val svb-win">£{_scan_ret:,.0f}</div>'
                        f'  </div>'
                        f'</div>'
                    )

            # Use the same fixture card visual as the Weekend tab — big badges
            # and the gradient HDA bar. Keeps Mock Portfolio visually consistent
            # with the rest of the app and lifts the team identity above the
            # "what to bet" decision.
            _result_dict = {"home_win": p_h, "draw": p_d, "away_win": p_a}
            _h_pct = round(p_h * 100, 1)
            _d_pct = round(p_d * 100, 1)
            _a_pct = round(p_a * 100, 1)
            _ou_pct = round(p_o25 * 100, 1)
            _u_pct = 100 - _ou_pct
            _src_chip = ('<span class="scan-src-chip scan-src-live">📡 Live odds</span>'
                         if has_api
                         else '<span class="scan-src-chip">✍️ Manual odds</span>')
            st.markdown(f"""
            <div class="scan-fixture-card">
                <div class="scan-fixture-row">
                    <div class="fixture-team-name fixture-team-home">{tb(home, 56)}</div>
                    <div class="fixture-prob-bar" style="height:48px">
                        <div class="bar-home" style="width:{_h_pct}%">{_h_pct:.0f}%</div>
                        <div class="bar-draw" style="width:{_d_pct}%">{_d_pct:.0f}%</div>
                        <div class="bar-away" style="width:{_a_pct}%">{_a_pct:.0f}%</div>
                    </div>
                    <div class="fixture-team-name fixture-team-away">{tb(away, 56)}</div>
                </div>
                <div class="scan-ou-row">
                    <span class="scan-ou-lbl">Goals (O/U 2.5)</span>
                    <div class="scan-ou-bar">
                        <div class="scan-ou-over"  style="width:{_ou_pct}%">Over {_ou_pct:.0f}%</div>
                        <div class="scan-ou-under" style="width:{_u_pct}%">Under {_u_pct:.0f}%</div>
                    </div>
                    {_src_chip}
                </div>
            </div>
            {value_html}""", unsafe_allow_html=True)

            # Bet slip expander — bigger, more prominent label
            with st.expander(f"💰  Place a bet — {home} vs {away}", expanded=False):
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
                        <div style="font-size:0.78rem;color:#b8c0d0;text-transform:uppercase;letter-spacing:1px">Model EV</div>
                        <div style="font-size:1.5rem;font-weight:800;color:{ev_color}">
                            {"+" if ev_val >= 0 else ""}{ev_val*100:.1f}%
                        </div>
                        <div style="font-size:0.78rem;color:#b8c0d0">Kelly: £{kelly_rec:.0f}</div>
                        <div style="font-size:0.78rem;color:#b8c0d0">Implied: {pf.implied_prob(odds_inp)*100:.1f}%</div>
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
                        f'<div style="font-size:0.88rem;color:#b8c0d0;text-transform:uppercase;letter-spacing:1px">Potential Return</div>'
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

    # ─────────────────────────────────────────────────────────────────────
    # Accumulator Bets — collapsed by default, separate from singles
    # ─────────────────────────────────────────────────────────────────────
    _n_acca_pending = len(pending_accas)
    _n_acca_settled = len([b for b in port["bets"]
                           if b.get("type") == "acca" and b["status"] in ("won", "lost")])
    _acca_label = (f"🎲  Accumulator Bets  ·  "
                   f"{_n_acca_pending} pending · {_n_acca_settled} settled  ·  "
                   f"builder + acca-only history")
    with st.expander(_acca_label, expanded=False):
        st.markdown(
            '<div style="background:rgba(124,77,255,0.06);'
            'border-left:3px solid #7c4dff;padding:0.7rem 1rem;'
            'border-radius:6px;font-size:0.78rem;color:#8892a4;line-height:1.5;'
            'margin-bottom:1rem">'
            '⚠️ Accumulators are kept separate from singles because they are a different '
            'risk profile. The model finds positive-EV combinations, but accas have '
            'historically lost money in backtest (-16% to -46% ROI) since needing all '
            'legs to hit compounds the miss rate. Use sparingly.</div>',
            unsafe_allow_html=True,
        )

        # ── Accumulator Builder ───────────────────────────────────────────
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
                    '<p style="color:#b8c0d0;font-size:0.82rem">Add an Odds API key in Settings to '
                    'generate accumulator suggestions based on live bookmaker odds.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="color:#b8c0d0;font-size:0.82rem">No positive-EV combinations found '
                    'for upcoming fixtures.</p>',
                    unsafe_allow_html=True,
                )
        else:
            doubles  = [s for s in suggestions if s["n_legs"] == 2][:5]
            trebles  = [s for s in suggestions if s["n_legs"] == 3][:5]

            for label, group in [("📗 Top Doubles", doubles), ("📘 Top Trebles", trebles)]:
                if not group:
                    continue
                st.markdown(f'<p style="font-size:0.82rem;font-weight:700;color:#b8c0d0;'
                            f'text-transform:uppercase;letter-spacing:2px;margin:0.8rem 0 0.4rem">'
                            f'{label}</p>', unsafe_allow_html=True)
                for i, sug in enumerate(group):
                    ev_val  = sug["ev"]
                    ev_col  = "#00e676" if ev_val >= min_ev else ("#ffd600" if ev_val >= 0 else "#ff4081")
                    legs_txt = " &nbsp;✕&nbsp; ".join(
                        f'<b style="color:#ccd">{lg["selection"]}</b>'
                        f' <span style="color:#b8c0d0">({tb(lg["home"], 16)} v {tb(lg["away"], 16)})</span>'
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
                            <span style="color:#a78bfa">Model: {sug['combined_prob']*100:.1f}%</span>
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
                            f'<div style="font-size:0.84rem;color:#b8c0d0;text-transform:uppercase;letter-spacing:1px">Returns</div>'
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

        # ── Acca Pending ─────────────────────────────────────────────────
        if pending_accas:
            st.markdown('<div class="divider" style="margin:1.5rem 0 0.8rem"></div>',
                        unsafe_allow_html=True)
            st.markdown('<p class="section-label">⏳  ACCA · PENDING</p>',
                        unsafe_allow_html=True)
            _acca_pend_rows = []
            for b in pending_accas:
                legs = b.get("legs", [])
                _acca_pend_rows.append({
                    "Date":     (b.get("date") or b.get("placed_at", ""))[:10],
                    "Legs":     " + ".join(f"{lg['selection']} ({lg['home']} v {lg['away']})"
                                            for lg in legs),
                    "Odds":     round(float(b.get("odds", 1)), 2),
                    "Stake":    float(b.get("stake", 0)),
                    "Pot.Ret":  round(float(b.get("stake", 0)) * float(b.get("odds", 1)), 2),
                })
            if _acca_pend_rows:
                st.dataframe(pd.DataFrame(_acca_pend_rows),
                             use_container_width=True, hide_index=True,
                             column_config={
                                 "Stake":    st.column_config.NumberColumn("Stake",   format="£%.2f"),
                                 "Odds":     st.column_config.NumberColumn("Odds",    format="%.2f"),
                                 "Pot.Ret":  st.column_config.NumberColumn("If wins", format="£%.2f"),
                             })

        # ── Acca History ─────────────────────────────────────────────────
        _acca_settled = [b for b in port["bets"]
                         if b.get("type") == "acca" and b["status"] in ("won", "lost")]
        if _acca_settled:
            st.markdown('<div class="divider" style="margin:1.5rem 0 0.8rem"></div>',
                        unsafe_allow_html=True)
            st.markdown('<p class="section-label">📋  ACCA · HISTORY</p>',
                        unsafe_allow_html=True)
            _acca_hist_rows = []
            for b in reversed(_acca_settled):
                legs = b.get("legs", [])
                _acca_hist_rows.append({
                    "Date":     (b.get("date") or b.get("placed_at", ""))[:10],
                    "Legs":     " + ".join(f"{lg['selection']}" for lg in legs),
                    "N":        len(legs),
                    "Odds":     round(float(b.get("odds", 1)), 2),
                    "Stake":    float(b.get("stake", 0)),
                    "Result":   "✅ Won" if b["status"] == "won" else "❌ Lost",
                    "P&L":      float(b.get("profit") or 0),
                })
            if _acca_hist_rows:
                _acca_total = sum(r["P&L"] for r in _acca_hist_rows)
                _acca_won   = sum(1 for r in _acca_hist_rows if r["Result"].startswith("✅"))
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#8892a4;margin-bottom:0.5rem">'
                    f'<b style="color:#e8eaf0">{_acca_won}</b> won / '
                    f'<b style="color:#e8eaf0">{len(_acca_hist_rows)}</b> total &nbsp;·&nbsp; '
                    f'Total P&L: <b style="color:'
                    f'{"#00e676" if _acca_total >= 0 else "#ff4081"}">'
                    f'£{_acca_total:+,.2f}</b></div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(_acca_hist_rows),
                             use_container_width=True, hide_index=True,
                             height=min(360, 60 + len(_acca_hist_rows) * 35),
                             column_config={
                                 "Stake":   st.column_config.NumberColumn("Stake",   format="£%.2f"),
                                 "Odds":    st.column_config.NumberColumn("Odds",    format="%.2f"),
                                 "P&L":     st.column_config.NumberColumn("P&L",     format="£%.2f"),
                             })

    # End acca expander — divider before Bet History
    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Bet History — singles only; accas have their own section above ────
    settled_hist = [b for b in port["bets"]
                    if b["status"] in ("won", "lost") and b.get("type") != "acca"]
    if settled_hist:
        st.markdown('<p class="section-label" id="bet-history">📋  BET HISTORY · Singles</p>',
                    unsafe_allow_html=True)

        # Build numbered rows in chronological order so bet # matches the chart,
        # then render newest first
        chrono = sorted(settled_hist, key=lambda b: (b.get("settled_at") or b.get("date") or "", b.get("placed_at") or ""))
        chrono_with_idx = [(i + 1, b) for i, b in enumerate(chrono)]

        rows_html = []
        for bet_idx, b in reversed(chrono_with_idx):
            profit_b = b["profit"] or 0.0
            stake_b  = b["stake"]
            odds_b   = b["odds"]
            model_p  = b.get("model_prob") or 0.0
            implied  = (1.0 / odds_b) if odds_b > 0 else 0.0
            edge_pp  = (model_p - implied) * 100  # percentage-point edge

            # Date — short format e.g. "02-May-26"
            try:
                d = pd.to_datetime(b.get("date") or b.get("placed_at"))
                date_short = d.strftime("%d-%b-%y")
            except Exception:
                date_short = (b.get("date") or "")[:10]

            home, away = b["home"], b["away"]
            sel_label = b["selection"]

            # Result chip
            won = b["status"] == "won"
            res_html = (f'<span class="bh-result bh-won">✅ WON</span>'
                        if won else
                        f'<span class="bh-result bh-lost">❌ LOST</span>')

            # P&L chip with strong colour
            pnl_cls = "bh-pnl-pos" if profit_b > 0 else ("bh-pnl-neg" if profit_b < 0 else "bh-pnl-flat")
            pnl_sign = "+" if profit_b >= 0 else "−"
            pnl_html = (f'<span class="bh-pnl {pnl_cls}">'
                        f'{pnl_sign}£{abs(profit_b):,.2f}</span>')

            # Edge cell — model_prob vs market implied prob
            edge_col  = "#00e676" if edge_pp >= 5 else ("#ffd600" if edge_pp >= 0 else "#ff4081")
            edge_html = (
                f'<div class="bh-edge" title="Model probability vs bookmaker implied probability. '
                f'A bigger gap means we thought the bet was more underpriced.">'
                f'  <div class="bh-edge-row">'
                f'    <span class="bh-edge-lbl">model</span>'
                f'    <span class="bh-edge-num" style="color:#a78bfa">{model_p*100:.1f}%</span>'
                f'  </div>'
                f'  <div class="bh-edge-row">'
                f'    <span class="bh-edge-lbl">bookie</span>'
                f'    <span class="bh-edge-num">{implied*100:.1f}%</span>'
                f'  </div>'
                f'  <div class="bh-edge-gap" style="color:{edge_col}">+{edge_pp:.1f}pp edge</div>'
                f'</div>'
            )

            # Highlight if user clicked through from chart
            highlight_idx = st.session_state.get("_bh_highlight")
            row_cls = "bh-row" + (" bh-row-highlight" if highlight_idx == bet_idx else "")

            rows_html.append(
                f'<div class="{row_cls}" id="bet-row-{bet_idx}">'
                f'  <div class="bh-num">#{bet_idx}</div>'
                f'  <div class="bh-date">{date_short}</div>'
                f'  <div class="bh-team bh-home">{tb(home, 32)}</div>'
                f'  <div class="bh-team bh-away">{tb(away, 32)}</div>'
                f'  <div class="bh-sel">{sel_label}</div>'
                f'  <div class="bh-odds">{odds_b:.2f}</div>'
                f'  {edge_html}'
                f'  <div class="bh-stake">£{stake_b:,.2f}</div>'
                f'  <div class="bh-cell-result">{res_html}</div>'
                f'  <div class="bh-cell-pnl">{pnl_html}</div>'
                f'</div>'
            )

        # Header row
        header_html = (
            '<div class="bh-row bh-header">'
            '<div class="bh-num">#</div>'
            '<div class="bh-date">DATE</div>'
            '<div class="bh-team">HOME</div>'
            '<div class="bh-team">AWAY</div>'
            '<div class="bh-sel">PICK</div>'
            '<div class="bh-odds">ODDS</div>'
            '<div class="bh-edge">WHY WE BET</div>'
            '<div class="bh-stake">STAKE</div>'
            '<div class="bh-cell-result">RESULT</div>'
            '<div class="bh-cell-pnl">P&amp;L</div>'
            '</div>'
        )

        # Optional jump-to-bet input — lets user navigate from chart bet # to row
        jc1, jc2, _ = st.columns([1.5, 5, 4])
        with jc1:
            jump_to = st.number_input(
                "Jump to bet #", min_value=1, max_value=len(chrono),
                value=int(st.session_state.get("_bh_highlight") or len(chrono)),
                key="bh_jump", step=1,
                help="Type a bet number from the chart to highlight it in the table below.",
            )
        with jc2:
            if st.button("🎯  Highlight bet", key="bh_jump_btn"):
                st.session_state["_bh_highlight"] = int(jump_to)
                st.rerun()

        st.markdown(
            '<div class="bh-scroll"><div class="bh-table">'
            + header_html + "".join(rows_html)
            + '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Sharpness diagnostics: CLV vs Pinnacle close + calibration drift ──
    # CLV is the strongest forward-looking signal of long-run profitability —
    # ROI is too high-variance on draws to read in <100 bets. CLV stabilises
    # in ~50 bets. Median ≥ +1% on draws is competitive; ≥ +2% is sharp-tier.
    _clv = pf.clv_summary(port)
    _drift = pf.compute_brier_drift(port)
    _rolling = pf.clv_rolling(port, windows=(5, 10, 20))
    st.markdown('<p class="section-label">📐  SHARPNESS · CLV vs Pinnacle close</p>',
                unsafe_allow_html=True)

    if _clv["n"] == 0:
        st.markdown(
            '<div style="padding:0.9rem 1rem;background:rgba(124,77,255,0.08);'
            'border-left:3px solid #7c4dff;border-radius:6px;font-size:0.84rem;color:#8892a4">'
            'No CLV data yet — settled bets will be tagged with closing-line value automatically '
            'once the football-data CSV updates with Pinnacle close prices.</div>',
            unsafe_allow_html=True,
        )
    else:
        _med  = _clv["median_clv"] * 100
        _mean = _clv["mean_clv"] * 100
        _pos  = _clv["pct_positive"]
        _n    = _clv["n"]
        _med_col = "#00e676" if _med >= 1.0 else ("#ffd600" if _med >= 0 else "#ff4081")
        _pos_col = "#00e676" if _pos >= 55 else ("#ffd600" if _pos >= 45 else "#ff4081")
        _all_verdict = (
            ("🎯 Sharp-tier — beating the closing line consistently", "#00e676") if _med >= 2.0 else
            ("✅ Competitive edge — positive CLV, monitor sample",     "#00e676") if _med >= 1.0 else
            ("⚠️ Marginal — hovering around break-even vs close",      "#ffd600") if _med >= 0 else
            ("🚨 Below close — current profit is likely variance",     "#ff4081")
        )
        # Rolling/recent
        rolling_alert = _rolling.get("alert", "insufficient")
        recent_msg, recent_col = {
            "ok":    ("✅ Recent trend looks alive — last bets median CLV positive",  "#00e676"),
            "watch": ("⚠️ Recent trend marginal — recent CLV near zero",               "#ffd600"),
            "drift": ("🚨 Recent trend drifting negative — review before more bets",   "#ff4081"),
            "insufficient": ("ℹ️  Need a few more settled bets for a recent-trend read", "#7c4dff"),
        }.get(rolling_alert, ("", "#7c4dff"))

        # Build the per-window mini blocks for the Recent panel
        win_blocks = []
        for w in (5, 10, 20):
            wd = _rolling["windows"].get(w)
            if wd is None:
                win_blocks.append(
                    f'<div class="clv-win"><div class="clv-win-lbl">LAST {w}</div>'
                    f'<div class="clv-win-val clv-win-na">—</div></div>'
                )
                continue
            wmed = wd["median"] * 100
            wpos = wd["pct_pos"]
            wcol = "#00e676" if wmed >= 1.0 else ("#ffd600" if wmed >= 0 else "#ff4081")
            win_blocks.append(
                f'<div class="clv-win">'
                f'  <div class="clv-win-lbl">LAST {w}</div>'
                f'  <div class="clv-win-val" style="color:{wcol}">{wmed:+.2f}%</div>'
                f'  <div class="clv-win-pos">{wpos:.0f}% positive</div>'
                f'</div>'
            )

        # ── Hero CLV panel: All-Time | Recent Trend ──────────────────
        # Built as a single flat HTML string — Streamlit's markdown treats
        # blank lines inside HTML as paragraph breaks and breaks the output,
        # so we collapse all newlines and indentation.
        _sample_note = ("sufficient" if _n >= 30 else "small, treat as preliminary")
        _all_html = (
            '<div class="clv-hero-side clv-hero-all">'
            f'<div class="clv-hero-side-lbl">📈  ALL-TIME ({_n} settled bets)</div>'
            '<div class="clv-hero-row">'
                '<div class="clv-hero-block">'
                    '<div class="clv-hero-lbl">Median CLV</div>'
                    f'<div class="clv-hero-val" style="color:{_med_col}">{_med:+.2f}%</div>'
                '</div>'
                '<div class="clv-hero-block">'
                    '<div class="clv-hero-lbl">% Positive</div>'
                    f'<div class="clv-hero-val" style="color:{_pos_col}">{_pos:.0f}%</div>'
                '</div>'
                '<div class="clv-hero-block">'
                    '<div class="clv-hero-lbl">Mean CLV</div>'
                    f'<div class="clv-hero-val">{_mean:+.2f}%</div>'
                '</div>'
            '</div>'
            f'<div class="clv-hero-status" style="border-left-color:{_all_verdict[1]}">'
                f'<b style="color:{_all_verdict[1]}">{_all_verdict[0]}</b>'
                f'<div class="clv-hero-note">CLV stabilises in ~50 bets — sample is currently {_sample_note}.</div>'
            '</div>'
            '</div>'
        )
        # Flatten win_blocks (they may include multi-line indentation)
        _win_blocks_flat = "".join(b.replace("\n", "").replace("  ", "") for b in win_blocks)
        _recent_html = (
            '<div class="clv-hero-side clv-hero-recent">'
            '<div class="clv-hero-side-lbl">⏱️  RECENT TREND</div>'
            f'<div class="clv-hero-row">{_win_blocks_flat}</div>'
            f'<div class="clv-hero-status" style="border-left-color:{recent_col}">'
                f'<b style="color:{recent_col}">{recent_msg}</b>'
            '</div>'
            '</div>'
        )
        st.markdown(
            '<div class="clv-hero"><div class="clv-hero-grid">'
            + _all_html + _recent_html
            + '</div></div>',
            unsafe_allow_html=True,
        )

        # ── Sparkline trend (rolling-5 mean) ──────────────────────────
        _series = _rolling.get("series", [])
        if len(_series) >= 5:
            clvs_pct = [c * 100 for _, c in _series]
            roll = []
            for i in range(len(clvs_pct)):
                start = max(0, i - 4)
                roll.append(sum(clvs_pct[start:i+1]) / (i - start + 1))
            fig_clv = go.Figure()
            fig_clv.add_hline(y=0, line_color="rgba(255,255,255,0.18)", line_dash="dot")
            fig_clv.add_trace(go.Scatter(
                x=list(range(1, len(roll) + 1)), y=roll,
                mode="lines", line=dict(color="#7c4dff", width=3),
                fill="tozeroy", fillcolor="rgba(124,77,255,0.14)",
                hovertemplate="Bet %{x}: %{y:+.2f}% (5-bet rolling avg)<extra></extra>",
            ))
            fig_clv.update_layout(
                **{k: v for k, v in DARK.items() if k != "margin"},
                height=160, margin=dict(t=20, b=30, l=10, r=10),
                xaxis=dict(title="bet # (chronological)",
                           gridcolor="rgba(255,255,255,0.04)",
                           title_font=dict(size=12, color="#8892a4"),
                           tickfont=dict(size=12, color="#8892a4")),
                yaxis=dict(title="rolling CLV %",
                           gridcolor="rgba(255,255,255,0.04)",
                           title_font=dict(size=12, color="#8892a4"),
                           tickfont=dict(size=12, color="#8892a4"),
                           zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_clv, use_container_width=True,
                            config={"displayModeBar": False})
        # ── Glossary expander — what each metric is and what to aim for ──
        with st.expander("ℹ️  What do these numbers mean? — quick guide", expanded=False):
            st.markdown(
                '<div class="clv-guide">'

                # ── Hero explainer card — what is CLV ──
                '<div class="clv-guide-hero">'
                  '<div class="clv-guide-hero-icon">📐</div>'
                  '<div>'
                    '<div class="clv-guide-hero-title">Closing-Line Value (CLV)</div>'
                    '<div class="clv-guide-hero-sub">'
                      'The single best predictor of long-run profitability in sports betting research.'
                    '</div>'
                  '</div>'
                '</div>'

                # ── Worked example ──
                '<div class="clv-guide-example">'
                  '<div class="clv-guide-example-title">📖 Example</div>'
                  '<div class="clv-guide-example-row">'
                    '<div class="clv-guide-example-block">'
                      '<div class="clv-guide-eg-lbl">YOU TOOK</div>'
                      '<div class="clv-guide-eg-val clv-guide-eg-took">£10 @ 4.00</div>'
                    '</div>'
                    '<div class="clv-guide-example-arrow">→</div>'
                    '<div class="clv-guide-example-block">'
                      '<div class="clv-guide-eg-lbl">PINNACLE CLOSED AT</div>'
                      '<div class="clv-guide-eg-val clv-guide-eg-close">3.80</div>'
                    '</div>'
                    '<div class="clv-guide-example-arrow">=</div>'
                    '<div class="clv-guide-example-block">'
                      '<div class="clv-guide-eg-lbl">YOUR CLV</div>'
                      '<div class="clv-guide-eg-val clv-guide-eg-result">+5.3%</div>'
                    '</div>'
                  '</div>'
                  '<div class="clv-guide-example-note">'
                    "Calc: <b>4.00 / 3.80 − 1 = +5.3%</b>. You beat the sharpest-pricing book "
                    "in the world by 5.3% on this bet. That gap, measured across many bets, "
                    "is the closest thing in betting to a real-world skill score."
                  '</div>'
                '</div>'

                # ── Why CLV instead of ROI ──
                '<div class="clv-guide-vs">'
                  '<div class="clv-guide-vs-card clv-guide-roi">'
                    '<div class="clv-guide-vs-icon">📊</div>'
                    '<div class="clv-guide-vs-title">ROI</div>'
                    '<div class="clv-guide-vs-body">'
                      'How much money you actually made. <b>Pure variance noise</b> '
                      'until ~100 settled draw bets. A 30% win rate at 4.0 odds takes '
                      'hundreds of trials to confirm.'
                    '</div>'
                  '</div>'
                  '<div class="clv-guide-vs-vs">VS</div>'
                  '<div class="clv-guide-vs-card clv-guide-clv">'
                    '<div class="clv-guide-vs-icon">🎯</div>'
                    '<div class="clv-guide-vs-title">CLV</div>'
                    '<div class="clv-guide-vs-body">'
                      'Whether <b>the prices you took had edge</b>, independent of '
                      'whether the bet won or lost. Stabilises in <b>~50 bets</b> — half '
                      "the sample size. That's why we watch CLV first."
                    '</div>'
                  '</div>'
                '</div>'

                # ── The four metrics ──
                '<div class="clv-guide-metric-title">📋 The four numbers above explained</div>'

                '<div class="clv-guide-metric">'
                  '<div class="clv-guide-metric-name">Median CLV</div>'
                  '<div class="clv-guide-metric-desc">'
                    'The <b>middle bet</b> — half your bets had a better CLV than this, '
                    'half had worse. Robust against one or two big outliers.'
                  '</div>'
                  '<div class="clv-guide-bands">'
                    '<div class="clv-guide-band clv-band-sharp">'
                      '<span class="clv-guide-band-tier">≥ +2%</span>'
                      '<span>🎯 SHARP-TIER · beating Pinnacle consistently</span>'
                    '</div>'
                    '<div class="clv-guide-band clv-band-good">'
                      '<span class="clv-guide-band-tier">+1% to +2%</span>'
                      '<span>✅ COMPETITIVE EDGE · positive CLV, monitor sample</span>'
                    '</div>'
                    '<div class="clv-guide-band clv-band-warn">'
                      '<span class="clv-guide-band-tier">0% to +1%</span>'
                      '<span>⚠️ MARGINAL · hovering around break-even vs close</span>'
                    '</div>'
                    '<div class="clv-guide-band clv-band-bad">'
                      '<span class="clv-guide-band-tier">below 0%</span>'
                      '<span>🚨 BELOW CLOSE · profit is likely variance, not skill</span>'
                    '</div>'
                  '</div>'
                '</div>'

                '<div class="clv-guide-metric">'
                  '<div class="clv-guide-metric-name">Mean CLV</div>'
                  '<div class="clv-guide-metric-desc">'
                    'The arithmetic average. <b>If mean ≫ median</b>, you have a '
                    'long tail of big-CLV wins (rare big edges). <b>If mean ≪ median</b>, '
                    "a few big-CLV losses are dragging things down (you're chasing)."
                  '</div>'
                '</div>'

                '<div class="clv-guide-metric">'
                  '<div class="clv-guide-metric-name">% Positive</div>'
                  '<div class="clv-guide-metric-desc">'
                    'The share of bets where you got a price <b>better than the close</b>. '
                    'Sharp bettors typically run ≥ 55%. The market is a coin flip if '
                    "you're at 50%. Below 45% means lines are moving against your picks "
                    'more often than for them — a bad sign.'
                  '</div>'
                '</div>'

                '<div class="clv-guide-metric">'
                  '<div class="clv-guide-metric-name">Tagged Bets</div>'
                  '<div class="clv-guide-metric-desc">'
                    'How many settled bets we have closing-odds data for. Pinnacle close '
                    'is published in football-data CSVs the day after the match. '
                    'The signal stabilises around <b>50 tagged bets</b>.'
                  '</div>'
                '</div>'

                # ── What we are looking for ──
                '<div class="clv-guide-target">'
                  '<div class="clv-guide-target-title">🎯 What success looks like in this portfolio</div>'
                  '<div class="clv-guide-target-body">'
                    'We want <b style="color:#00e676">median CLV ≥ +1%</b> and '
                    '<b style="color:#00e676">% positive ≥ 55%</b> — sustained across '
                    'at least <b>50 settled bets</b>. Until then, profitable runs may '
                    'just be soft-pricing-window variance and will revert.'
                  '</div>'
                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )
        # CLV-by-market mini table
        _by_mkt = _clv.get("by_market", {})
        if _by_mkt:
            with st.expander("📊  CLV by market", expanded=False):
                _mkt_label = {"H": "Home Win", "D": "Draw", "A": "Away Win",
                              "over25": "Over 2.5", "under25": "Under 2.5"}
                _rows = []
                for _m, _data in _by_mkt.items():
                    _rows.append({
                        "Market":     _mkt_label.get(_m, _m),
                        "N":          _data["n"],
                        "Median CLV": f"{_data['median']*100:+.2f}%",
                        "Mean CLV":   f"{_data['mean']*100:+.2f}%",
                        "% Positive": f"{_data['pct_pos']:.0f}%",
                    })
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        # Per-bet CLV trail
        with st.expander("📜  Per-bet CLV trail", expanded=False):
            _clv_rows = []
            for _b in port["bets"]:
                if _b.get("type") == "acca" or _b.get("clv") is None:
                    continue
                # Look up final scoreline from the match data — useful context for
                # losing draw bets (was it 1-0? 4-3?) and winners alike.
                _score = "—"
                try:
                    _md = pd.to_datetime(_b.get("date")).normalize()
                    _r  = df[(df["HomeTeam"] == _b["home"]) &
                             (df["AwayTeam"] == _b["away"]) &
                             (df["Date"]     == _md)]
                    if len(_r) > 0:
                        _score = f"{int(_r.iloc[0]['FTHG'])}-{int(_r.iloc[0]['FTAG'])}"
                except Exception:
                    pass
                _clv_rows.append({
                    "Match":   f"{_b['home']} v {_b['away']}",
                    "Score":   _score,
                    "Date":    (_b.get("date") or "")[:10],
                    "Mkt":     _b.get("market", ""),
                    "Taken":   f"{_b['odds']:.2f}",
                    "Close":   f"{_b['closing_odds']:.2f}",
                    "CLV":     f"{_b['clv']*100:+.2f}%",
                    "Result":  _b["status"].title(),
                })
            if _clv_rows:
                st.dataframe(pd.DataFrame(_clv_rows), use_container_width=True, hide_index=True)
            else:
                st.markdown('<div style="color:#b8c0d0;font-size:0.82rem">No tagged bets yet.</div>',
                            unsafe_allow_html=True)

    # Drift signal — small banner that flags when recent calibration has shifted
    if _drift["drift_signal"] not in ("insufficient", "insufficient_baseline"):
        _ds = _drift["drift_signal"]
        _delta = _drift["delta"]
        _drift_col = ("#ff4081" if _ds == "drift_worse" else
                      "#00e676" if _ds == "drift_better" else "#445")
        _drift_msg = ("⚠️ Recent Brier {:+.3f} vs baseline — calibration drifting worse, "
                      "consider refitting isotonic" if _ds == "drift_worse" else
                      "✓ Recent Brier {:+.3f} vs baseline — calibration drifting better"
                      if _ds == "drift_better" else
                      "● Recent Brier {:+.3f} vs baseline — calibration stable")
        st.markdown(
            f'<div style="padding:0.55rem 1rem;margin-top:0.5rem;'
            f'background:rgba(124,77,255,0.05);'
            f'border-left:3px solid {_drift_col};border-radius:6px;font-size:0.78rem;color:#e8eaf0">'
            f'{_drift_msg.format(_delta)} '
            f'&nbsp;·&nbsp; <span style="color:#b8c0d0">recent {_drift["n_recent"]} vs '
            f'baseline {_drift["n_baseline"]} bets</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Historical EV Backtest ────────────────────────────────────────────
    with st.expander("📜  Historical EV Backtest — could you beat the bookies on past data?", expanded=False):
        st.markdown("""
        <div style="font-size:0.82rem;color:#b8c0d0;margin-bottom:1rem;line-height:1.6">
            Simulates what would have happened if you had placed Kelly-sized bets on every match
            where the <b style="color:#ccd">DC + XGB + Draw Specialist</b> ensemble identified a
            value opportunity vs actual <b style="color:#ccd">Bet365 closing odds</b> from our
            historical CSV data.
            Training is strictly cut off before each test window — no data leakage.
            Results directly answer: <i>does our model have long-run edge over the bookmaker?</i>
        </div>""", unsafe_allow_html=True)

        # ── Quick presets ─────────────────────────────────────────────
        _render_backtest_presets("hbt")

        # Multi-season note (matches Mock Two)
        st.markdown(
            '<div style="background:linear-gradient(135deg,rgba(0,229,255,0.07),rgba(124,77,255,0.04));'
            'border-left:3px solid #00e5ff;border-radius:8px;padding:0.7rem 1rem;'
            'margin-bottom:0.9rem;font-size:0.82rem;color:#cdd;line-height:1.5">'
            '💡 <b style="color:#00e5ff">Multi-season backtests:</b> '
            "test_weeks ≥ 80 spans more than one season. Recent seasons are "
            "the most representative — older data includes teams since "
            "relegated/promoted, and ELO ratings stabilise as more matches "
            "accumulate. <b>2025-26 is most predictive of next season's "
            "behaviour.</b> Defaults below match your saved live config."
            '</div>', unsafe_allow_html=True,
        )

        hc1, hc2, hc3, hc4, hc5, hc6 = st.columns(6)
        with hc1:
            hbt_weeks  = st.slider("Test Window (weeks)", 8, 200, 40,
                                    key="hbt_weeks",
                                    help="Default 40 ≈ one full season back from "
                                         "the latest match (Aug–May). 27 ≈ recent "
                                         "6 months (mid-season view). Beyond ~52 "
                                         "the single-split design goes stale — "
                                         "the model trains only on pre-window data.")
        with hc2:
            hbt_min_ev = st.slider("Min EV (%)", 1, 60,
                                    int(float(settings.get("min_ev", 0.40)) * 100),
                                    key="hbt_minev")
        with hc3:
            hbt_min_prob = st.slider(
                "Min Prob Gate (%)", 0, 80,
                int(settings.get("min_prob", 0.30) * 100),
                key="hbt_minprob",
                help="Reject candidates whose model probability is below this floor.",
            )
        with hc4:
            _kf = float(settings.get("kelly_fraction", 1.0))
            _kf_options = [0.25, 0.5, 0.75, 1.0]
            _kf_default = min(_kf_options, key=lambda x: abs(x - _kf))
            hbt_kelly  = st.select_slider(
                "Kelly Fraction", _kf_options, _kf_default, key="hbt_kelly",
                format_func=lambda x: f"{int(x*100)}%",
            )
        with hc5:
            hbt_max_stake = st.slider(
                "Max Stake (% of bankroll)", 5, 50,
                int(settings.get("max_stake_pct", 0.33) * 100),
                key="hbt_maxstake",
                help="Cap on a single bet. 33% = practical saturation.",
            )
        with hc6:
            hbt_bankroll = st.number_input(
                "Bankroll (£)", 100.0, 100000.0, 10000.0, 1000.0,
                key="hbt_bankroll", format="%.0f",
            )

        fc1, fc2, fc3, fc4 = st.columns([1.1, 1.3, 1.2, 1])
        with fc1:
            hbt_skip_late = st.checkbox(
                "Skip Mar-Apr",
                value=bool(settings.get("skip_late_season", True)),
                key="hbt_skip_late",
                help="0/7 wins across 2024-25 + 2025-26 in March-April. "
                     "(May was previously bundled in but is now allowed.)",
            )
        with fc2:
            hbt_skip_title = st.checkbox(
                "Skip home_title_race",
                value=bool(settings.get("skip_home_title_race", False)),
                key="hbt_skip_title",
                help="0/6 wins when home team chasing title (2025-26).",
            )
        with fc3:
            hbt_sim_main = st.checkbox(
                "Sim-bet correction",
                value=bool(settings.get("use_simultaneous_kelly", True)),
                key="hbt_sim",
                help="Reduce stake when multiple bets settle the same day.",
            )
        with fc4:
            _odds_options = ["B365", "Max", "Avg", "PS"]
            hbt_odds_src = st.selectbox(
                "Place at",
                options=_odds_options,
                index=_odds_options.index("Max"),
                key="hbt_odds_src",
                help="The price you actually win at if your bet hits. "
                     "Multi-season grid optimum: Max (best CLV when paired with PS).",
            )

        # Markets row — U2.5 separate-gates toggle
        mc1, mc2 = st.columns([1.5, 3])
        with mc1:
            hbt_u25_gates = st.checkbox(
                "Include Under 2.5 (separate gates)",
                value="under25" in (settings.get("market_gates") or {}),
                key="hbt_u25_gates",
                help="Adds U2.5 to the simulated market set with mp=50%, mev=5%.",
            )
        with mc2:
            if hbt_u25_gates:
                st.markdown(
                    '<div style="font-size:0.86rem;color:#a78bfa;padding-top:0.55rem">'
                    'Backtest will include <b>Draw + Under 2.5</b> with separate per-market gates '
                    '(U2.5: mp ≥ 50%, ev ≥ 5%).</div>',
                    unsafe_allow_html=True,
                )

        _detect_options = ["(same as place)", "B365", "Max", "Avg", "PS"]
        hbt_detect_src = st.selectbox(
            "Detect EV against (optional — leave 'same as place' for single-source)",
            options=_detect_options,
            index=_detect_options.index("PS"),
            key="hbt_detect_src",
            help="Multi-season grid optimum: detect at PS, place at Max.",
        )

        # ── Optional Phase 4 / ELO filter controls — defaulted from settings ──
        st.markdown(
            '<div style="margin:0.7rem 0 0.4rem;padding-top:0.6rem;'
            'border-top:1px dashed rgba(0,229,255,0.25);'
            'font-size:0.78rem;font-weight:800;letter-spacing:1.4px;'
            'text-transform:uppercase;color:#00e5ff">'
            '🧠 Optional escape-hatch filters (defaults match saved live settings)'
            '</div>', unsafe_allow_html=True,
        )
        _all_dows_h   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _all_months_h = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        bf1, bf2, bf3 = st.columns([1.2, 1.2, 1.6])
        with bf1:
            hbt_banned_dows = st.multiselect(
                "Banned days (DOW)",
                options=_all_dows_h,
                default=list(settings.get("main_banned_dows", [])),
                key="hbt_banned_dows",
            )
        with bf2:
            hbt_banned_months = st.multiselect(
                "Banned months",
                options=_all_months_h,
                default=list(settings.get("main_banned_months", [])),
                key="hbt_banned_months",
            )
        with bf3:
            _max_ev_default = settings.get("main_max_ev_pct")
            hbt_max_ev_on = st.checkbox(
                "Cap claimed EV (overconfidence guard)",
                value=_max_ev_default is not None,
                key="hbt_max_ev_on",
            )
            hbt_max_ev = (
                st.slider("Max EV cap (%)", 40, 200,
                          int((_max_ev_default or 1.0) * 100), 5,
                          key="hbt_max_ev_val")
                if hbt_max_ev_on else None
            )

        # ELO-profile filter row
        st.markdown(
            '<div style="margin-top:0.7rem;font-size:0.78rem;font-weight:700;'
            'color:#a78bfa">🎯 ELO-profile filter (Mock Two grid winner: 1500)</div>',
            unsafe_allow_html=True,
        )
        be1, be2, be3, be4 = st.columns(4)
        with be1:
            _mte = settings.get("main_min_team_elo")
            hbt_min_te_on = st.checkbox(
                "Min team ELO floor", value=_mte is not None,
                key="hbt_min_te_on",
                help="Skip if either team is below this ELO. Trade peak £85k for £6k floor.",
            )
            hbt_min_te = (st.slider("Min ELO", 1300, 1700, int(_mte or 1500),
                                     10, key="hbt_min_te_val")
                          if hbt_min_te_on else None)
        with be2:
            _mxe = settings.get("main_max_team_elo")
            hbt_max_te_on = st.checkbox(
                "Max team ELO ceiling", value=_mxe is not None,
                key="hbt_max_te_on",
            )
            hbt_max_te = (st.slider("Max ELO", 1700, 2100, int(_mxe or 1900),
                                     10, key="hbt_max_te_val")
                          if hbt_max_te_on else None)
        with be3:
            _gmin = settings.get("main_elo_gap_min")
            hbt_gap_min_on = st.checkbox(
                "Min |ΔELO|", value=_gmin is not None, key="hbt_gap_min_on",
            )
            hbt_gap_min = (st.slider("Min gap", 20, 200, int(_gmin or 80),
                                      10, key="hbt_gap_min_val")
                           if hbt_gap_min_on else None)
        with be4:
            _gmax = settings.get("main_elo_gap_max")
            hbt_gap_max_on = st.checkbox(
                "Max |ΔELO|", value=_gmax is not None, key="hbt_gap_max_on",
            )
            hbt_gap_max = (st.slider("Max gap", 100, 500, int(_gmax or 300),
                                      20, key="hbt_gap_max_val")
                           if hbt_gap_max_on else None)

        if st.button("🔄  Run Simulation", key="run_hbt", type="primary"):
            with st.spinner("Running EV simulation on historical match data..."):
                bt_df = backtest_models(df, df_features, test_weeks=hbt_weeks)
                _bt_markets = set(settings.get("auto_markets", list(pf.PROFITABLE_MARKETS)))
                _detect_main = None if hbt_detect_src == "(same as place)" else str(hbt_detect_src)
                # When U2.5 toggle is on, force-add it to the market set and apply its gates
                if hbt_u25_gates:
                    _bt_markets = set(_bt_markets) | {"under25"}
                    _market_gates = {"under25": {"min_prob": 0.50, "min_ev": 0.05}}
                else:
                    _market_gates = settings.get("market_gates")
                log_df, summary = pf.ev_backtest_simulate(
                    bt_df, df,
                    min_ev_pct=float(hbt_min_ev),
                    kelly_frac=float(hbt_kelly),
                    max_stake_pct=float(hbt_max_stake) / 100.0,
                    initial_bankroll=float(hbt_bankroll),
                    allowed_markets=_bt_markets,
                    min_prob=float(hbt_min_prob) / 100.0,
                    skip_late_season=bool(hbt_skip_late),
                    skip_home_title_race=bool(hbt_skip_title),
                    odds_source=str(hbt_odds_src),
                    detect_source=_detect_main,
                    enable_simultaneous_correction=bool(hbt_sim_main),
                    market_gates=_market_gates,
                    # Honest calibration: fitted strictly BEFORE the eval window
                    # (never the live cached_calibrators — those overlap it)
                    calibrators=cached_honest_calibrators(len(df), int(hbt_weeks)),
                    # Optional Phase 4 + ELO filters (default OFF)
                    banned_dows=set(hbt_banned_dows) if hbt_banned_dows else None,
                    banned_months=set(hbt_banned_months) if hbt_banned_months else None,
                    max_ev_pct=(hbt_max_ev / 100.0 if hbt_max_ev is not None else None),
                    min_team_elo=float(hbt_min_te) if hbt_min_te is not None else None,
                    max_team_elo=float(hbt_max_te) if hbt_max_te is not None else None,
                    elo_gap_min=float(hbt_gap_min) if hbt_gap_min is not None else None,
                    elo_gap_max=float(hbt_gap_max) if hbt_gap_max is not None else None,
                    # No-history gate, mirroring live so the backtest measures
                    # the gate as it actually runs
                    min_team_matches=settings.get("min_team_matches"),
                    df_features=df_features,
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
                with ms3:
                    _skipped = summary.get("skipped_min_prob", 0)
                    st.metric(
                        "Total Bets", str(summary["n_bets"]),
                        f"−{_skipped} below min-prob" if _skipped else None,
                        delta_color="off",
                    )
                with ms4: st.metric("Win Rate", f"{summary['win_rate']:.0f}%")
                with ms5: st.metric("Avg Odds", f"{summary['avg_odds']:.2f}")

                if not log_df.empty:
                    _init_br = summary["initial"]
                    x_h = list(range(len(log_df) + 1))
                    y_h = [float(_init_br)] + [float(v) for v in log_df["Bankroll"].tolist()]
                    line_col = "#00e676" if y_h[-1] >= _init_br else "#ff4081"

                    fig_hbt = go.Figure()

                    # Insert baseline-crossing points so green/red fills don't bleed
                    # past the actual line on segments that straddle the baseline.
                    x_exp: list[float] = [x_h[0]]
                    y_exp: list[float] = [y_h[0]]
                    for i in range(1, len(y_h)):
                        y_prev, y_cur = y_h[i - 1], y_h[i]
                        if (y_prev - _init_br) * (y_cur - _init_br) < 0:
                            t = (_init_br - y_prev) / (y_cur - y_prev)
                            x_cross = x_h[i - 1] + t * (x_h[i] - x_h[i - 1])
                            x_exp.append(x_cross); y_exp.append(_init_br)
                        x_exp.append(x_h[i]); y_exp.append(y_cur)

                    y_up = [max(v, _init_br) for v in y_exp]
                    y_dn = [min(v, _init_br) for v in y_exp]
                    baseline = [_init_br] * len(y_exp)

                    # Green fill from baseline up to bankroll where bankroll ≥ baseline
                    fig_hbt.add_trace(go.Scatter(
                        x=x_exp, y=baseline, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_hbt.add_trace(go.Scatter(
                        x=x_exp, y=y_up, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        fill="tonexty", fillcolor="rgba(0,230,118,0.22)",
                        hoverinfo="skip", showlegend=False,
                    ))
                    # Red fill from baseline down to bankroll where bankroll ≤ baseline
                    fig_hbt.add_trace(go.Scatter(
                        x=x_exp, y=baseline, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_hbt.add_trace(go.Scatter(
                        x=x_exp, y=y_dn, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        fill="tonexty", fillcolor="rgba(255,64,129,0.22)",
                        hoverinfo="skip", showlegend=False,
                    ))

                    # Baseline line + label
                    fig_hbt.add_hline(
                        y=_init_br, line_color="rgba(255,255,255,0.30)", line_dash="dot",
                        annotation_text=f"Start £{_init_br:,.0f}",
                        annotation_font=dict(color="#8892a4", size=14, family="Inter"),
                        annotation_position="top left",
                    )

                    # Soft halo + main line with win/loss markers
                    deltas = [0.0] + [y_h[i] - y_h[i - 1] for i in range(1, len(y_h))]
                    marker_symbols = ["circle"] + [
                        "triangle-up" if d > 0 else ("triangle-down" if d < 0 else "circle-open")
                        for d in deltas[1:]
                    ]
                    marker_colors = ["#8892a4"] + [
                        "#00e676" if d > 0 else ("#ff4081" if d < 0 else "#8892a4")
                        for d in deltas[1:]
                    ]
                    marker_sizes = [0] + [11 if d != 0 else 5 for d in deltas[1:]]
                    hover_texts = ["Start"] + [
                        ("▲ WON "  + f"+£{d:,.2f}") if d > 0 else
                        ("▼ LOST " + f"−£{abs(d):,.2f}") if d < 0 else
                        "No change"
                        for d in deltas[1:]
                    ]

                    fig_hbt.add_trace(go.Scatter(
                        x=x_h, y=y_h, mode="lines",
                        line=dict(color=f"rgba({_hex_to_rgb(line_col)},0.30)", width=10),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_hbt.add_trace(go.Scatter(
                        x=x_h, y=y_h, mode="lines+markers",
                        line=dict(color=line_col, width=3.2, shape="linear"),
                        marker=dict(
                            symbol=marker_symbols, size=marker_sizes,
                            color=marker_colors,
                            line=dict(color="#0a0e1a", width=1.2),
                        ),
                        text=hover_texts,
                        hovertemplate=("<b>Bet %{x}</b><br>%{text}<br>"
                                       "<b style='font-size:14px'>Bankroll £%{y:,.2f}</b><extra></extra>"),
                        hoverlabel=dict(
                            bgcolor="#1a1d27", bordercolor=line_col,
                            font=dict(size=14, family="Inter", color="#e8eaf0"),
                        ),
                        showlegend=False,
                    ))

                    # Peak / Low / NOW badges
                    peak_idx   = int(np.argmax(y_h))
                    trough_idx = int(np.argmin(y_h))
                    if peak_idx > 0 and y_h[peak_idx] > _init_br * 1.05:
                        fig_hbt.add_annotation(
                            x=x_h[peak_idx], y=y_h[peak_idx],
                            text=f"<b>Peak</b><br>£{y_h[peak_idx]:,.0f}",
                            showarrow=True, arrowhead=2, arrowcolor="#00e676",
                            arrowsize=1.2, arrowwidth=1.5, ax=0, ay=-38,
                            font=dict(size=12, color="#00e676", family="Inter"),
                            bgcolor="rgba(0,230,118,0.10)",
                            bordercolor="rgba(0,230,118,0.4)",
                            borderpad=4, borderwidth=1,
                        )
                    if (trough_idx > 0 and y_h[trough_idx] < _init_br * 0.95
                            and trough_idx != peak_idx):
                        fig_hbt.add_annotation(
                            x=x_h[trough_idx], y=y_h[trough_idx],
                            text=f"<b>Low</b><br>£{y_h[trough_idx]:,.0f}",
                            showarrow=True, arrowhead=2, arrowcolor="#ff4081",
                            arrowsize=1.2, arrowwidth=1.5, ax=0, ay=38,
                            font=dict(size=12, color="#ff4081", family="Inter"),
                            bgcolor="rgba(255,64,129,0.10)",
                            bordercolor="rgba(255,64,129,0.4)",
                            borderpad=4, borderwidth=1,
                        )
                    fig_hbt.add_annotation(
                        x=x_h[-1], y=y_h[-1],
                        text=f"<b>NOW · £{y_h[-1]:,.0f}</b>",
                        showarrow=False, xshift=15,
                        font=dict(size=14, color="#fff", family="Inter"),
                        bgcolor=line_col, bordercolor=line_col,
                        borderpad=8, borderwidth=2, xanchor="left",
                    )

                    fig_hbt.update_layout(
                        **{k: v for k, v in DARK.items() if k != "margin"},
                        height=460, showlegend=False,
                        margin=dict(t=40, b=40, l=20, r=140),
                        xaxis=dict(
                            title=dict(text="BET NUMBER",
                                       font=dict(size=12, color="#7c4dff", family="Inter"),
                                       standoff=18),
                            showgrid=False, showticklabels=True,
                            tickfont=dict(size=13, color="#8892a4", family="Inter"),
                            zeroline=False,
                        ),
                        yaxis=dict(
                            title=dict(text="BANKROLL",
                                       font=dict(size=12, color="#7c4dff", family="Inter"),
                                       standoff=14),
                            gridcolor="rgba(255,255,255,0.05)",
                            tickprefix="£",
                            tickfont=dict(size=14, color="#cdd", family="Inter"),
                            zeroline=False, tickformat=",.0f",
                        ),
                    )
                    st.plotly_chart(fig_hbt, use_container_width=True, config={"displayModeBar": False})

                    _render_backtest_clv_trend(log_df, df, key_prefix="hbt")

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
# Tab 7 — Mock Portfolio Two (research-track A/B vs main)
# ─────────────────────────────────────────────────────────────────────────────

def tab_portfolio_two(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                      draw_xgb_m, draw_fc, teams, elo_dict):
    """Parallel paper-trading portfolio with the research-track stack:
    Dixon-Coles + Karlis-Ntzoufras diagonal inflation, Baker-McHale
    uncertainty-shrunk Kelly, and simultaneous-bet correction. Lives next to
    the main portfolio so end-of-season comparison decides what graduates.
    """
    # Fit K-N model (cached) and per-bin variances (cached on backtest)
    with st.spinner("Fitting research-track model (DC + Karlis-Ntzoufras)…"):
        try:
            dc_kn_r = cached_dc_kn(f"{len(df)}_{df['Date'].max().date()}")
        except Exception as e:
            st.error(f"Could not fit K-N model: {e}")
            return
        try:
            bin_vars = cached_bin_variances(len(df))
        except Exception:
            bin_vars = {}

    # Load both portfolios for comparison
    main_port = pf.load_portfolio()
    port2     = pf.load_portfolio_two()

    # Auto-settle Mock Two against same df (main is settled in tab_portfolio)
    n_settled2 = pf.auto_settle(port2, df)
    n_clv2     = pf.backfill_clv_for_settled_bets(port2, df)
    if n_settled2 > 0 or n_clv2 > 0:
        pf.save_portfolio_two(port2)
    # Make sure main is up-to-date too (so the comparison numbers match)
    pf.auto_settle(main_port, df)
    pf.backfill_clv_for_settled_bets(main_port, df)
    pf.save_portfolio(main_port)

    settings = port2["settings"]
    api_key  = pf.resolve_odds_api_key(main_port.get("settings", {}).get("odds_api_key", ""))

    try:
        with st.spinner("Fitting probability calibration…"):
            calibrators = cached_calibrators(len(df))
    except Exception:
        calibrators = {}

    # ── Header: research-track tagline + γ readout ────────────────────────
    gamma_val = dc_kn_r.get("gamma", 0.0)
    gamma_col = "#00e676" if gamma_val > 0.02 else ("#ffd600" if gamma_val > -0.02 else "#ff4081")
    st.markdown(
        f'<div style="padding:0.9rem 1.1rem;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(124,77,255,0.10),rgba(0,229,255,0.04));'
        f'border:1px solid rgba(124,77,255,0.25);margin-bottom:1.2rem">'
        f'<div style="font-size:0.86rem;letter-spacing:3px;color:#a78bfa;font-weight:700;'
        f'text-transform:uppercase;margin-bottom:0.4rem">🧪  Research-Track Portfolio</div>'
        f'<div style="font-size:0.94rem;color:#e8eaf0;line-height:1.5">'
        f'Dixon-Coles <b>+ Karlis-Ntzoufras γ-inflation</b> (γ={gamma_val:+.4f} '
        f'<span style="color:{gamma_col}">'
        f'{"⬆ inflating draws" if gamma_val > 0.02 else "⬇ deflating draws" if gamma_val < -0.02 else "≈ near zero"}'
        f'</span>) &nbsp;·&nbsp; '
        f'Baker-McHale uncertainty-shrunk Kelly &nbsp;·&nbsp; '
        f'simultaneous-bet correction</div>'
        f'<div style="font-size:0.78rem;color:#8892a4;margin-top:0.4rem">'
        f'Parallel paper-trade vs main portfolio. Same gates (D + Under 2.5 by default), '
        f'different model + sizing. End-of-season delta tells us what to graduate.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── A/B Comparison header — main vs Mock Two ──────────────────────────
    main_stats = pf.portfolio_stats(main_port)
    mt_stats   = pf.portfolio_stats(port2)
    main_clv   = pf.clv_summary(main_port)
    mt_clv     = pf.clv_summary(port2)

    st.markdown('<p class="section-label">⚖️  A/B Compare · Main vs Research-Track</p>',
                unsafe_allow_html=True)
    cmp_cols = st.columns(2)
    for col, label, color, st_dict, cl_dict, port_dict, accent in [
        (cmp_cols[0], "MAIN PORTFOLIO", "#3d6eff", main_stats, main_clv, main_port, "#3d6eff"),
        (cmp_cols[1], "MOCK TWO (research)", "#7c4dff", mt_stats, mt_clv, port2, "#7c4dff"),
    ]:
        profit = st_dict["profit"]
        roi    = st_dict["roi"]
        sign   = "+" if profit >= 0 else ""
        prof_col = "#00e676" if profit >= 0 else "#ff4081"
        clv_med  = (cl_dict["median_clv"] * 100) if cl_dict["n"] > 0 else None
        clv_col  = ("#00e676" if (clv_med or 0) >= 1 else "#ffd600" if (clv_med or 0) >= 0 else "#ff4081") if clv_med is not None else "#556"
        clv_str  = f"{clv_med:+.2f}%" if clv_med is not None else "—"
        with col:
            st.markdown(f"""<div style="padding:1rem 1.2rem;border-radius:10px;
                background:rgba(255,255,255,0.02);border-left:4px solid {accent}">
                <div style="font-size:0.84rem;letter-spacing:2px;color:{accent};font-weight:800">{label}</div>
                <div style="display:flex;gap:1.4rem;align-items:baseline;margin-top:0.5rem;flex-wrap:wrap">
                    <div>
                        <div style="font-size:0.8rem;color:#b8c0d0;letter-spacing:1px">P&amp;L</div>
                        <div style="font-size:1.7rem;font-weight:900;color:{prof_col}">{sign}£{abs(profit):,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#b8c0d0;letter-spacing:1px">ROI</div>
                        <div style="font-size:1.2rem;font-weight:700;color:{prof_col}">{sign}{roi:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#b8c0d0;letter-spacing:1px">SETTLED</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#e8eaf0">{st_dict["n_settled"]}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#b8c0d0;letter-spacing:1px">MEDIAN CLV</div>
                        <div style="font-size:1.2rem;font-weight:700;color:{clv_col}">{clv_str}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#b8c0d0;letter-spacing:1px">BANKROLL</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#e8eaf0">£{st_dict["bankroll"]:,.0f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
    # Verdict
    delta_roi = mt_stats["roi"] - main_stats["roi"]
    delta_clv = ((mt_clv["median_clv"] or 0) - (main_clv["median_clv"] or 0)) * 100
    if mt_stats["n_settled"] >= 5 or main_stats["n_settled"] >= 5:
        verd_col = "#00e676" if delta_roi > 0 else ("#ffd600" if delta_roi > -2 else "#ff4081")
        verd_msg = (
            f"Research track {'leading' if delta_roi > 0 else 'trailing'} main by "
            f"{abs(delta_roi):.1f}pp ROI &nbsp;·&nbsp; CLV gap {delta_clv:+.2f}pp"
        )
        sample_msg = (f"Both portfolios still small-sample — verdict needs ≥50 settled bets to mean much."
                      if min(mt_stats["n_settled"], main_stats["n_settled"]) < 50
                      else "Sample is large enough to read directionally.")
        st.markdown(
            f'<div style="padding:0.7rem 1rem;margin-top:0.6rem;'
            f'background:rgba(124,77,255,0.06);border-left:3px solid {verd_col};'
            f'border-radius:6px;font-size:0.84rem;color:#e8eaf0">'
            f'<b style="color:{verd_col}">{verd_msg}</b> '
            f'&nbsp;·&nbsp; <span style="color:#8892a4">{sample_msg}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider" style="margin:1.4rem 0"></div>', unsafe_allow_html=True)

    # ── Settings panel (Mock Two only) ────────────────────────────────────
    # ── Status chip — current Mock Two config at a glance ────────────────
    _render_settings_chip(port2, label_prefix="p2")

    # ── Phase 4 findings expander — explain the validated config ─────────
    with st.expander("🧠  Phase 4 — what's behind the current config", expanded=False):
        _label  = port2["settings"].get("v2_settings_label", "—")
        _source = port2["settings"].get("v2_settings_source", "—")
        st.markdown(
            '<div style="background:linear-gradient(135deg,rgba(0,230,118,0.08),rgba(124,77,255,0.04));'
            'border:1px solid rgba(0,230,118,0.25);border-left:5px solid #00e676;'
            'border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem;'
            'font-size:0.92rem;line-height:1.55;color:#cdd">'
            f'<b style="color:#00e676">Deployed config:</b> '
            f'<code style="background:rgba(0,0,0,0.25);padding:2px 8px;border-radius:6px;'
            f'color:#00e5ff;font-size:0.84rem">{_label}</code><br>'
            f'<b style="color:#a78bfa">52-week WF result:</b> '
            f'£10k → <b style="color:#e8eaf0">£631,537</b> '
            f'<span style="color:#8892a4">(+96% ROI, 33% max DD, 35 bets, 49% win)</span><br>'
            f'<b style="color:#a78bfa">vs Main baseline:</b> '
            f'£59,320 (+24% ROI, 65% max DD) → <b style="color:#00e676">10× profit, half the DD</b><br>'
            f'<b style="color:#a78bfa">vs £93k ceiling:</b> 6.8× past target<br>'
            f'<span style="color:#8892a4;font-size:0.84rem">'
            f'Source: <code>data/diagnostics/{_source}.json</code></span>'
            '</div>', unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '<p style="font-size:0.84rem;font-weight:800;color:#a78bfa;'
                'letter-spacing:1.4px;text-transform:uppercase;margin-bottom:0.5rem">'
                'Phase 1 — top loss leaks</p>', unsafe_allow_html=True,
            )
            _phase1_losses = [
                ("Everton",       6, -23629, -69.8),
                ("Nott'm Forest", 7, -15829, -74.6),
                ("Aston Villa",   5, -15687, -72.3),
                ("Fulham",        9,  -9684, -38.1),
                ("Crystal Palace",1,  -9311,-100.0),
                ("Leeds",         1,  -4260,-100.0),
            ]
            _rows = "".join([
                f'<tr><td>{t}</td><td style="text-align:right">{n}</td>'
                f'<td style="text-align:right;color:#ff4081">£{p:,.0f}</td>'
                f'<td style="text-align:right;color:#ff4081">{r:+.1f}%</td></tr>'
                for t, n, p, r in _phase1_losses
            ])
            st.markdown(
                '<table style="width:100%;font-size:0.86rem;color:#cdd;'
                'border-collapse:collapse;font-variant-numeric:tabular-nums">'
                '<thead><tr style="color:#a78bfa;font-size:0.78rem;letter-spacing:1.2px">'
                '<th style="text-align:left;padding-bottom:0.4rem">TEAM</th>'
                '<th style="text-align:right">N</th>'
                '<th style="text-align:right">P&amp;L</th>'
                '<th style="text-align:right">ROI</th></tr></thead>'
                f'<tbody>{_rows}</tbody></table>', unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="font-size:0.78rem;color:#8892a4;margin-top:0.6rem">'
                'These six teams alone leaked <b style="color:#ff4081">−£77.4k</b> '
                "of the 52-week sample. Worth filtering on paper — but the "
                "rolling team-ROI filter <i>failed walk-forward</i> "
                "(reactive, blacklists too late). Keep visible for analysis "
                "but don't deploy as a gate.</p>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                '<p style="font-size:0.84rem;font-weight:800;color:#a78bfa;'
                'letter-spacing:1.4px;text-transform:uppercase;margin-bottom:0.5rem">'
                'Phase 4 — Top 5 WF variants</p>', unsafe_allow_html=True,
            )
            _phase4_top = [
                ("dowMF + ev≤1.00 + octX + mp30", 680390, 41.4, 63),
                ("dowMF + ev≤1.00 + octX + mp32", 631537, 32.8, 96, "★ deployed"),
                ("dowMFS + ev≤1.00 + octX + mp30", 543839, 41.1, 105),
                ("dowMF + ev free + octX + mp32", 479291, 32.8, 95),
                ("dowMFS + ev≤1.00 + oct keep + mp30", 427047, 41.1, 103),
            ]
            _rows4 = ""
            for entry in _phase4_top:
                tag = entry[4] if len(entry) > 4 else ""
                tag_html = (f'<span style="color:#00e676;font-weight:700;'
                            f'font-size:0.78rem;margin-left:0.3rem">{tag}</span>'
                            if tag else "")
                _rows4 += (
                    f'<tr><td style="font-size:0.78rem">{entry[0]}{tag_html}</td>'
                    f'<td style="text-align:right;color:#00e676">£{entry[1]:,.0f}</td>'
                    f'<td style="text-align:right">{entry[2]:.1f}%</td>'
                    f'<td style="text-align:right;color:#a78bfa">+{entry[3]}%</td></tr>'
                )
            st.markdown(
                '<table style="width:100%;font-size:0.86rem;color:#cdd;'
                'border-collapse:collapse;font-variant-numeric:tabular-nums">'
                '<thead><tr style="color:#a78bfa;font-size:0.78rem;letter-spacing:1.2px">'
                '<th style="text-align:left;padding-bottom:0.4rem">CONFIG</th>'
                '<th style="text-align:right">FINAL</th>'
                '<th style="text-align:right">DD</th>'
                '<th style="text-align:right">ROI</th></tr></thead>'
                f'<tbody>{_rows4}</tbody></table>', unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="font-size:0.78rem;color:#8892a4;margin-top:0.6rem">'
                "Filters that <b style='color:#ff4081'>failed</b> WF: "
                "team-ROI blacklist, drawdown stake throttle, min-prob ≥ 0.40, "
                "stacking everything. Why deploy a smaller-profit variant? "
                "Lower DD = higher staying power if the winning regime breaks.</p>",
                unsafe_allow_html=True,
            )

    with st.expander("⚙️  Mock Two Settings — research-track switches", expanded=False):

        # ── Quick presets ──────────────────────────────────────────────
        _render_preset_buttons(port2, key_prefix="p2", save_fn=pf.save_portfolio_two)

        # ── Section: Stake & Bankroll ──────────────────────────────────
        _section_header("💰  Stake & Bankroll", "How much you bet, sized by Kelly")
        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            new_init = st.number_input("Starting Bankroll (£)", 100.0, 100_000.0,
                                        float(port2["initial_bankroll"]), 100.0,
                                        key="p2_budget")
        with sb2:
            new_kelly = st.select_slider("Base Kelly Fraction",
                options=[0.25, 0.5, 0.75, 1.0],
                value=float(settings["kelly_fraction"]), key="p2_kelly",
                format_func=lambda x: f"{int(x*100)}%",
                help="Pre-shrinkage Kelly. Baker-McHale shrinks further from this.")
        with sb3:
            new_max_stake = st.slider("Max Stake (% of bankroll)", 5, 50,
                int(settings.get("max_stake_pct", 0.33) * 100), key="p2_maxstake",
                help="33% = WF saturation point.")

        # ── Section: Edge Gates ────────────────────────────────────────
        _section_header("🎯  Edge Gates",
                        "Default thresholds — per-market overrides via U2.5 toggle below")
        eg1, eg2 = st.columns(2)
        with eg1:
            new_min_ev = st.slider("Min EV (default, %)", 1, 60,
                int(settings["min_ev"] * 100), key="p2_minev")
        with eg2:
            new_min_prob = st.slider("Min Prob Gate (%)", 10, 80,
                int(settings["min_prob"] * 100), key="p2_minprob")

        # ── Section: Markets ───────────────────────────────────────────
        _section_header("📊  Markets",
                        "Which markets auto-bet considers + per-market gates")
        _existing_gates2 = settings.get("market_gates") or {}
        new_u25_gates2 = st.toggle(
            "🎚️ Use separate Under 2.5 gates (mp=50%, mev=5%)",
            value="under25" in _existing_gates2,
            key="p2_u25_gates",
            help="Adds U2.5 with its own thresholds; lets the V2 stack extract "
                 "value from a market the global EV gate would exclude.",
        )

        # ── Section: Filters ───────────────────────────────────────────
        _section_header("🚫  Filters",
                        "Skip systematically losing patterns")
        f1, f2, _ = st.columns([1, 1.5, 2])
        with f1:
            new_skip_late = st.toggle("Skip Mar-May",
                value=bool(settings.get("skip_late_season", True)), key="p2_skip_late",
                help="0/7 wins in Mar-Apr across 2 seasons.")
        with f2:
            new_skip_title = st.toggle("Skip home_title_race",
                value=bool(settings.get("skip_home_title_race", False)),
                key="p2_skip_title",
                help="Only proven on 2025-26.")

        # ── Section: Phase 4 validated filters ─────────────────────────
        _section_header("🧠  Phase 4 — Validated Filters",
                        "Walk-forward validated levers — DOW + Month + Max-EV")
        _all_dows   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pf1, pf2 = st.columns(2)
        with pf1:
            new_banned_dows = st.multiselect(
                "Banned days of week",
                options=_all_dows,
                default=list(settings.get("v2_banned_dows", [])),
                key="p2_banned_dows",
                help=("Phase 4 winner: skip Mon+Fri (Phase 1 saw 0/6 across 2 "
                      "seasons). Adding Sun reduces n but raises win rate."),
            )
            new_banned_months = st.multiselect(
                "Banned months",
                options=_all_months,
                default=list(settings.get("v2_banned_months", [])),
                key="p2_banned_months",
                help=("Phase 4 winner: ban Oct (Phase 1 saw 0/5 ROI -100%). "
                      "Skip Mar-May still controlled by `Skip Mar-May` toggle "
                      "above (a separate code path)."),
            )
        with pf2:
            _max_ev_cur = settings.get("v2_max_ev_pct")
            new_max_ev_on = st.toggle(
                "Cap claimed EV (overconfidence guard)",
                value=_max_ev_cur is not None,
                key="p2_max_ev_on",
                help=("Phase 1 found EV bucket 0.50–0.60 had ROI −33.7% — "
                      "model gets *over-confident* at high EV. Cap stops the "
                      "biggest fake edges from getting full Kelly stake."),
            )
            if new_max_ev_on:
                new_max_ev = st.slider(
                    "Max EV cap (%)",
                    min_value=40, max_value=200, step=5,
                    value=int((_max_ev_cur if _max_ev_cur is not None else 1.0) * 100),
                    key="p2_max_ev_val",
                    help="Phase 4 best at 100%. Tighter caps at 50–80% also win.",
                )
            else:
                new_max_ev = None

        # ── ELO-profile filter (multi-season grid winner) ──────────────
        st.markdown(
            '<div style="margin-top:0.9rem;padding-top:0.6rem;'
            'border-top:1px dashed rgba(0,229,255,0.25);'
            'font-size:0.78rem;font-weight:800;letter-spacing:1.4px;'
            'text-transform:uppercase;color:#00e5ff">'
            '🎯 ELO-profile filter (deployed: min team ELO 1500)'
            '</div>', unsafe_allow_html=True,
        )
        ec1, ec2 = st.columns(2)
        with ec1:
            _min_te_cur = settings.get("v2_min_team_elo")
            new_min_te_on = st.toggle(
                "Min team ELO floor",
                value=_min_te_cur is not None,
                key="p2_min_te_on",
                help=("Skip matches where EITHER team's ELO is below this. "
                      "Multi-season grid winner: 1500. Survives 4/4 seasons "
                      "with no crashes (£5,647 worst case vs £1,505 baseline)."),
            )
            new_min_te = (st.slider("Min team ELO",
                                     min_value=1300, max_value=1700, step=10,
                                     value=int(_min_te_cur or 1500),
                                     key="p2_min_te_val")
                          if new_min_te_on else None)

            _gap_min_cur = settings.get("v2_elo_gap_min")
            new_gap_min_on = st.toggle(
                "Min ELO gap (skip too-close)",
                value=_gap_min_cur is not None,
                key="p2_gap_min_on",
                help="Skip matches where |home_elo - away_elo| is below this.",
            )
            new_gap_min = (st.slider("Min |ΔELO|",
                                      min_value=20, max_value=200, step=10,
                                      value=int(_gap_min_cur or 80),
                                      key="p2_gap_min_val")
                           if new_gap_min_on else None)
        with ec2:
            _max_te_cur = settings.get("v2_max_team_elo")
            new_max_te_on = st.toggle(
                "Max team ELO ceiling",
                value=_max_te_cur is not None,
                key="p2_max_te_on",
                help="Skip matches where EITHER team's ELO is above this "
                     "(rare use — top-vs-top games).",
            )
            new_max_te = (st.slider("Max team ELO",
                                     min_value=1700, max_value=2100, step=10,
                                     value=int(_max_te_cur or 1900),
                                     key="p2_max_te_val")
                          if new_max_te_on else None)

            _gap_max_cur = settings.get("v2_elo_gap_max")
            new_gap_max_on = st.toggle(
                "Max ELO gap (skip lopsided)",
                value=_gap_max_cur is not None,
                key="p2_gap_max_on",
                help="Skip matches where |home_elo - away_elo| is above this.",
            )
            new_gap_max = (st.slider("Max |ΔELO|",
                                      min_value=100, max_value=500, step=20,
                                      value=int(_gap_max_cur or 300),
                                      key="p2_gap_max_val")
                           if new_gap_max_on else None)

        # ── Experimental (failed WF) — gated by a checkbox to keep the panel
        # tidy; can't use st.expander here because we're already inside one.
        st.markdown(
            '<div style="margin-top:0.8rem;padding-top:0.8rem;'
            'border-top:1px dashed rgba(255,255,255,0.08)"></div>',
            unsafe_allow_html=True,
        )
        _show_exp = st.checkbox(
            "🧪 Show experimental filters (failed walk-forward)",
            value=bool(settings.get("v2_team_roi_filter", False)
                       or settings.get("v2_drawdown_throttle", False)),
            key="p2_show_experimental",
            help=("These filters looked promising in Phase 1 diagnostics but did "
                  "NOT improve OOS profit in walk-forward. Hidden by default."),
        )
        if _show_exp:
            st.markdown(
                '<div style="font-size:0.82rem;color:#cdd;margin-bottom:0.6rem">'
                "Toggle on if you want to experiment further; otherwise leave off "
                "and they'll persist as <code>False</code> in your settings."
                '</div>', unsafe_allow_html=True,
            )
            ef1, ef2 = st.columns(2)
            with ef1:
                new_team_filter = st.toggle(
                    "Team-ROI filter",
                    value=bool(settings.get("v2_team_roi_filter", False)),
                    key="p2_team_filter",
                    help="Blacklist teams with bad historical draw-bet ROI.",
                )
                new_team_thr = st.slider(
                    "Team-ROI threshold (%)",
                    min_value=-100, max_value=0, step=5,
                    value=int(float(settings.get("v2_team_roi_threshold", -25.0))),
                    key="p2_team_thr",
                    disabled=not new_team_filter,
                )
                new_team_min_n = st.slider(
                    "Min bets before filtering",
                    min_value=2, max_value=10, step=1,
                    value=int(settings.get("v2_team_roi_min_n", 3)),
                    key="p2_team_min_n",
                    disabled=not new_team_filter,
                )
            with ef2:
                new_dd_throttle = st.toggle(
                    "Drawdown stake throttle",
                    value=bool(settings.get("v2_drawdown_throttle", False)),
                    key="p2_dd_throttle",
                    help="Shrink stake during drawdowns.",
                )
                new_dd_at = st.slider(
                    "Throttle at drawdown (%)",
                    min_value=10, max_value=50, step=5,
                    value=int(float(settings.get("v2_drawdown_at_pct", 0.20)) * 100),
                    key="p2_dd_at",
                    disabled=not new_dd_throttle,
                )
                new_dd_min = st.slider(
                    "Min stake factor (Kelly multiplier)",
                    min_value=0.10, max_value=1.00, step=0.05,
                    value=float(settings.get("v2_drawdown_min_factor", 0.25)),
                    key="p2_dd_min",
                    disabled=not new_dd_throttle,
                )
        else:
            # Preserve existing values when the section is hidden — so saving
            # without toggling doesn't blank out user-set experimental settings.
            new_team_filter = bool(settings.get("v2_team_roi_filter", False))
            new_team_thr    = float(settings.get("v2_team_roi_threshold", -25.0))
            new_team_min_n  = int(settings.get("v2_team_roi_min_n", 3))
            new_dd_throttle = bool(settings.get("v2_drawdown_throttle", False))
            new_dd_at       = float(settings.get("v2_drawdown_at_pct", 0.20)) * 100
            new_dd_min      = float(settings.get("v2_drawdown_min_factor", 0.25))

        # ── Section: Auto-Bet ──────────────────────────────────────────
        _section_header("🤖  Auto-Bet",
                        "Automatic placement using the research-track stack")
        ab1, ab2 = st.columns(2)
        with ab1:
            new_auto_en = st.toggle("Auto-Bet enabled",
                value=settings.get("auto_bet_enabled", False), key="p2_auto",
                help="Auto-place research-track Kelly bets when API odds available")
        with ab2:
            new_auto_thr = st.slider("Auto-Bet EV (%)", 1, 60,
                int(settings.get("auto_bet_threshold", 0.40) * 100), key="p2_thr")

        # ── Section: Research-Track Stack ──────────────────────────────
        _section_header("🧪  Research-Track Stack",
                        "Mock Two's distinguishing model + sizing logic")
        sw1, sw2, sw3 = st.columns(3)
        with sw1:
            new_use_kn = st.toggle("🧬 K-N γ-inflation",
                value=settings.get("use_kn_model", True), key="p2_kn",
                help="Karlis-Ntzoufras diagonal inflation model")
        with sw2:
            new_use_uncert = st.toggle("📐 Uncertainty Kelly",
                value=settings.get("use_uncertainty_kelly", True), key="p2_uncert",
                help="Baker-McHale shrinkage by per-bin Var(p̂)")
        with sw3:
            new_use_sim = st.toggle("🔗 Simultaneous-bet Kelly",
                value=settings.get("use_simultaneous_kelly", True), key="p2_sim",
                help="Reduce stakes when multiple bets settle concurrently")

        # ── Section: API note ──────────────────────────────────────────
        _section_header("🔌  Live Odds API", "Reuses Main's API key + cache")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#8892a4;line-height:1.5">'
            f'API key + monthly usage are shared with the Main portfolio — '
            f'no separate configuration needed here.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
        save_c, reset_c, _ = st.columns([1, 1, 4])
        with save_c:
            if st.button("💾 Save Mock Two Settings", key="p2_save"):
                port2["initial_bankroll"]              = new_init
                port2["settings"]["min_ev"]            = new_min_ev / 100
                port2["settings"]["kelly_fraction"]    = new_kelly
                port2["settings"]["min_prob"]          = new_min_prob / 100
                port2["settings"]["use_kn_model"]      = new_use_kn
                port2["settings"]["use_uncertainty_kelly"]  = new_use_uncert
                port2["settings"]["use_simultaneous_kelly"] = new_use_sim
                port2["settings"]["auto_bet_enabled"]  = new_auto_en
                port2["settings"]["auto_bet_threshold"] = new_auto_thr / 100
                port2["settings"]["max_stake_pct"]     = new_max_stake / 100
                port2["settings"]["skip_late_season"]  = new_skip_late
                port2["settings"]["skip_home_title_race"] = new_skip_title
                # Phase 4 validated filters
                port2["settings"]["v2_banned_dows"]   = list(new_banned_dows)
                port2["settings"]["v2_banned_months"] = list(new_banned_months)
                port2["settings"]["v2_max_ev_pct"]    = (
                    float(new_max_ev) / 100 if new_max_ev is not None else None
                )
                # Multi-season grid winner: ELO-profile filters
                port2["settings"]["v2_min_team_elo"] = (
                    float(new_min_te) if new_min_te is not None else None
                )
                port2["settings"]["v2_max_team_elo"] = (
                    float(new_max_te) if new_max_te is not None else None
                )
                port2["settings"]["v2_elo_gap_min"]  = (
                    float(new_gap_min) if new_gap_min is not None else None
                )
                port2["settings"]["v2_elo_gap_max"]  = (
                    float(new_gap_max) if new_gap_max is not None else None
                )
                # Experimental (failed WF)
                port2["settings"]["v2_team_roi_filter"]    = bool(new_team_filter)
                port2["settings"]["v2_team_roi_threshold"] = float(new_team_thr)
                port2["settings"]["v2_team_roi_min_n"]     = int(new_team_min_n)
                port2["settings"]["v2_drawdown_throttle"]  = bool(new_dd_throttle)
                port2["settings"]["v2_drawdown_at_pct"]    = float(new_dd_at) / 100
                port2["settings"]["v2_drawdown_min_factor"] = float(new_dd_min)
                # Per-market U2.5 gates: when on, register the gates AND add to auto_markets
                if new_u25_gates2:
                    port2["settings"]["market_gates"] = {
                        "under25": {"min_prob": 0.50, "min_ev": 0.05},
                    }
                    _existing_mkts = port2["settings"].get("auto_markets", ["D"])
                    if "under25" not in _existing_mkts:
                        _existing_mkts = list(_existing_mkts) + ["under25"]
                        port2["settings"]["auto_markets"] = _existing_mkts
                else:
                    port2["settings"].pop("market_gates", None)
                pf.save_portfolio_two(port2)
                st.success("✅ Mock Two settings saved.")
                st.rerun()
        with reset_c:
            if st.session_state.get("_p2_confirm_reset", False):
                yc, nc = st.columns(2)
                with yc:
                    if st.button("Reset", key="p2_reset_yes", type="primary"):
                        fresh = {
                            "initial_bankroll": new_init,
                            "bankroll":         new_init,
                            "bets":             [],
                            "settings": port2["settings"],
                        }
                        pf.save_portfolio_two(fresh)
                        st.session_state["_p2_confirm_reset"] = False
                        st.rerun()
                with nc:
                    if st.button("Cancel", key="p2_reset_no"):
                        st.session_state["_p2_confirm_reset"] = False
                        st.rerun()
            else:
                if st.button("🔄 Reset Mock Two", key="p2_reset"):
                    st.session_state["_p2_confirm_reset"] = True
                    st.rerun()

    # ── Auto-bet (research-track stack) ────────────────────────────────────
    auto_enabled   = settings.get("auto_bet_enabled", False)
    auto_threshold = float(settings.get("auto_bet_threshold", 0.03))
    if auto_enabled and api_key.strip():
        _auto_odds = pf.fetch_live_odds(api_key)   # shares cache with main
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
                # Use K-N model for Mock Two predictions
                _, _, _res = full_predict_v2(
                    _h, _a, dc_kn_r, dc_draw_r, xgb_m, feat_cols,
                    draw_xgb_m, draw_fc, _hs, _as,
                )
                _ds = _fix["date"].isoformat()
                _p_o25 = _res.get("over_25", 0.5)
                _pinn  = _api_o.get("_pinnacle") if isinstance(_api_o, dict) else None
                for _mkt, _prob, _lbl in [
                    ("H",      _res["home_win"], f"Home Win ({_h})"),
                    ("D",      _res["draw"],     "Draw"),
                    ("A",      _res["away_win"], f"Away Win ({_a})"),
                    ("over25", _p_o25,           "Over 2.5 Goals"),
                    ("under25", 1 - _p_o25,      "Under 2.5 Goals"),
                ]:
                    _place_odds  = _api_o.get(_mkt)
                    _detect_odds = _pinn.get(_mkt) if _pinn else None
                    # EV reported on candidate row uses detect_odds when sharp ref is available
                    _ev_ref = _detect_odds if (_detect_odds and _detect_odds > 1) else _place_odds
                    _candidates.append({
                        "home": _h, "away": _a, "date": _ds,
                        "market": _mkt, "selection": _lbl,
                        "model_prob": _prob,
                        "odds": _place_odds,
                        "detect_odds": _detect_odds,
                        "home_elo": elo_dict.get(_h),
                        "away_elo": elo_dict.get(_a),
                        "ev": pf.compute_ev(_prob, _ev_ref) if _ev_ref else -1,
                    })
            _placed = pf.auto_place_value_bets_v2(
                port2, _candidates, auto_threshold,
                calibrators=calibrators, bin_variances=bin_vars,
            )
            if _placed:
                pf.save_portfolio_two(port2)
                st.success(
                    f"🧪 Mock Two auto-placed {len(_placed)} bet"
                    f"{'s' if len(_placed) > 1 else ''} "
                    f"with EV ≥ +{int(auto_threshold*100)}%! "
                    + " · ".join(f"{b['selection']} @ {b['odds']}" for b in _placed)
                )
                mt_stats = pf.portfolio_stats(port2)

    # ── Mock Two stats row ────────────────────────────────────────────────
    profit2   = mt_stats["profit"]
    bankroll2 = mt_stats["bankroll"]
    roi2      = mt_stats["roi"]
    pending2  = [b for b in port2["bets"] if b["status"] == "pending"]
    pending_stake2 = sum(b["stake"] for b in pending2)
    sign2     = "+" if profit2 >= 0 else ""

    if mt_stats["n_settled"] == 0 and not pending2:
        hero_class2 = "pnl-neutral"
    elif profit2 >= 0:
        hero_class2 = "pnl-profit"
    else:
        hero_class2 = "pnl-loss"
    arrow2 = "▲" if profit2 >= 0 else "▼"

    st.markdown(f"""
    <div class="pnl-hero {hero_class2}">
        <div class="pnl-tag">🧪 MOCK PORTFOLIO TWO · RESEARCH TRACK · NOT REAL MONEY</div>
        <div class="pnl-amount">{sign2}£{abs(profit2):,.2f}</div>
        <div class="pnl-subtitle">
            {arrow2} {sign2}{roi2:.1f}% ROI &nbsp;·&nbsp;
            £{bankroll2:,.2f} bankroll &nbsp;·&nbsp;
            {mt_stats['n_pending']} pending (£{pending_stake2:.0f} at risk)
        </div>
    </div>""", unsafe_allow_html=True)

    def _stat2(val, lbl, color="#e8eaf0"):
        return (f'<div class="pstat-card">'
                f'<div class="pstat-val" style="color:{color}">{val}</div>'
                f'<div class="pstat-lbl">{lbl}</div></div>')
    roi_col2 = "#00e676" if roi2 >= 0 else "#ff4081"
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(_stat2(f"£{bankroll2:,.0f}", "BANKROLL"), unsafe_allow_html=True)
    with c2: st.markdown(_stat2(str(mt_stats["n_settled"]), "SETTLED"), unsafe_allow_html=True)
    with c3: st.markdown(_stat2(f"{mt_stats['win_rate']:.0f}%", "WIN RATE"), unsafe_allow_html=True)
    with c4: st.markdown(_stat2(f"{sign2}{roi2:.1f}%", "ROI", roi_col2), unsafe_allow_html=True)
    with c5: st.markdown(_stat2(f"+{mt_stats['avg_ev_pct']:.1f}%" if mt_stats["avg_ev_pct"] >= 0
                                else f"{mt_stats['avg_ev_pct']:.1f}%",
                                "AVG MODEL EV", "#7c4dff"), unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin:1.2rem 0"></div>', unsafe_allow_html=True)

    # ── Mock Two bankroll history chart (Main parity) ─────────────────────
    history2 = pf.bankroll_history(port2)
    if len(history2) >= 2:
        st.markdown('<p class="section-label">📈  Mock Two Bankroll Journey</p>',
                    unsafe_allow_html=True)
        init_br_p2 = float(port2["initial_bankroll"])
        x_vals_p2  = history2["idx"].tolist()
        y_vals_p2  = history2["bankroll"].tolist()
        labels_p2  = history2["label"].tolist()
        line_col_p2 = "#00e676" if y_vals_p2[-1] >= init_br_p2 else "#ff4081"

        fig_p2 = go.Figure()

        # Insert baseline-crossing points so green/red fills don't bleed
        x_exp_p2: list[float] = [x_vals_p2[0]]
        y_exp_p2: list[float] = [y_vals_p2[0]]
        for i in range(1, len(y_vals_p2)):
            y_prev, y_cur = y_vals_p2[i - 1], y_vals_p2[i]
            if (y_prev - init_br_p2) * (y_cur - init_br_p2) < 0:
                t = (init_br_p2 - y_prev) / (y_cur - y_prev)
                x_cross = x_vals_p2[i - 1] + t * (x_vals_p2[i] - x_vals_p2[i - 1])
                x_exp_p2.append(x_cross); y_exp_p2.append(init_br_p2)
            x_exp_p2.append(x_vals_p2[i]); y_exp_p2.append(y_cur)

        y_up_p2 = [max(v, init_br_p2) for v in y_exp_p2]
        y_dn_p2 = [min(v, init_br_p2) for v in y_exp_p2]
        baseline_p2 = [init_br_p2] * len(y_exp_p2)

        # Green above
        fig_p2.add_trace(go.Scatter(x=x_exp_p2, y=baseline_p2, mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False))
        fig_p2.add_trace(go.Scatter(x=x_exp_p2, y=y_up_p2, mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor="rgba(0,230,118,0.22)",
            hoverinfo="skip", showlegend=False))
        # Red below
        fig_p2.add_trace(go.Scatter(x=x_exp_p2, y=baseline_p2, mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False))
        fig_p2.add_trace(go.Scatter(x=x_exp_p2, y=y_dn_p2, mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor="rgba(255,64,129,0.22)",
            hoverinfo="skip", showlegend=False))

        # Baseline line + label
        fig_p2.add_hline(
            y=init_br_p2, line_color="rgba(255,255,255,0.30)", line_dash="dot",
            annotation_text=f"Start £{init_br_p2:,.0f}",
            annotation_font=dict(color="#8892a4", size=14, family="Inter"),
            annotation_position="top left",
        )

        # Halo + main line with win/loss markers
        deltas_p2 = [0.0] + [y_vals_p2[i] - y_vals_p2[i - 1]
                              for i in range(1, len(y_vals_p2))]
        marker_symbols_p2 = ["circle"] + [
            "triangle-up" if d > 0 else ("triangle-down" if d < 0 else "circle-open")
            for d in deltas_p2[1:]
        ]
        marker_colors_p2 = ["#8892a4"] + [
            "#00e676" if d > 0 else ("#ff4081" if d < 0 else "#8892a4")
            for d in deltas_p2[1:]
        ]
        marker_sizes_p2 = [0] + [13 if d != 0 else 6 for d in deltas_p2[1:]]
        hover_texts_p2 = ["Start"] + [
            ("▲ WON " + f"+£{d:,.2f}") if d > 0 else
            ("▼ LOST " + f"−£{abs(d):,.2f}") if d < 0 else
            "No change"
            for d in deltas_p2[1:]
        ]

        fig_p2.add_trace(go.Scatter(x=x_vals_p2, y=y_vals_p2, mode="lines",
            line=dict(color=f"rgba({_hex_to_rgb(line_col_p2)},0.30)", width=10),
            hoverinfo="skip", showlegend=False))
        fig_p2.add_trace(go.Scatter(
            x=x_vals_p2, y=y_vals_p2, mode="lines+markers",
            line=dict(color=line_col_p2, width=3.5, shape="linear"),
            marker=dict(symbol=marker_symbols_p2, size=marker_sizes_p2,
                        color=marker_colors_p2,
                        line=dict(color="#0a0e1a", width=1.5)),
            text=hover_texts_p2,
            hovertemplate=("<b>Bet %{x}</b><br>%{text}<br>"
                           "<b style='font-size:14px'>Bankroll £%{y:,.2f}</b>"
                           "<extra></extra>"),
            hoverlabel=dict(bgcolor="#1a1d27", bordercolor=line_col_p2,
                            font=dict(size=14, family="Inter", color="#e8eaf0")),
            showlegend=False,
        ))

        # Peak / Low / NOW badges
        peak_idx_p2   = int(np.argmax(y_vals_p2))
        trough_idx_p2 = int(np.argmin(y_vals_p2))
        if peak_idx_p2 > 0 and y_vals_p2[peak_idx_p2] > init_br_p2 * 1.05:
            fig_p2.add_annotation(
                x=x_vals_p2[peak_idx_p2], y=y_vals_p2[peak_idx_p2],
                text=f"<b>Peak</b><br>£{y_vals_p2[peak_idx_p2]:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor="#00e676",
                arrowsize=1.2, arrowwidth=1.5, ax=0, ay=-38,
                font=dict(size=12, color="#00e676", family="Inter"),
                bgcolor="rgba(0,230,118,0.10)",
                bordercolor="rgba(0,230,118,0.4)",
                borderpad=4, borderwidth=1,
            )
        if (trough_idx_p2 > 0 and y_vals_p2[trough_idx_p2] < init_br_p2 * 0.95
                and trough_idx_p2 != peak_idx_p2):
            fig_p2.add_annotation(
                x=x_vals_p2[trough_idx_p2], y=y_vals_p2[trough_idx_p2],
                text=f"<b>Low</b><br>£{y_vals_p2[trough_idx_p2]:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor="#ff4081",
                arrowsize=1.2, arrowwidth=1.5, ax=0, ay=38,
                font=dict(size=12, color="#ff4081", family="Inter"),
                bgcolor="rgba(255,64,129,0.10)",
                bordercolor="rgba(255,64,129,0.4)",
                borderpad=4, borderwidth=1,
            )
        fig_p2.add_annotation(
            x=x_vals_p2[-1], y=y_vals_p2[-1],
            text=f"<b>NOW · £{y_vals_p2[-1]:,.0f}</b>",
            showarrow=False, xshift=15,
            font=dict(size=14, color="#fff", family="Inter"),
            bgcolor=line_col_p2, bordercolor=line_col_p2,
            borderpad=8, borderwidth=2, xanchor="left",
        )

        # ── Mock Two pending bets projection ───────────────────────────
        # Same projection logic as Main: starts from LAST SETTLED bet's
        # bankroll (not current cash) so the lines visually continue from
        # the chart. Math: profit_if_won = stake*(odds-1), loss = -stake.
        # Terminal values match the gross-return formulation.
        pending2_for_proj = [b for b in port2["bets"]
                             if b["status"] == "pending" and b.get("type") != "acca"]
        if pending2_for_proj:
            n_pend_p2 = len(pending2_for_proj)
            pending_sorted_p2 = sorted(
                pending2_for_proj,
                key=lambda b: (b.get("date") or "", b.get("placed_at") or ""),
            )
            last_settled_p2 = y_vals_p2[-1]  # already the last settled bankroll
            x_proj_p2 = list(range(x_vals_p2[-1], x_vals_p2[-1] + n_pend_p2 + 1))
            best_y_p2  = [last_settled_p2]
            exp_y_p2   = [last_settled_p2]
            worst_y_p2 = [last_settled_p2]
            for b in pending_sorted_p2:
                stake = float(b["stake"])
                odds  = float(b["odds"])
                p_win = float(b.get("model_prob") or 0.0)
                profit_win  = stake * (odds - 1)
                profit_loss = -stake
                exp_change  = p_win * profit_win + (1.0 - p_win) * profit_loss
                best_y_p2.append(round(best_y_p2[-1]   + profit_win,  2))
                worst_y_p2.append(round(worst_y_p2[-1] + profit_loss, 2))
                exp_y_p2.append (round(exp_y_p2[-1]    + exp_change,  2))

            # Best (all pending win)
            fig_p2.add_trace(go.Scatter(
                x=x_proj_p2, y=best_y_p2, mode="lines+markers",
                line=dict(color="#00e676", width=2.2, dash="dot"),
                marker=dict(size=[0] + [8] * n_pend_p2,
                            color="#00e676", symbol="diamond"),
                hovertemplate=("<b>Pending #%{x}</b><br>"
                               "If all win → £%{y:,.2f}<extra></extra>"),
                showlegend=False,
            ))
            # Expected (model-weighted)
            fig_p2.add_trace(go.Scatter(
                x=x_proj_p2, y=exp_y_p2, mode="lines+markers",
                line=dict(color="#00e5ff", width=2.5, dash="dash"),
                marker=dict(size=[0] + [8] * n_pend_p2,
                            color="#00e5ff", symbol="circle"),
                hovertemplate=("<b>Pending #%{x}</b><br>"
                               "Expected → £%{y:,.2f}<extra></extra>"),
                showlegend=False,
            ))
            # Worst (all lose)
            fig_p2.add_trace(go.Scatter(
                x=x_proj_p2, y=worst_y_p2, mode="lines+markers",
                line=dict(color="#ff4081", width=2.2, dash="dot"),
                marker=dict(size=[0] + [8] * n_pend_p2,
                            color="#ff4081", symbol="x"),
                hovertemplate=("<b>Pending #%{x}</b><br>"
                               "If all lose → £%{y:,.2f}<extra></extra>"),
                showlegend=False,
            ))
            # Endpoint annotations
            for y_val, color, label in (
                (best_y_p2[-1],  "#00e676", "If all win"),
                (exp_y_p2[-1],   "#00e5ff", "Expected"),
                (worst_y_p2[-1], "#ff4081", "If all lose"),
            ):
                fig_p2.add_annotation(
                    x=x_proj_p2[-1], y=y_val,
                    text=f"<b>{label}<br>£{y_val:,.0f}</b>",
                    showarrow=False, xshift=12,
                    font=dict(size=11, color=color, family="Inter"),
                    bgcolor=f"rgba({_hex_to_rgb(color)},0.10)",
                    bordercolor=f"rgba({_hex_to_rgb(color)},0.4)",
                    borderpad=5, borderwidth=1, xanchor="left",
                )
            # Vertical separator settled → pending
            fig_p2.add_vline(
                x=x_vals_p2[-1], line_color="rgba(255,255,255,0.15)",
                line_dash="dot",
            )
            fig_p2.add_annotation(
                x=x_vals_p2[-1], y=1.0, yref="paper",
                text="settled  →  pending",
                showarrow=False, yshift=-6,
                font=dict(size=11, color="#7c4dff", family="Inter"),
            )

        fig_p2.update_layout(
            **{k: v for k, v in DARK.items() if k != "margin"},
            height=460, showlegend=False,
            margin=dict(t=40, b=40, l=20, r=140),
            xaxis=dict(
                title=dict(text="BET NUMBER",
                           font=dict(size=12, color="#7c4dff", family="Inter"),
                           standoff=18),
                showgrid=False, showticklabels=True,
                tickfont=dict(size=13, color="#8892a4", family="Inter"),
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="BANKROLL",
                           font=dict(size=12, color="#7c4dff", family="Inter"),
                           standoff=14),
                gridcolor="rgba(255,255,255,0.05)",
                tickprefix="£",
                tickfont=dict(size=14, color="#cdd", family="Inter"),
                zeroline=False, tickformat=",.0f",
            ),
        )
        st.plotly_chart(fig_p2, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('<div class="divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    # ── Mock Two CLV diagnostics ──────────────────────────────────────────
    if mt_clv["n"] > 0:
        st.markdown('<p class="section-label">📐  CLV vs Pinnacle close · Mock Two</p>',
                    unsafe_allow_html=True)
        _med  = mt_clv["median_clv"] * 100
        _mean = mt_clv["mean_clv"] * 100
        _pos  = mt_clv["pct_positive"]
        _med_col = "#00e676" if _med >= 1.0 else ("#ffd600" if _med >= 0 else "#ff4081")
        cv1, cv2, cv3, cv4 = st.columns(4)
        with cv1: st.markdown(_stat2(f"{_med:+.2f}%", "MEDIAN CLV", _med_col), unsafe_allow_html=True)
        with cv2: st.markdown(_stat2(f"{_mean:+.2f}%", "MEAN CLV", _med_col), unsafe_allow_html=True)
        with cv3: st.markdown(_stat2(f"{_pos:.0f}%", "% POSITIVE",
            "#00e676" if _pos >= 55 else "#ffd600" if _pos >= 45 else "#ff4081"), unsafe_allow_html=True)
        with cv4: st.markdown(_stat2(f"{mt_clv['n']}", "TAGGED"), unsafe_allow_html=True)
        _render_clv_trend(port2, key_prefix="mt")
        st.markdown('<div class="divider" style="margin:1rem 0"></div>', unsafe_allow_html=True)

    # ── Mock Two Pending Bets — pend-card-v2 style (parity with Main) ────
    pending2_singles = [b for b in pending2 if b.get("type") != "acca"]
    if pending2_singles:
        st.markdown('<p class="section-label">⏳  MOCK TWO PENDING BETS</p>',
                    unsafe_allow_html=True)
        mkt_colors_p2 = {"H": "#3d6eff", "D": "#ffd600", "A": "#ff4081",
                         "over25": "#7c4dff", "under25": "#00e5ff"}

        def _render_p2_pending_card(bet: dict) -> None:
            ev_pct = bet.get("ev", 0) * 100
            _pot_ret    = round(bet["stake"] * bet["odds"], 2)
            _pot_profit = round(_pot_ret - bet["stake"], 2)
            mc = mkt_colors_p2.get(bet["market"], "#aab")
            try:
                _d = pd.to_datetime(bet.get("date") or bet.get("placed_at"))
                _date_short = _d.strftime("%d-%b-%y")
            except Exception:
                _date_short = (bet.get("date") or "")[:10]
            _model_p    = bet.get("model_prob", 0) * 100
            _profit_pct = round((_pot_profit / max(bet["stake"], 0.01)) * 100, 0)
            # v2 audit chips — only render when the stack actually adjusted
            audit_chips = []
            if bet.get("v2_kelly_shrinkage") is not None:
                audit_chips.append(
                    f'<span class="pend-v2-stat">'
                    f'<div class="pend-v2-stat-lbl">SHRINK</div>'
                    f'<div class="pend-v2-stat-val pend-v2-model">'
                    f'{bet["v2_kelly_shrinkage"]:.2f}</div></span>'
                )
            if bet.get("v2_sim_factor") is not None and abs(bet["v2_sim_factor"] - 1.0) > 0.001:
                audit_chips.append(
                    f'<span class="pend-v2-stat">'
                    f'<div class="pend-v2-stat-lbl">SIM ×</div>'
                    f'<div class="pend-v2-stat-val pend-v2-model">'
                    f'{bet["v2_sim_factor"]:.2f}</div></span>'
                )
            audit_html = "".join(audit_chips)
            st.markdown(f"""
            <div class="pend-card-v2">
                <div class="pend-v2-top">
                    <span class="pend-v2-date">📅 {_date_short}</span>
                    <span class="pend-v2-badge" style="color:{mc};border-color:{mc}55;background:rgba({_hex_to_rgb(mc)},0.10)">{bet['market'].upper()}</span>
                </div>
                <div class="pend-v2-teams">
                    <div class="pend-v2-team"><img class="team-badge" src="{_BADGE_URL.get(bet['home'], '')}" width="56" /><span>{bet['home']}</span></div>
                    <div class="pend-v2-vs">VS</div>
                    <div class="pend-v2-team"><img class="team-badge" src="{_BADGE_URL.get(bet['away'], '')}" width="56" /><span>{bet['away']}</span></div>
                </div>
                <div class="pend-v2-pick" style="color:{mc}">
                    🎯 {bet['selection']} <span class="pend-v2-at">@</span> <span class="pend-v2-odds">{bet['odds']:.2f}</span>
                </div>
                <div class="pend-v2-stats">
                    <div class="pend-v2-stat">
                        <div class="pend-v2-stat-lbl">STAKE</div>
                        <div class="pend-v2-stat-val">£{bet['stake']:,.2f}</div>
                    </div>
                    <div class="pend-v2-stat">
                        <div class="pend-v2-stat-lbl">MODEL</div>
                        <div class="pend-v2-stat-val pend-v2-model">{_model_p:.1f}%</div>
                    </div>
                    <div class="pend-v2-stat">
                        <div class="pend-v2-stat-lbl">EV</div>
                        <div class="pend-v2-stat-val pend-v2-ev">+{ev_pct:.1f}%</div>
                    </div>
                    {audit_html}
                </div>
                <div class="pend-v2-return">
                    <div class="pend-v2-return-block">
                        <div class="pend-v2-return-lbl">IF IT WINS</div>
                        <div class="pend-v2-return-val">£{_pot_ret:,.2f}</div>
                    </div>
                    <div class="pend-v2-arrow">→</div>
                    <div class="pend-v2-return-block">
                        <div class="pend-v2-return-lbl">PROFIT</div>
                        <div class="pend-v2-return-profit">+£{_pot_profit:,.2f}</div>
                        <div class="pend-v2-return-pct">+{_profit_pct:.0f}% on stake</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button("✕ Cancel", key=f"p2_cancel_{bet['id']}", type="secondary"):
                pf.remove_pending_bet(port2, bet["id"])
                pf.save_portfolio_two(port2)
                st.rerun()

        # Two cards per row, latest 8
        recent_p2 = pending2_singles[-8:]
        for i in range(0, len(recent_p2), 2):
            cols = st.columns(2)
            with cols[0]:
                _render_p2_pending_card(recent_p2[i])
            if i + 1 < len(recent_p2):
                with cols[1]:
                    _render_p2_pending_card(recent_p2[i + 1])
        st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Mock Two Bet History — bh-row style (parity with Main) ─────────
    settled2 = [b for b in port2["bets"]
                if b["status"] in ("won", "lost") and b.get("type") != "acca"]
    if settled2:
        st.markdown('<p class="section-label" id="bet-history-p2">📋  MOCK TWO BET HISTORY · Singles</p>',
                    unsafe_allow_html=True)

        chrono_p2 = sorted(settled2,
                           key=lambda b: (b.get("settled_at") or b.get("date") or "",
                                          b.get("placed_at") or ""))
        chrono_p2_idx = [(i + 1, b) for i, b in enumerate(chrono_p2)]

        rows_html_p2 = []
        for bet_idx, b in reversed(chrono_p2_idx):
            profit_b = b["profit"] or 0.0
            stake_b  = b["stake"]
            odds_b   = b["odds"]
            model_p  = b.get("model_prob") or 0.0
            implied  = (1.0 / odds_b) if odds_b > 0 else 0.0
            edge_pp  = (model_p - implied) * 100

            try:
                d = pd.to_datetime(b.get("date") or b.get("placed_at"))
                date_short = d.strftime("%d-%b-%y")
            except Exception:
                date_short = (b.get("date") or "")[:10]

            home, away  = b["home"], b["away"]
            sel_label   = b["selection"]
            won = b["status"] == "won"
            res_html = (f'<span class="bh-result bh-won">✅ WON</span>' if won
                        else f'<span class="bh-result bh-lost">❌ LOST</span>')
            pnl_cls = ("bh-pnl-pos" if profit_b > 0
                       else ("bh-pnl-neg" if profit_b < 0 else "bh-pnl-flat"))
            pnl_sign = "+" if profit_b >= 0 else "−"
            pnl_html = (f'<span class="bh-pnl {pnl_cls}">'
                        f'{pnl_sign}£{abs(profit_b):,.2f}</span>')

            edge_col  = "#00e676" if edge_pp >= 5 else ("#ffd600" if edge_pp >= 0 else "#ff4081")
            edge_html = (
                f'<div class="bh-edge" title="Model probability vs bookmaker implied probability.">'
                f'  <div class="bh-edge-row">'
                f'    <span class="bh-edge-lbl">model</span>'
                f'    <span class="bh-edge-num" style="color:#a78bfa">{model_p*100:.1f}%</span>'
                f'  </div>'
                f'  <div class="bh-edge-row">'
                f'    <span class="bh-edge-lbl">bookie</span>'
                f'    <span class="bh-edge-num">{implied*100:.1f}%</span>'
                f'  </div>'
                f'  <div class="bh-edge-gap" style="color:{edge_col}">+{edge_pp:.1f}pp edge</div>'
                f'</div>'
            )

            highlight_idx = st.session_state.get("_bh_p2_highlight")
            row_cls = "bh-row" + (" bh-row-highlight" if highlight_idx == bet_idx else "")

            rows_html_p2.append(
                f'<div class="{row_cls}" id="bet-row-p2-{bet_idx}">'
                f'  <div class="bh-num">#{bet_idx}</div>'
                f'  <div class="bh-date">{date_short}</div>'
                f'  <div class="bh-team bh-home">{tb(home, 32)}</div>'
                f'  <div class="bh-team bh-away">{tb(away, 32)}</div>'
                f'  <div class="bh-sel">{sel_label}</div>'
                f'  <div class="bh-odds">{odds_b:.2f}</div>'
                f'  {edge_html}'
                f'  <div class="bh-stake">£{stake_b:,.2f}</div>'
                f'  <div class="bh-cell-result">{res_html}</div>'
                f'  <div class="bh-cell-pnl">{pnl_html}</div>'
                f'</div>'
            )

        header_html_p2 = (
            '<div class="bh-row bh-header">'
            '<div class="bh-num">#</div>'
            '<div class="bh-date">DATE</div>'
            '<div class="bh-team">HOME</div>'
            '<div class="bh-team">AWAY</div>'
            '<div class="bh-sel">PICK</div>'
            '<div class="bh-odds">ODDS</div>'
            '<div class="bh-edge">WHY WE BET</div>'
            '<div class="bh-stake">STAKE</div>'
            '<div class="bh-cell-result">RESULT</div>'
            '<div class="bh-cell-pnl">P&amp;L</div>'
            '</div>'
        )

        st.markdown(
            '<div class="bh-scroll"><div class="bh-table">'
            + header_html_p2 + "".join(rows_html_p2)
            + '</div></div>',
            unsafe_allow_html=True,
        )
    elif not pending2:
        st.markdown(
            '<div style="padding:1rem 1.2rem;background:rgba(124,77,255,0.05);'
            'border-left:3px solid #7c4dff;border-radius:6px;font-size:0.84rem;color:#8892a4">'
            'Mock Two is fresh — no bets placed yet. Toggle <b>Auto-Bet</b> in settings '
            'and ensure the Odds API key is set in the main portfolio. The research-track '
            'stack will auto-place qualifying bets on the same fixtures the main portfolio '
            'considers, but with K-N predictions and uncertainty-shrunk Kelly sizing.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

    # ── Historical EV Backtest — Mock Two engine ─────────────────────────
    with st.expander("📜  Historical EV Backtest · Mock Two engine (K-N + uncertainty-Kelly)",
                     expanded=False):
        st.markdown("""
        <div style="font-size:0.82rem;color:#b8c0d0;margin-bottom:1rem;line-height:1.6">
            Replays the same B365 historical odds as the Main backtest but routes them through
            the <b style="color:#a78bfa">research-track stack</b>: Karlis-Ntzoufras γ-inflated
            Dixon-Coles for probabilities, Baker-McHale uncertainty-shrunk Kelly for sizing,
            and Busseti-Ryu-Boyd simultaneous-bet correction across same-day cards.
            Side-by-side with the Main result, this answers: <i>does the research stack
            actually beat the production stack on identical matches?</i>
            Per-bin variance is fitted in-sample on the same window — directionally honest,
            but not a true held-out estimator.
        </div>""", unsafe_allow_html=True)

        # ── Quick presets ─────────────────────────────────────────────
        _render_backtest_presets("hbt2")

        # 2025-26 representativeness note — current squads & ELO ratings
        st.markdown(
            '<div style="background:linear-gradient(135deg,rgba(0,229,255,0.07),rgba(124,77,255,0.04));'
            'border-left:3px solid #00e5ff;border-radius:8px;padding:0.7rem 1rem;'
            'margin-bottom:0.9rem;font-size:0.82rem;color:#cdd;line-height:1.5">'
            '💡 <b style="color:#00e5ff">Multi-season backtests:</b> '
            "test_weeks ≥ 80 spans more than one season. Recent seasons are "
            "the most representative — older data includes teams that have "
            "since been relegated/promoted, and ELO ratings stabilise as more "
            "matches accumulate. <b>2025-26 is the most predictive of next "
            "season's behaviour.</b>"
            '</div>', unsafe_allow_html=True,
        )

        h2c1, h2c2, h2c3, h2c4, h2c5, h2c6 = st.columns(6)
        with h2c1:
            hbt2_weeks = st.slider("Test Window (weeks)", 8, 200, 52,
                                    key="hbt2_weeks",
                                    help="52 = one season. 80–104 covers 1–2 seasons. "
                                         "150+ covers 3+ seasons but training window "
                                         "shrinks (we only have ~5 seasons total).")
        with h2c2:
            hbt2_min_ev = st.slider("Min EV (%)", 1, 60,
                                     int(float(settings.get("min_ev", 0.40)) * 100),
                                     key="hbt2_minev")
        with h2c3:
            hbt2_min_prob = st.slider(
                "Min Prob Gate (%)", 0, 80,
                int(settings.get("min_prob", 0.30) * 100),
                key="hbt2_minprob",
            )
        with h2c4:
            _kf = float(settings.get("kelly_fraction", 1.0))
            _kf_options = [0.25, 0.5, 0.75, 1.0]
            _kf_default = min(_kf_options, key=lambda x: abs(x - _kf))
            hbt2_kelly = st.select_slider(
                "Base Kelly (pre-shrinkage)", _kf_options, _kf_default,
                key="hbt2_kelly", format_func=lambda x: f"{int(x*100)}%",
            )
        with h2c5:
            hbt2_max_stake = st.slider(
                "Max Stake (% of bankroll)", 5, 50,
                int(settings.get("max_stake_pct", 0.33) * 100),
                key="hbt2_maxstake",
            )
        with h2c6:
            hbt2_bankroll = st.number_input(
                "Bankroll (£)", 100.0, 100000.0, 10000.0, 1000.0,
                key="hbt2_bankroll", format="%.0f",
            )

        sc1, sc2, sc3, sc4 = st.columns([1.2, 1, 1.3, 1])
        with sc1:
            hbt2_sim = st.checkbox("Simultaneous-bet correction",
                                   value=bool(settings.get("use_simultaneous_kelly", True)),
                                   key="hbt2_sim")
        with sc2:
            hbt2_skip_late = st.checkbox(
                "Skip Mar-Apr",
                value=bool(settings.get("skip_late_season", True)),
                key="hbt2_skip_late",
                help="0/7 wins in March-April across 2024-25 + 2025-26. "
                     "(May was previously bundled in but is now allowed.)",
            )
        with sc3:
            hbt2_skip_title = st.checkbox(
                "Skip home_title_race",
                value=bool(settings.get("skip_home_title_race", False)),
                key="hbt2_skip_title",
                help="0/6 wins when home team chasing title (2025-26).",
            )
        with sc4:
            # Default place-source to Max (multi-season grid winner used Max + PS)
            _odds_options = ["B365", "Max", "Avg", "PS"]
            hbt2_odds_src = st.selectbox(
                "Place at",
                options=_odds_options,
                index=_odds_options.index("Max"),
                key="hbt2_odds_src",
                help="The price you actually win at if your bet hits. "
                     "Multi-season grid winner places at Max.",
            )

        # Default detect-source to PS (multi-season grid winner detects at PS)
        _detect_options = ["(same as place)", "B365", "Max", "Avg", "PS"]
        hbt2_detect_src = st.selectbox(
            "Detect EV against (optional — leave 'same as place' for single-source)",
            options=_detect_options,
            index=_detect_options.index("PS"),
            key="hbt2_detect_src",
            help="Set to PS for sharp-edge detection (best CLV in WF: +2.68%). "
                 "Multi-season grid winner detects at PS, places at Max.",
        )

        # Markets row — U2.5 toggle (same as Main expander)
        m2c1, m2c2 = st.columns([1.5, 3])
        with m2c1:
            hbt2_u25_gates = st.checkbox(
                "Include Under 2.5 (separate gates)",
                value="under25" in (settings.get("market_gates") or {}),
                key="hbt2_u25_gates",
                help="Adds U2.5 to the simulated market set with mp=50%, mev=5%.",
            )
        with m2c2:
            if hbt2_u25_gates:
                st.markdown(
                    '<div style="font-size:0.86rem;color:#a78bfa;padding-top:0.55rem">'
                    'Backtest will include <b>Draw + Under 2.5</b> with separate per-market gates '
                    '(U2.5: mp ≥ 50%, ev ≥ 5%).</div>',
                    unsafe_allow_html=True,
                )

        # ── Phase 4 validated filters — defaulted from saved settings ──────
        st.markdown(
            '<div style="margin:0.9rem 0 0.4rem;padding-top:0.6rem;'
            'border-top:1px dashed rgba(124,77,255,0.25);'
            'font-size:0.78rem;font-weight:800;letter-spacing:1.4px;'
            'text-transform:uppercase;color:#a78bfa">'
            '🧠 Phase 4 filters — replay with deployed config or experiment'
            '</div>', unsafe_allow_html=True,
        )
        _all_dows_bt = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _all_months_bt = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pfb1, pfb2, pfb3 = st.columns([1.2, 1.2, 1.6])
        with pfb1:
            hbt2_banned_dows = st.multiselect(
                "Banned days (DOW)",
                options=_all_dows_bt,
                default=list(settings.get("v2_banned_dows", [])),
                key="hbt2_banned_dows",
                help="Phase 4 winner: Mon+Fri.",
            )
        with pfb2:
            hbt2_banned_months = st.multiselect(
                "Banned months",
                options=_all_months_bt,
                default=list(settings.get("v2_banned_months", [])),
                key="hbt2_banned_months",
                help="Phase 4 winner: Oct.",
            )
        with pfb3:
            _max_ev_default = settings.get("v2_max_ev_pct")
            hbt2_max_ev_on = st.checkbox(
                "Cap claimed EV (overconfidence guard)",
                value=_max_ev_default is not None,
                key="hbt2_max_ev_on",
                help="Phase 1 found high-EV bucket calibrates badly.",
            )
            if hbt2_max_ev_on:
                hbt2_max_ev = st.slider(
                    "Max EV cap (%)",
                    min_value=40, max_value=200, step=5,
                    value=int((_max_ev_default if _max_ev_default is not None else 1.0) * 100),
                    key="hbt2_max_ev_val",
                )
            else:
                hbt2_max_ev = None

        # ── ELO-profile filter controls — defaulted from saved settings ──
        st.markdown(
            '<div style="margin:0.7rem 0 0.4rem;padding-top:0.6rem;'
            'border-top:1px dashed rgba(0,229,255,0.25);'
            'font-size:0.78rem;font-weight:800;letter-spacing:1.4px;'
            'text-transform:uppercase;color:#00e5ff">'
            '🎯 ELO-profile filter — multi-season grid winner: min team ELO 1500'
            '</div>', unsafe_allow_html=True,
        )
        be1, be2, be3, be4 = st.columns(4)
        with be1:
            _min_te = settings.get("v2_min_team_elo")
            hbt2_min_te_on = st.checkbox(
                "Min team ELO floor",
                value=_min_te is not None, key="hbt2_min_te_on",
                help="Skip if either team's ELO is below this. Grid winner = 1500.",
            )
            hbt2_min_te = (st.slider("Min ELO", 1300, 1700, int(_min_te or 1500),
                                      step=10, key="hbt2_min_te_val")
                           if hbt2_min_te_on else None)
        with be2:
            _max_te = settings.get("v2_max_team_elo")
            hbt2_max_te_on = st.checkbox(
                "Max team ELO ceiling",
                value=_max_te is not None, key="hbt2_max_te_on",
                help="Skip if either team's ELO is above this (rare).",
            )
            hbt2_max_te = (st.slider("Max ELO", 1700, 2100, int(_max_te or 1900),
                                      step=10, key="hbt2_max_te_val")
                           if hbt2_max_te_on else None)
        with be3:
            _gap_min = settings.get("v2_elo_gap_min")
            hbt2_gap_min_on = st.checkbox(
                "Min |ΔELO|", value=_gap_min is not None,
                key="hbt2_gap_min_on",
                help="Skip too-close matches (small ELO gap).",
            )
            hbt2_gap_min = (st.slider("Min gap", 20, 200, int(_gap_min or 80),
                                       step=10, key="hbt2_gap_min_val")
                            if hbt2_gap_min_on else None)
        with be4:
            _gap_max = settings.get("v2_elo_gap_max")
            hbt2_gap_max_on = st.checkbox(
                "Max |ΔELO|", value=_gap_max is not None,
                key="hbt2_gap_max_on",
                help="Skip lopsided matches (large ELO gap).",
            )
            hbt2_gap_max = (st.slider("Max gap", 100, 500, int(_gap_max or 300),
                                       step=20, key="hbt2_gap_max_val")
                            if hbt2_gap_max_on else None)

        if st.button("🔄  Run Mock Two Simulation", key="run_hbt2", type="primary"):
            with st.spinner("Running K-N + uncertainty-Kelly backtest…"):
                bt_v2 = backtest_models_v2(df, df_features, test_weeks=hbt2_weeks)
                _bt_markets = set(settings.get("auto_markets", list(pf.PROFITABLE_MARKETS)))
                _detect = None if hbt2_detect_src == "(same as place)" else str(hbt2_detect_src)
                # U2.5 toggle — force-add to allowed markets and apply its gates
                if hbt2_u25_gates:
                    _bt_markets = _bt_markets | {"under25"}
                    _market_gates_v2 = {"under25": {"min_prob": 0.50, "min_ev": 0.05}}
                else:
                    _market_gates_v2 = settings.get("market_gates")
                log2_df, summary2 = pf.ev_backtest_simulate_v2(
                    bt_v2, df,
                    min_ev_pct=float(hbt2_min_ev),
                    base_kelly_frac=float(hbt2_kelly),
                    max_stake_pct=float(hbt2_max_stake) / 100.0,
                    initial_bankroll=float(hbt2_bankroll),
                    allowed_markets=_bt_markets,
                    min_prob=float(hbt2_min_prob) / 100.0,
                    enable_simultaneous_correction=bool(hbt2_sim),
                    skip_late_season=bool(hbt2_skip_late),
                    skip_home_title_race=bool(hbt2_skip_title),
                    odds_source=str(hbt2_odds_src),
                    detect_source=_detect,
                    market_gates=_market_gates_v2,
                    # Honest calibration + bin variances: fitted strictly BEFORE
                    # the eval window so the backtest never sees its own outcomes
                    calibrators=cached_honest_calibrators(len(df), int(hbt2_weeks)),
                    bin_variances=cached_honest_bin_variances(len(df), int(hbt2_weeks)),
                    # Phase 4 validated filters
                    banned_dows=set(hbt2_banned_dows) if hbt2_banned_dows else None,
                    banned_months=set(hbt2_banned_months) if hbt2_banned_months else None,
                    max_ev_pct=(hbt2_max_ev / 100.0 if hbt2_max_ev is not None else None),
                    # ELO-profile filters (multi-season grid winner)
                    min_team_elo=float(hbt2_min_te) if hbt2_min_te is not None else None,
                    max_team_elo=float(hbt2_max_te) if hbt2_max_te is not None else None,
                    elo_gap_min=float(hbt2_gap_min) if hbt2_gap_min is not None else None,
                    elo_gap_max=float(hbt2_gap_max) if hbt2_gap_max is not None else None,
                    # No-history gate, mirroring live (see the Main backtest)
                    min_team_matches=settings.get("min_team_matches"),
                    df_features=df_features,
                )
                st.session_state["_hbt2_log"]     = log2_df
                st.session_state["_hbt2_summary"] = summary2

        if "_hbt2_summary" in st.session_state:
            summary2 = st.session_state["_hbt2_summary"]
            log2_df  = st.session_state["_hbt2_log"]

            if "error" in summary2:
                st.error(summary2["error"])
            else:
                sim_profit2 = summary2["profit"]
                sim_roi2    = summary2["roi"]
                sim_col2    = "#00e676" if sim_profit2 >= 0 else "#ff4081"

                m2c1, m2c2, m2c3, m2c4, m2c5 = st.columns(5)
                with m2c1: st.metric("Final Bankroll", f"£{summary2['final']:,.0f}",
                                     f"{'+' if sim_profit2 >= 0 else ''}£{sim_profit2:,.0f}")
                with m2c2: st.metric("ROI", f"{'+' if sim_roi2 >= 0 else ''}{sim_roi2:.1f}%")
                with m2c3:
                    _sk = summary2.get("skipped_min_prob", 0)
                    st.metric("Total Bets", str(summary2["n_bets"]),
                              f"−{_sk} below min-prob" if _sk else None,
                              delta_color="off")
                with m2c4: st.metric("Win Rate", f"{summary2['win_rate']:.0f}%")
                with m2c5: st.metric("Avg Odds", f"{summary2['avg_odds']:.2f}")

                # Research-stack diagnostics — what the v2 sizers actually did
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#8892a4;margin-top:0.6rem">'
                    f'Mean Baker-McHale shrinkage: <b style="color:#e8eaf0">'
                    f'{summary2.get("mean_shrinkage", 0):.2f}</b> '
                    f'(1.0 = no shrinkage, 0.0 = total) &nbsp;·&nbsp; '
                    f'Mean simultaneous-bet factor: <b style="color:#e8eaf0">'
                    f'{summary2.get("mean_sim_factor", 1):.2f}</b> '
                    f'(1.0 = solo bet, &lt;1 = multi-bet day)</div>',
                    unsafe_allow_html=True,
                )

                # Main-style bankroll chart (parity with the Main backtest):
                # green-above / red-below baseline crossings, halo line, win/loss
                # markers, peak/low/now badges, bigger axis fonts.
                if not log2_df.empty:
                    _init_br2 = summary2["initial"]
                    x_h2 = list(range(len(log2_df) + 1))
                    y_h2 = [float(_init_br2)] + [float(v) for v in log2_df["Bankroll"].tolist()]
                    line_col2 = "#00e676" if y_h2[-1] >= _init_br2 else "#ff4081"

                    fig2 = go.Figure()

                    # Insert crossing points so fills don't bleed past the line
                    x_exp2: list[float] = [x_h2[0]]
                    y_exp2: list[float] = [y_h2[0]]
                    for i in range(1, len(y_h2)):
                        y_prev, y_cur = y_h2[i - 1], y_h2[i]
                        if (y_prev - _init_br2) * (y_cur - _init_br2) < 0:
                            t = (_init_br2 - y_prev) / (y_cur - y_prev)
                            x_cross = x_h2[i - 1] + t * (x_h2[i] - x_h2[i - 1])
                            x_exp2.append(x_cross); y_exp2.append(_init_br2)
                        x_exp2.append(x_h2[i]); y_exp2.append(y_cur)

                    y_up2 = [max(v, _init_br2) for v in y_exp2]
                    y_dn2 = [min(v, _init_br2) for v in y_exp2]
                    baseline2 = [_init_br2] * len(y_exp2)

                    # Green fill above baseline
                    fig2.add_trace(go.Scatter(x=x_exp2, y=baseline2, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        hoverinfo="skip", showlegend=False))
                    fig2.add_trace(go.Scatter(x=x_exp2, y=y_up2, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        fill="tonexty", fillcolor="rgba(0,230,118,0.22)",
                        hoverinfo="skip", showlegend=False))
                    # Red fill below baseline
                    fig2.add_trace(go.Scatter(x=x_exp2, y=baseline2, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        hoverinfo="skip", showlegend=False))
                    fig2.add_trace(go.Scatter(x=x_exp2, y=y_dn2, mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        fill="tonexty", fillcolor="rgba(255,64,129,0.22)",
                        hoverinfo="skip", showlegend=False))

                    # Baseline line + label
                    fig2.add_hline(
                        y=_init_br2, line_color="rgba(255,255,255,0.30)", line_dash="dot",
                        annotation_text=f"Start £{_init_br2:,.0f}",
                        annotation_font=dict(color="#8892a4", size=14, family="Inter"),
                        annotation_position="top left",
                    )

                    # Halo + main line with win/loss markers
                    deltas2 = [0.0] + [y_h2[i] - y_h2[i - 1] for i in range(1, len(y_h2))]
                    marker_symbols2 = ["circle"] + [
                        "triangle-up" if d > 0 else ("triangle-down" if d < 0 else "circle-open")
                        for d in deltas2[1:]
                    ]
                    marker_colors2 = ["#8892a4"] + [
                        "#00e676" if d > 0 else ("#ff4081" if d < 0 else "#8892a4")
                        for d in deltas2[1:]
                    ]
                    marker_sizes2 = [0] + [11 if d != 0 else 5 for d in deltas2[1:]]
                    hover_texts2 = ["Start"] + [
                        ("▲ WON " + f"+£{d:,.2f}") if d > 0 else
                        ("▼ LOST " + f"−£{abs(d):,.2f}") if d < 0 else
                        "No change"
                        for d in deltas2[1:]
                    ]

                    fig2.add_trace(go.Scatter(x=x_h2, y=y_h2, mode="lines",
                        line=dict(color=f"rgba({_hex_to_rgb(line_col2)},0.30)", width=10),
                        hoverinfo="skip", showlegend=False))
                    fig2.add_trace(go.Scatter(
                        x=x_h2, y=y_h2, mode="lines+markers",
                        line=dict(color=line_col2, width=3.2, shape="linear"),
                        marker=dict(symbol=marker_symbols2, size=marker_sizes2,
                                    color=marker_colors2,
                                    line=dict(color="#0a0e1a", width=1.2)),
                        text=hover_texts2,
                        hovertemplate=("<b>Bet %{x}</b><br>%{text}<br>"
                                       "<b style='font-size:14px'>Bankroll £%{y:,.2f}</b>"
                                       "<extra></extra>"),
                        hoverlabel=dict(bgcolor="#1a1d27", bordercolor=line_col2,
                                        font=dict(size=14, family="Inter", color="#e8eaf0")),
                        showlegend=False,
                    ))

                    # Optional Main overlay for direct comparison
                    main_summary = st.session_state.get("_hbt_summary")
                    main_log     = st.session_state.get("_hbt_log")
                    if (main_summary and "error" not in main_summary
                            and main_log is not None and not main_log.empty):
                        fig2.add_trace(go.Scatter(
                            x=list(range(len(main_log) + 1)),
                            y=[main_summary["initial"]] + main_log["Bankroll"].tolist(),
                            mode="lines",
                            line=dict(color="#3d6eff", width=2.2, dash="dash"),
                            name="Main (DC)",
                            hovertemplate=("<b>Main bet %{x}</b><br>£%{y:,.2f}"
                                           "<extra></extra>"),
                            hoverlabel=dict(bgcolor="#1a1d27", bordercolor="#3d6eff",
                                            font=dict(size=13, family="Inter", color="#e8eaf0")),
                            showlegend=True,
                        ))

                    # Peak / Low / NOW badges
                    peak_idx2   = int(np.argmax(y_h2))
                    trough_idx2 = int(np.argmin(y_h2))
                    if peak_idx2 > 0 and y_h2[peak_idx2] > _init_br2 * 1.05:
                        fig2.add_annotation(
                            x=x_h2[peak_idx2], y=y_h2[peak_idx2],
                            text=f"<b>Peak</b><br>£{y_h2[peak_idx2]:,.0f}",
                            showarrow=True, arrowhead=2, arrowcolor="#00e676",
                            arrowsize=1.2, arrowwidth=1.5, ax=0, ay=-38,
                            font=dict(size=12, color="#00e676", family="Inter"),
                            bgcolor="rgba(0,230,118,0.10)",
                            bordercolor="rgba(0,230,118,0.4)",
                            borderpad=4, borderwidth=1,
                        )
                    if (trough_idx2 > 0 and y_h2[trough_idx2] < _init_br2 * 0.95
                            and trough_idx2 != peak_idx2):
                        fig2.add_annotation(
                            x=x_h2[trough_idx2], y=y_h2[trough_idx2],
                            text=f"<b>Low</b><br>£{y_h2[trough_idx2]:,.0f}",
                            showarrow=True, arrowhead=2, arrowcolor="#ff4081",
                            arrowsize=1.2, arrowwidth=1.5, ax=0, ay=38,
                            font=dict(size=12, color="#ff4081", family="Inter"),
                            bgcolor="rgba(255,64,129,0.10)",
                            bordercolor="rgba(255,64,129,0.4)",
                            borderpad=4, borderwidth=1,
                        )
                    fig2.add_annotation(
                        x=x_h2[-1], y=y_h2[-1],
                        text=f"<b>NOW · £{y_h2[-1]:,.0f}</b>",
                        showarrow=False, xshift=15,
                        font=dict(size=14, color="#fff", family="Inter"),
                        bgcolor=line_col2, bordercolor=line_col2,
                        borderpad=8, borderwidth=2, xanchor="left",
                    )

                    fig2.update_layout(
                        **{k: v for k, v in DARK.items() if k != "margin"},
                        height=460,
                        margin=dict(t=40, b=40, l=20, r=140),
                        xaxis=dict(
                            title=dict(text="BET NUMBER",
                                       font=dict(size=12, color="#7c4dff", family="Inter"),
                                       standoff=18),
                            showgrid=False, showticklabels=True,
                            tickfont=dict(size=13, color="#8892a4", family="Inter"),
                            zeroline=False,
                        ),
                        yaxis=dict(
                            title=dict(text="BANKROLL",
                                       font=dict(size=12, color="#7c4dff", family="Inter"),
                                       standoff=14),
                            gridcolor="rgba(255,255,255,0.05)",
                            tickprefix="£",
                            tickfont=dict(size=14, color="#cdd", family="Inter"),
                            zeroline=False, tickformat=",.0f",
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1,
                                    font=dict(size=12, family="Inter", color="#cdd"),
                                    bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig2, use_container_width=True,
                                    config={"displayModeBar": False})

                if not log2_df.empty:
                    _render_backtest_clv_trend(log2_df, df, key_prefix="hbt2")

                    # Checkbox-gated (can't nest expanders in Streamlit)
                    if st.checkbox("📋  Show Mock Two bet log",
                                   value=False, key="hbt2_show_log"):
                        st.dataframe(log2_df, use_container_width=True, hide_index=True,
                                     height=min(420, 60 + len(log2_df) * 35),
                                     column_config={
                                         "Stake":    st.column_config.NumberColumn("Stake", format="£%.2f"),
                                         "Profit":   st.column_config.NumberColumn("Profit", format="£%.2f"),
                                         "Bankroll": st.column_config.NumberColumn("Bankroll", format="£%.2f"),
                                         "Shrink":   st.column_config.NumberColumn("Shrink", format="%.2f"),
                                         "SimFactor": st.column_config.NumberColumn("Sim×",  format="%.2f"),
                                     })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Season Review (off-season wrap-up)
# ─────────────────────────────────────────────────────────────────────────────

_MKT_CHIP = {
    "D":       ("DRAW",  "#ffd600"),
    "H":       ("HOME",  "#3d6eff"),
    "A":       ("AWAY",  "#ff4081"),
    "under25": ("U2.5",  "#00e5ff"),
    "over25":  ("O2.5",  "#7c4dff"),
}

_MKT_LABEL_TO_CODE = {"Home Win": "H", "Draw": "D", "Away Win": "A",
                      "Over 2.5": "over25", "Under 2.5": "under25"}


def _render_backtest_clv_trend(log_df, df, key_prefix: str) -> None:
    """Rolling CLV-vs-Pinnacle-close for a simulated bet log.

    Answers the question the bankroll chart can't: was the edge consistent
    through the window, or one lucky cluster? Closing odds come from the
    football-data CSVs via the same fallback chain as the live CLV backfill.
    """
    if log_df.empty or "Match" not in log_df.columns:
        return
    clvs: list[float] = []
    for _, r in log_df.iterrows():
        parts = str(r["Match"]).split(" vs ")
        mkt = _MKT_LABEL_TO_CODE.get(str(r.get("Market", "")))
        if len(parts) != 2 or not mkt:
            continue
        try:
            close = pf.extract_closing_odds(parts[0].strip(), parts[1].strip(),
                                            str(r["Date"]), mkt, df)
            clv = pf.compute_clv(float(r["Odds"]), close) if close else None
        except Exception:
            clv = None
        if clv is not None:
            clvs.append(clv * 100)
    if len(clvs) < 3:
        return

    s = pd.Series(clvs)
    roll = s.rolling(5, min_periods=1).median()
    med  = float(s.median())
    beat = float((s > 0).mean() * 100)

    st.markdown('<p class="section-label" style="margin-top:1.2rem">CLV vs Pinnacle close</p>',
                unsafe_allow_html=True)
    cv1, cv2, cv3 = st.columns(3)
    with cv1: st.metric("Median CLV", f"{med:+.2f}%")
    with cv2: st.metric("Beat the close", f"{beat:.0f}% of bets")
    with cv3: st.metric("Bets with closing line", f"{len(clvs)}/{len(log_df)}")

    xs = list(range(1, len(clvs) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=clvs,
        marker_color=["#00e676" if v >= 0 else "#ff4081" for v in clvs],
        opacity=0.45, name="per-bet CLV",
        hovertemplate="Bet %{x}: %{y:+.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xs, y=roll, mode="lines",
        line=dict(color="#00e5ff", width=3), name="rolling-5 median",
        hovertemplate="Rolling median: %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_dash="dot")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter", size=12, color="#8892a4"),
        legend=dict(orientation="h", y=1.15, x=0),
        yaxis=dict(gridcolor="#1c2440", ticksuffix="%"),
        xaxis=dict(gridcolor="#1c2440", title="bet #"),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=f"{key_prefix}_clv_fig")
    st.caption("CLV = taken odds ÷ Pinnacle closing odds − 1. A consistently "
               "positive rolling median is stronger evidence of real edge than "
               "any single-window profit figure.")


def _offseason_info(df) -> dict:
    """Season state from the loaded data: whether we're between seasons and
    how long until the next one kicks off (PL openers are mid-August)."""
    last = df["Date"].max().date()
    today = date.today()
    # Mid-August is the usual opener, but the real date moves year to year
    # (2026-27 starts on the 21st). Prefer the published calendar when it's out.
    kickoff = (date(last.year, 8, 15) if last.month <= 7
               else date(last.year + 1, 8, 15))
    try:
        scheduled = [f["date"] for f in cached_fixtures() if f["date"] > last]
        if scheduled:
            kickoff = min(scheduled)
    except Exception:
        pass
    return {
        "last_match":      last,
        "is_offseason":    (today - last).days >= 14 and today < kickoff,
        "kickoff":         kickoff,
        "days_to_kickoff": max(0, (kickoff - today).days),
        "season_label":    str(df["Season"].max()),
    }


def _render_offseason_card(df, blurb: str, key_prefix: str) -> bool:
    """Replace a dead empty-state with the off-season story: season-complete
    banner, kickoff countdown, and a jump to the Season Review tab.
    Returns True when rendered (caller should skip its normal empty-state)."""
    info = _offseason_info(df)
    if not info["is_offseason"]:
        return False
    logo = (f'<img src="{PL_SYMBOL_URI}" width="56" style="opacity:0.9" />'
            if PL_SYMBOL_URI else "")
    st.markdown(f"""
    <div style="animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
         text-align:center;padding:2.2rem 1.5rem;margin:1rem 0 1.2rem;
         background:linear-gradient(160deg,#11162a 0%,#0d1322 100%);
         border:1px solid #1c2440;border-radius:18px">
        {logo}
        <div style="font-size:1.25rem;font-weight:900;color:#e8eaf0;margin-top:0.7rem">
            The {info['season_label']} season is complete</div>
        <div style="font-size:0.85rem;color:#8892a4;margin-top:0.35rem">
            Final matchday was {info['last_match'].strftime('%-d %B %Y')}. {blurb}</div>
        <div style="margin-top:1.3rem;font-size:0.78rem;letter-spacing:2.5px;
             color:#8892a4;font-weight:800">NEXT SEASON KICKS OFF IN</div>
        <div style="font-size:2.6rem;font-weight:900;
             background:linear-gradient(90deg,#3d6eff,#7c4dff);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            {info['days_to_kickoff']} days</div>
        <div style="font-size:0.78rem;color:#b8c0d0">~{info['kickoff'].strftime('%-d %B %Y')}</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        if st.button("🏅  View the Season Review", key=f"{key_prefix}_to_review",
                     use_container_width=True, type="primary"):
            st.session_state["_active_view"] = "review"
            st.rerun()
    return True


def tab_season_review(df):
    info = _offseason_info(df)

    # ── Season picker ─────────────────────────────────────────────────────
    # The live portfolios only ever hold the current season. Past seasons come
    # from data/seasons/<season>/, written by season_archive.archive_season().
    live = sa.live_season(default=info["season_label"])
    options = sa.season_options(live)
    season = options[0]
    if len(options) > 1:
        pick_c, _ = st.columns([1, 3])
        with pick_c:
            season = st.selectbox(
                "Season", options, index=0, key="review_season",
                format_func=lambda s: f"{s} · live" if s == live else s,
            )

    try:
        view = sa.load_season_view(season, live_season=live)
    except sa.ArchiveError as e:
        st.error(f"Could not load {season}: {e}")
        return

    main_p = view["main"]
    mt_p   = view["mock_two"]
    main_s = pf.portfolio_stats(main_p)
    mt_s   = pf.portfolio_stats(mt_p)
    main_clv = pf.clv_summary(main_p)
    mt_clv   = pf.clv_summary(mt_p)

    season_df = df[df["Season"] == season]
    tbl = get_current_table(season_df) if len(season_df) else season_df.iloc[0:0]
    champion = tbl.iloc[0]["Team"] if len(tbl) else "—"
    in_progress = view["is_live"] and len(season_df) < 380

    # ── Hero ──────────────────────────────────────────────────────────────
    if len(season_df):
        last_played = season_df["Date"].max().date()
        matchday_lbl = ("latest matchday" if in_progress else "final matchday")
        sub = (f"{len(season_df)} matches modelled · {matchday_lbl} "
               f"{last_played.strftime('%-d %B %Y')}")
    else:
        sub = f"No matches played yet · kicks off in {info['days_to_kickoff']} days"

    if in_progress:
        crest = ('<div style="font-size:1.05rem;font-weight:800;color:#00e5ff;'
                 'margin-top:0.45rem">Season in progress</div>')
    else:
        crest = (f'<div style="margin-top:1.1rem">{badge(champion, 64)}</div>'
                 f'<div style="font-size:1.05rem;font-weight:800;color:#ffd600;'
                 f'margin-top:0.45rem">🏆 {champion} — Premier League Champions</div>')

    st.markdown(f"""
    <div style="animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
         text-align:center;padding:2.2rem 1rem 1.9rem;margin-bottom:1.4rem;
         background:linear-gradient(160deg,#11162a 0%,#0d1322 100%);
         border:1px solid #1c2440;border-radius:18px">
        <div style="font-size:0.78rem;letter-spacing:3px;color:#8892a4;font-weight:800">SEASON REVIEW</div>
        <div style="font-size:2.4rem;font-weight:900;margin:0.15rem 0;
             background:linear-gradient(90deg,#3d6eff,#7c4dff,#ff4081);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">{season}</div>
        <div style="font-size:0.85rem;color:#8892a4">{sub}</div>
        {crest}
    </div>
    """, unsafe_allow_html=True)

    # ── Portfolio scorecards ──────────────────────────────────────────────
    st.markdown(
        '<p class="section-label">'
        f'{"How the portfolios are doing" if in_progress else "How the portfolios finished"}'
        '</p>', unsafe_allow_html=True)

    def _scorecard(title, accent, stats, clv, start):
        profit = stats["profit"]
        pcol = "#00e676" if profit >= 0 else "#ff4081"
        clv_v = (clv.get("median_clv") or 0) * 100
        clv_col = "#00e676" if clv_v >= 1.0 else ("#ffd600" if clv_v >= 0 else "#ff4081")
        chip = ('display:inline-block;padding:0.22rem 0.7rem;border-radius:999px;'
                'background:#0d1322;border:1px solid #1c2440;font-size:0.78rem;'
                'color:#8892a4;margin:0.15rem 0.2rem;font-weight:700')
        return f"""
        <div style="animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both;
             padding:1.4rem 1.5rem;background:#11162a;border:1px solid #1c2440;
             border-left:4px solid {accent};border-radius:16px;height:100%">
            <div style="font-size:0.78rem;letter-spacing:2px;font-weight:800;color:{accent}">{title}</div>
            <div style="font-size:2rem;font-weight:900;color:#e8eaf0;margin:0.25rem 0 0">£{stats['bankroll']:,.0f}</div>
            <div style="font-size:0.95rem;font-weight:800;color:{pcol}">
                {'+' if profit >= 0 else ''}£{profit:,.0f} from £{start:,.0f}</div>
            <div style="margin-top:0.8rem">
                <span style="{chip}">ROI <b style="color:#e8eaf0">{stats['roi']:+.1f}%</b></span>
                <span style="{chip}">Bets <b style="color:#e8eaf0">{stats['n_settled']}</b></span>
                <span style="{chip}">Win rate <b style="color:#e8eaf0">{stats['win_rate']:.0f}%</b></span>
                <span style="{chip}">Median CLV <b style="color:{clv_col}">{clv_v:+.2f}%</b></span>
            </div>
        </div>"""

    sc1, sc2 = st.columns(2, gap="medium")
    with sc1:
        st.markdown(_scorecard("MAIN PORTFOLIO", "#3d6eff", main_s, main_clv,
                               main_p["initial_bankroll"]), unsafe_allow_html=True)
    with sc2:
        st.markdown(_scorecard("MOCK TWO · RESEARCH TRACK", "#7c4dff", mt_s, mt_clv,
                               mt_p["initial_bankroll"]), unsafe_allow_html=True)

    # ── Bankroll journey ──────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:1.6rem">Bankroll journey</p>',
                unsafe_allow_html=True)

    def _bk_series(p):
        settled = sorted(
            [b for b in p["bets"] if b["status"] in ("won", "lost") and b.get("settled_at")],
            key=lambda b: b["settled_at"])
        if not settled:
            return [], []
        xs = [pd.to_datetime(settled[0].get("placed_at") or settled[0]["settled_at"])]
        ys = [p["initial_bankroll"]]
        running = p["initial_bankroll"]
        for b in settled:
            running += b.get("profit", 0.0)
            xs.append(pd.to_datetime(b["settled_at"]))
            ys.append(running)
        return xs, ys

    mx, my = _bk_series(main_p)
    tx, ty = _bk_series(mt_p)
    if mx or tx:
        fig = go.Figure()
        if mx:
            fig.add_trace(go.Scatter(x=mx, y=my, mode="lines+markers", name="Main",
                                     line=dict(color="#3d6eff", width=3, shape="hv"),
                                     marker=dict(size=5),
                                     hovertemplate="%{x|%d %b}: £%{y:,.0f}<extra>Main</extra>"))
        if tx:
            fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines+markers", name="Mock Two",
                                     line=dict(color="#7c4dff", width=3, shape="hv"),
                                     marker=dict(size=5),
                                     hovertemplate="%{x|%d %b}: £%{y:,.0f}<extra>Mock Two</extra>"))
        fig.add_hline(y=main_p["initial_bankroll"], line_dash="dot",
                      line_color="#556", annotation_text="start",
                      annotation_font_color="#556")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=12, color="#8892a4"),
            legend=dict(orientation="h", y=1.08, x=0),
            yaxis=dict(gridcolor="#1c2440", tickprefix="£", tickformat=",.0f"),
            xaxis=dict(gridcolor="#1c2440"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bets of the season ────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:1.2rem">Bets of the season</p>',
                unsafe_allow_html=True)
    all_settled = (
        [{**b, "_port": "Main", "_pcol": "#3d6eff"} for b in main_p["bets"]
         if b["status"] in ("won", "lost")]
        + [{**b, "_port": "Mock Two", "_pcol": "#7c4dff"} for b in mt_p["bets"]
           if b["status"] in ("won", "lost")]
    )
    if all_settled:
        top5 = sorted(all_settled, key=lambda b: b.get("profit", 0), reverse=True)[:5]
        worst = min(all_settled, key=lambda b: b.get("profit", 0))

        def _bet_row(b, rank=None):
            mkt_lbl, mkt_col = _MKT_CHIP.get(b["market"], (b["market"], "#8892a4"))
            profit = b.get("profit", 0.0)
            pcol = "#00e676" if profit >= 0 else "#ff4081"
            rank_html = (f'<span style="color:#b8c0d0;font-weight:900;width:1.4rem;'
                         f'display:inline-block">{rank}</span>' if rank else
                         '<span style="color:#ff4081;font-weight:900">✗</span> ')
            return f"""
            <div style="display:flex;align-items:center;gap:0.7rem;padding:0.55rem 0.9rem;
                 background:#11162a;border:1px solid #1c2440;border-radius:12px;
                 margin-bottom:0.4rem;flex-wrap:wrap;
                 animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both">
                {rank_html}
                <span style="font-size:0.84rem;color:#e8eaf0;font-weight:700;flex:1;min-width:220px">
                    {tb(b['home'], 24)} <span style="color:#b8c0d0">vs</span> {tb(b['away'], 24)}</span>
                <span style="font-size:0.78rem;font-weight:800;color:{mkt_col};
                     border:1px solid {mkt_col};border-radius:999px;padding:0.1rem 0.55rem">{mkt_lbl}</span>
                <span style="font-size:0.78rem;color:#8892a4">@{b['odds']:.2f} · £{b['stake']:,.0f}</span>
                <span style="font-size:0.78rem;color:{b['_pcol']};font-weight:700">{b['_port']}</span>
                <span style="font-size:0.9rem;font-weight:900;color:{pcol};min-width:80px;
                     text-align:right">{'+' if profit >= 0 else ''}£{profit:,.0f}</span>
            </div>"""

        st.markdown("".join(_bet_row(b, i + 1) for i, b in enumerate(top5)),
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#b8c0d0;margin:0.7rem 0 0.3rem;'
            'letter-spacing:1px;font-weight:800">AND THE ONE THAT HURT</p>',
            unsafe_allow_html=True)
        st.markdown(_bet_row(worst), unsafe_allow_html=True)
    else:
        st.info("No settled bets this season.")

    # ── Final table ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:1.6rem">Final league table</p>',
                unsafe_allow_html=True)
    if len(tbl):
        rows_html = []
        for i, r in tbl.iterrows():
            pos = i + 1
            accent = ("#ffd600" if pos == 1 else
                      "#00e676" if pos <= 4 else
                      "#00e5ff" if pos <= 6 else
                      "#ff4081" if pos >= len(tbl) - 2 else "transparent")
            rows_html.append(f"""
            <tr style="border-left:3px solid {accent}">
                <td style="color:#b8c0d0;font-weight:800;padding:0.35rem 0.6rem">{pos}</td>
                <td style="padding:0.35rem 0.6rem;color:#e8eaf0;font-weight:700;text-align:left">{tb(r['Team'], 22)}</td>
                <td>{r['Played']}</td><td>{r['W']}</td><td>{r['D']}</td><td>{r['L']}</td>
                <td>{r['GF']}</td><td>{r['GA']}</td>
                <td style="color:{'#00e676' if r['GD'] >= 0 else '#ff4081'}">{r['GD']:+d}</td>
                <td style="color:#e8eaf0;font-weight:900">{r['Pts']}</td>
            </tr>""")
        st.markdown(f"""
        <div style="background:#11162a;border:1px solid #1c2440;border-radius:14px;
             padding:0.7rem 0.6rem;overflow-x:auto;
             animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;
               color:#8892a4;text-align:center">
            <thead><tr style="color:#a78bfa;font-size:0.78rem;letter-spacing:1.2px">
                <th style="padding:0.3rem 0.6rem">#</th>
                <th style="text-align:left;padding:0.3rem 0.6rem">TEAM</th>
                <th>P</th><th>W</th><th>D</th><th>L</th>
                <th>GF</th><th>GA</th><th>GD</th><th>PTS</th>
            </tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table></div>
        <p style="font-size:0.78rem;color:#b8c0d0;margin-top:0.4rem">
            <span style="color:#ffd600">▌</span> Champions &nbsp;
            <span style="color:#00e676">▌</span> Champions League &nbsp;
            <span style="color:#00e5ff">▌</span> Europe &nbsp;
            <span style="color:#ff4081">▌</span> Relegated</p>
        """, unsafe_allow_html=True)

    # ── What the season taught the model ──────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:1.6rem">What this season taught the model</p>',
                unsafe_allow_html=True)
    lessons = [
        ("🎯", "The edge is EPL-specific", "#3d6eff",
         "The same gates lose money on La Liga, Bundesliga and Serie A — "
         "bookmakers there price draws more accurately. Validated on 9,871 "
         "matches across 5 leagues. No expansion."),
        ("🧪", "Honest calibration is non-negotiable", "#ff4081",
         "A calibration-leakage bug inflated one backtest from a £3.5k loss "
         "to +£97k. Every result now uses calibrators fitted strictly before "
         "the evaluation window."),
        ("✂️", "Simple gates beat complex ones", "#00e676",
         "ELO floors, EV caps and day-of-week bans looked great in leaky "
         "backtests and mostly vanished under honest evaluation. The deployed "
         "config is a probability gate, an EV gate and Kelly."),
        ("📉", "Volume and ROI trade off structurally", "#ffd600",
         "Under honest calibration this model finds 3–15 genuinely mispriced "
         "draws per season — not 30+. Forcing more volume destroys the ROI."),
        ("📅", "Per-season variance is huge", "#7c4dff",
         "2023-24 was a bad year for nearly every config; 2025-26 suited the "
         "model unusually well. One season proves little — the multi-season "
         "grid is the referee."),
    ]
    lc1, lc2 = st.columns(2, gap="medium")
    for i, (icon, title, accent, body) in enumerate(lessons):
        with (lc1 if i % 2 == 0 else lc2):
            st.markdown(f"""
            <div style="padding:1rem 1.2rem;background:#11162a;border:1px solid #1c2440;
                 border-left:4px solid {accent};border-radius:14px;margin-bottom:0.8rem;
                 animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both">
                <div style="font-size:0.9rem;font-weight:800;color:#e8eaf0">{icon} {title}</div>
                <div style="font-size:0.82rem;color:#8892a4;margin-top:0.3rem;line-height:1.5">{body}</div>
            </div>""", unsafe_allow_html=True)

    # ── Next season ───────────────────────────────────────────────────────
    # Only meaningful while the live season is still to start. Once it is under
    # way this block would count down to a season already in progress.
    if len(df[df["Season"] == live]) > 0:
        return
    st.markdown(f"""
    <div style="text-align:center;padding:1.6rem 1rem;margin-top:1rem;
         background:linear-gradient(160deg,#11162a 0%,#0d1322 100%);
         border:1px solid #1c2440;border-radius:16px;
         animation:fadeInUp 0.5s cubic-bezier(.22,.61,.36,1) both">
        <div style="font-size:0.78rem;letter-spacing:2.5px;color:#8892a4;font-weight:800">
            {live} KICKS OFF IN</div>
        <div style="font-size:2.4rem;font-weight:900;
             background:linear-gradient(90deg,#3d6eff,#7c4dff);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            {info['days_to_kickoff']} days</div>
        <div style="font-size:0.82rem;color:#8892a4;margin-top:0.5rem">
            Pre-season checklist: re-run the honest random search on updated data ·
            re-validate the edge cross-league · review the simplified config ·
            let the odds-snapshot pipeline warm up</div>
    </div>
    """, unsafe_allow_html=True)


TAB_REGISTRY = [
    ("predict",     "🔮", "Predict Match",  "Pick any two teams"),
    ("weekend",     "📅", "This Weekend",   "Upcoming fixtures"),
    ("results",     "📋", "Last Gameweek",  "How the model did"),
    ("backtest",    "📊", "Backtesting",    "Accuracy & calibration"),
    ("season",      "🏆", "Season Outlook", "Monte Carlo forecast"),
    ("review",      "🏅", "Season Review",  "Wrap-up & lessons"),
    ("elo",         "📈", "Team Elo",       "Live ratings + form"),
    ("teamdeep",    "🔍", "Team Deep Dive", "All data on one team"),
    ("portfolio",   "💰", "Mock Portfolio", "Paper-trade bets"),
    ("portfolio2",  "🧪", "Mock Two",       "Research-track A/B"),
    ("preflight",   "🛫", "Pre-Flight",     "Season readiness"),
]


def _read_activity_log(max_events: int = 80) -> list[dict]:
    """Read the last N events from data/activity.log. Returns newest first.
    Empty list if the log doesn't exist or all lines are malformed.
    """
    log_path = Path("data/activity.log")
    if not log_path.exists():
        return []
    try:
        # Read last few KB only — enough for hundreds of events without
        # loading hours of history each render.
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in tail.split("\n") if ln.strip()]
        events = []
        for ln in lines:
            try:
                events.append(json.loads(ln))
            except Exception:
                continue
        return list(reversed(events))[:max_events]
    except Exception:
        return []


def _humanize_age(ts_str: str) -> str:
    """ISO timestamp → '2h ago' / '4d ago' / etc."""
    try:
        # Strip timezone for naive comparison
        ts = ts_str.rstrip("Z").split("+")[0]
        t = datetime.fromisoformat(ts)
        now = datetime.utcnow()
        delta = now - t
        secs = int(delta.total_seconds())
        if secs < 60:        return "just now"
        if secs < 3600:      return f"{secs // 60}m ago"
        if secs < 86400:     return f"{secs // 3600}h ago"
        if secs < 86400 * 7: return f"{secs // 86400}d ago"
        return t.strftime("%d %b")
    except Exception:
        return ""


def _render_home_activity() -> None:
    """4-tile snapshot row + recent-events feed above the home tile grid.
    Reads from both portfolios + data/activity.log."""
    # Pull state from both portfolios + activity log
    try:
        main_p = pf.load_portfolio()
        mt_p   = pf.load_portfolio_two()
    except Exception:
        return

    main_stats = pf.portfolio_stats(main_p)
    mt_stats   = pf.portfolio_stats(mt_p)
    main_pending = [b for b in main_p["bets"] if b["status"] == "pending"]
    mt_pending   = [b for b in mt_p["bets"]   if b["status"] == "pending"]
    main_clv = pf.clv_summary(main_p)
    mt_clv   = pf.clv_summary(mt_p)
    events   = _read_activity_log(60)

    # Last auto-bet run timestamp (most recent run_completed or run_started)
    last_run = next((e for e in events if e["type"] in ("run_completed","run_started")), None)
    last_run_age = _humanize_age(last_run["ts"]) if last_run else "never"

    # Auto-bet count in the last 24h
    cutoff = (datetime.utcnow() - timedelta(hours=24))
    auto_24h = sum(
        1 for e in events
        if e.get("type") == "auto_bet_placed"
        and datetime.fromisoformat(e["ts"].rstrip("Z").split("+")[0]) > cutoff
    )
    settled_24h = sum(
        1 for e in events
        if e.get("type") == "settled"
        and datetime.fromisoformat(e["ts"].rstrip("Z").split("+")[0]) > cutoff
    )

    # Next fixture lookup
    next_fix = None
    try:
        for fx in fetch_upcoming_fixtures(lookahead_days=7):
            if not next_fix or fx["date"] < next_fix["date"]:
                next_fix = fx
    except Exception:
        pass

    # ── 4 quick-snapshot tiles ────────────────────────────────────────
    main_pl   = main_stats["profit"]
    mt_pl     = mt_stats["profit"]
    main_pl_col = "#00e676" if main_pl >= 0 else "#ff4081"
    mt_pl_col   = "#00e676" if mt_pl   >= 0 else "#ff4081"

    main_clv_v = (main_clv.get("median_clv") or 0) * 100
    mt_clv_v   = (mt_clv.get("median_clv")   or 0) * 100
    main_clv_col = "#00e676" if main_clv_v >= 1.0 else ("#ffd600" if main_clv_v >= 0 else "#ff4081")
    mt_clv_col   = "#00e676" if mt_clv_v   >= 1.0 else ("#ffd600" if mt_clv_v   >= 0 else "#ff4081")

    # Next fixture countdown
    next_fix_html = ""
    if next_fix:
        try:
            from datetime import date as _date
            today = _date.today()
            days_to = (next_fix["date"] - today).days
            next_str = f"{next_fix['home']} vs {next_fix['away']}"
            if days_to == 0:    when = "TODAY"
            elif days_to == 1:  when = "TOMORROW"
            else:               when = f"in {days_to}d"
            next_fix_html = f'<div class="hf-next-when">{when}</div><div class="hf-next-match">{next_str}</div>'
        except Exception:
            next_fix_html = '<div class="hf-next-when">—</div>'
    else:
        next_fix_html = '<div class="hf-next-when">—</div><div class="hf-next-match">No upcoming fixtures</div>'

    main_pl_sign = "+" if main_pl >= 0 else "−"
    mt_pl_sign   = "+" if mt_pl   >= 0 else "−"

    st.markdown(
        '<div class="home-feed-grid">'

        # ── Tile 1: Auto-bet status ──
        '<div class="hf-tile hf-auto">'
        f'<div class="hf-tile-icon">🤖</div>'
        f'<div class="hf-tile-lbl">AUTO-BET</div>'
        f'<div class="hf-tile-val">{auto_24h}</div>'
        f'<div class="hf-tile-sub">placed in last 24h<br>'
        f'<span style="color:#a78bfa">last run {last_run_age}</span></div>'
        '</div>'

        # ── Tile 2: Portfolios at a glance ──
        '<div class="hf-tile hf-portfolios">'
        f'<div class="hf-tile-icon">💰</div>'
        f'<div class="hf-tile-lbl">PORTFOLIOS</div>'
        f'<div class="hf-portfolios-row">'
            f'<span class="hf-port-line"><b>Main</b> '
            f'<span style="color:{main_pl_col}">{main_pl_sign}£{abs(main_pl):,.0f}</span></span>'
            f'<span class="hf-port-line"><b>Mock 2</b> '
            f'<span style="color:{mt_pl_col}">{mt_pl_sign}£{abs(mt_pl):,.0f}</span></span>'
        f'</div>'
        f'<div class="hf-tile-sub">'
        f'{len(main_pending)} + {len(mt_pending)} pending · {settled_24h} settled 24h</div>'
        '</div>'

        # ── Tile 3: CLV health ──
        '<div class="hf-tile hf-clv">'
        f'<div class="hf-tile-icon">📐</div>'
        f'<div class="hf-tile-lbl">EDGE HEALTH (CLV)</div>'
        f'<div class="hf-portfolios-row">'
            f'<span class="hf-port-line"><b>Main</b> '
            f'<span style="color:{main_clv_col}">{main_clv_v:+.2f}%</span></span>'
            f'<span class="hf-port-line"><b>Mock 2</b> '
            f'<span style="color:{mt_clv_col}">{mt_clv_v:+.2f}%</span></span>'
        f'</div>'
        f'<div class="hf-tile-sub">'
        f'{main_clv.get("n", 0)} + {mt_clv.get("n", 0)} tagged bets · '
        f'{"sharp" if max(main_clv_v, mt_clv_v) >= 1 else "monitor"}</div>'
        '</div>'

        # ── Tile 4: Next fixture ──
        '<div class="hf-tile hf-next">'
        f'<div class="hf-tile-icon">📅</div>'
        f'<div class="hf-tile-lbl">NEXT FIXTURE</div>'
        f'{next_fix_html}'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )

    # ── Recent events feed ─────────────────────────────────────────────
    feed_events = [e for e in events
                   if e.get("type") in ("auto_bet_placed", "settled",
                                         "clv_snapshot", "warning", "error", "fatal")][:10]
    if not feed_events:
        return  # nothing to show yet

    item_html = []
    for e in feed_events:
        et = e.get("type")
        age = _humanize_age(e.get("ts", ""))
        port = e.get("portfolio", "").upper()
        port_chip = f'<span class="hf-evt-port">{port}</span>' if port else ''
        if et == "auto_bet_placed":
            txt = (f'<b>🤖 Auto-bet</b> {e.get("selection","?")} @ '
                   f'<b style="color:#00e5ff">{e.get("odds","?"):.2f}</b> · '
                   f'£{e.get("stake",0):,.0f} stake · '
                   f'<span style="color:#00e676">{e.get("ev",0)*100:+.1f}% EV</span> '
                   f'<span class="hf-evt-match">{e.get("match","")}</span>')
            row_cls = "hf-evt-place"
        elif et == "settled":
            res = e.get("result", "?")
            profit = e.get("profit") or 0
            sign = "+" if profit >= 0 else "−"
            colr = "#00e676" if res == "won" else "#ff4081"
            ico  = "✅" if res == "won" else "❌"
            txt = (f'<b>{ico} Settled</b> · '
                   f'<span class="hf-evt-match">{e.get("match","")}</span> '
                   f'· <span style="color:{colr};font-weight:800">'
                   f'{sign}£{abs(profit):,.2f}</span>')
            row_cls = f"hf-evt-{'won' if res=='won' else 'lost'}"
        elif et == "clv_snapshot":
            v = (e.get("median_clv") or 0) * 100
            colr = "#00e676" if v >= 0 else "#ff4081"
            txt = (f'<b>📐 CLV check</b> · median '
                   f'<span style="color:{colr};font-weight:800">{v:+.2f}%</span> '
                   f'over {e.get("n",0)} tagged')
            row_cls = "hf-evt-clv"
        elif et in ("warning", "error", "fatal"):
            txt = f'<b>⚠️ {et.title()}</b> · {e.get("message", e.get("error",""))}'
            row_cls = "hf-evt-warn"
        else:
            continue
        item_html.append(
            f'<div class="hf-evt {row_cls}">'
              f'<div class="hf-evt-time">{age}</div>'
              f'<div class="hf-evt-body">{txt}</div>'
              f'<div class="hf-evt-port-cell">{port_chip}</div>'
            f'</div>'
        )
    st.markdown(
        '<div class="hf-feed-title">📡 RECENT ACTIVITY</div>'
        '<div class="hf-feed-list">' + "".join(item_html) + '</div>',
        unsafe_allow_html=True,
    )


def _render_home_screen() -> None:
    """Large PL logo + title + 6-tile grid for tab entry."""
    # Hero: big logo + title
    logo_html = (f'<img src="{PL_SYMBOL_URI}" alt="Premier League" />'
                 if PL_SYMBOL_URI else '<div style="font-size:9rem">⚽</div>')
    st.markdown(f"""
    <div class="home-hero">
        {logo_html}
        <div class="home-title">Premier League Match Predictor</div>
        <div class="home-sub">Dixon-Coles · Poisson · XGBoost · Draw Specialist</div>
    </div>
    """, unsafe_allow_html=True)

    # Recent activity feed — appears between hero and tile grid
    _render_home_activity()

    st.markdown("<br>", unsafe_allow_html=True)

    # Tile grid — 3 columns, as many rows as needed. Scope tile-look CSS to the container.
    st.markdown('<div class="home-tile-mode">', unsafe_allow_html=True)

    n_rows = (len(TAB_REGISTRY) + 2) // 3   # ceil division
    cols = [st.columns(3, gap="medium") for _ in range(n_rows)]
    for idx, (key, emoji, name, blurb) in enumerate(TAB_REGISTRY):
        col = cols[idx // 3][idx % 3]
        with col:
            if st.button(f"{emoji}\n\n**{name}**\n\n{blurb}",
                         key=f"home_tile_{key}",
                         use_container_width=True):
                st.session_state["_active_view"] = key
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Minimal footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;font-size:0.78rem;color:#9aa6ba;letter-spacing:2px">'
        'DATA: FOOTBALL-DATA.CO.UK + UNDERSTAT · FOR ENTERTAINMENT PURPOSES</p>',
        unsafe_allow_html=True,
    )



# ── Pre-Flight: is the season actually ready to trade? ─────────────────────────

def _season_of(day) -> str:
    """The season a date belongs to. August onward starts a new one.

    Needed because the loaded CSVs still end in 2025-26 — football-data does not
    publish a season's file until it starts — so the data's own max season is a
    year behind the one being counted down to.
    """
    return (f"{day.year}-{str(day.year + 1)[-2:]}" if day.month >= 8
            else f"{day.year - 1}-{str(day.year)[-2:]}")


def _preflight_runner_state() -> dict:
    """Last headless-runner activity, and whether the launchd job is loaded.

    The runner places bets when nobody has the app open, so "did it run" is a
    question the app should answer rather than one you check by tailing a log.
    """
    events = _read_activity_log(max_events=200)
    last_run = next((e for e in events if e.get("type") == "run_completed"), None)
    skips = [e for e in events if e.get("type") == "fixture_skipped"]
    loaded = False
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=5).stdout
        loaded = "com.eoinhoustoun.fpred" in out
    except Exception:
        loaded = False
    return {"last_run": last_run, "skips": skips[:6], "job_loaded": loaded}


def _preflight_chip(text: str, tone: str) -> str:
    """A status pill. Tones map to the palette: go / hold / stop."""
    colours = {
        "go":   ("#00e676", "rgba(0,230,118,0.14)"),
        "hold": ("#ffd600", "rgba(255,214,0,0.14)"),
        "stop": ("#ff4081", "rgba(255,64,129,0.14)"),
        "info": ("#00e5ff", "rgba(0,229,255,0.14)"),
    }
    fg, bg = colours.get(tone, colours["info"])
    return (f'<span style="display:inline-block;padding:0.2rem 0.65rem;'
            f'border-radius:999px;background:{bg};color:{fg};'
            f'font-size:0.78rem;font-weight:800;letter-spacing:0.6px;'
            f'white-space:nowrap">{text}</span>')


def _preflight_portfolio_card(port: dict, title: str, accent: str) -> str:
    """One portfolio's trading posture, as a single-line HTML card.

    Built as one line on purpose: st.markdown stops passing raw HTML through
    once any line is whitespace-only, which happens as soon as an interpolated
    value comes back empty.
    """
    s = port["settings"]
    live = bool(s.get("auto_bet_enabled", False))
    posture = (_preflight_chip("AUTO-BET LIVE", "go") if live
               else _preflight_chip("AUTO-BET OFF", "hold"))
    gates = [
        ("EV gate",     f"{int(float(s.get('min_ev', 0.40)) * 100)}%"),
        ("Min prob",    f"{int(float(s.get('min_prob', 0.21)) * 100)}%"),
        ("Kelly",       f"{float(s.get('kelly_fraction', 1.0)):g}x"),
        ("Max stake",   f"{int(float(s.get('max_stake_pct', 0.25)) * 100)}%"),
        ("Elo floor",   str(s.get("main_min_team_elo") or s.get("v2_min_team_elo") or "off")),
        ("History gate", f"{s['min_team_matches']} matches"
                         if s.get("min_team_matches") else "off"),
        ("Club cap",    f"{s['max_bets_per_club']} bets"
                        if s.get("max_bets_per_club") else "off"),
        ("Markets",     "+".join(s.get("auto_markets", [])) or "none"),
    ]
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:0.6rem;'
        f'padding:0.28rem 0;border-bottom:1px solid rgba(255,255,255,0.05)">'
        f'<span style="font-size:0.82rem;color:#b8c0d0">{k}</span>'
        f'<span style="font-size:0.82rem;color:#e8eaf0;font-weight:700;'
        f'font-variant-numeric:tabular-nums">{v}</span></div>'
        for k, v in gates)
    label = s.get("main_settings_label") or s.get("v2_settings_label") or ""
    html = f"""
    <div class="metric-card" style="text-align:left;border-color:{accent}55">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  gap:0.5rem;margin-bottom:0.6rem">
        <span style="font-size:0.95rem;font-weight:900;color:{accent};
                     letter-spacing:0.5px">{title}</span>{posture}
      </div>
      <div style="font-size:1.5rem;font-weight:900;color:#e8eaf0;
                  font-variant-numeric:tabular-nums">£{port['bankroll']:,.2f}</div>
      <div style="font-size:0.78rem;color:#b8c0d0;margin-bottom:0.7rem">
        {len(port['bets'])} bets this season · opened £{port['initial_bankroll']:,.0f}</div>
      {rows}
      <div style="font-size:0.78rem;color:#b8c0d0;margin-top:0.7rem;
                  line-height:1.45">{label}</div>
    </div>
    """
    return "".join(seg.strip() for seg in html.splitlines())


def tab_preflight(df, dc_r, dc_draw_r, xgb_m, feat_cols,
                  draw_xgb_m, draw_fc, teams, elo_dict):
    """Season readiness in one screen: countdown, posture, and what is gated.

    Everything here was previously only visible by reading data/activity.log or
    the portfolio JSON. A gate that silently drops two of ten opening fixtures
    is indistinguishable from a gate that found nothing, which is exactly how
    the promoted-team problem stayed invisible.
    """
    info = _offseason_info(df)
    main_port = pf.load_portfolio()
    mt_port   = pf.load_portfolio_two()
    match_counts = team_match_counts(df)

    # ── Countdown hero ───────────────────────────────────────────────────
    days = info["days_to_kickoff"]
    tone = "#00e676" if days == 0 else ("#ffd600" if days <= 14 else "#7c4dff")
    st.markdown(
        f'<div class="pnl-hero pnl-neutral" style="border-color:{tone}55">'
        f'<div class="pnl-tag" style="color:#b8c0d0">Season {_season_of(info["kickoff"])} '
        f'· first bet-eligible fixture</div>'
        f'<div class="pnl-amount" style="color:{tone}">{days}</div>'
        f'<div style="font-size:0.95rem;color:#e8eaf0;font-weight:700">'
        f'days to kickoff · {info["kickoff"].strftime("%A %-d %B %Y")}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Portfolio posture ────────────────────────────────────────────────
    _section_header("Trading posture",
                    "What each line would do if a qualifying fixture appeared "
                    "right now.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_preflight_portfolio_card(main_port, "MAIN", "#3d6eff"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(_preflight_portfolio_card(mt_port, "MOCK TWO", "#7c4dff"),
                    unsafe_allow_html=True)

    # ── Fixture board ────────────────────────────────────────────────────
    _section_header("Opening fixtures",
                    "Model draw probability, and whether the fixture is "
                    "eligible to be staked.")
    try:
        fixtures = cached_fixtures()[:12]
    except Exception as e:
        fixtures = []
        st.warning(f"Could not load fixtures: {e}")

    if not fixtures:
        st.info("No upcoming fixtures published yet.")
    else:
        # main() already seeds promoted sides before anything predicts, so
        # re-seeding here would find nothing new and report an empty list.
        # Trust the marker it leaves, and only seed if we were handed raw
        # ratings (which happens if the seeding step upstream failed).
        if dc_r.get("seeded_teams") is None:
            names = [t for f in fixtures for t in (f["home"], f["away"])]
            seeded_dc = seed_promoted_teams(dc_r, names)
            seeded_draw = seed_promoted_teams(dc_draw_r, names)
        else:
            seeded_dc, seeded_draw = dc_r, dc_draw_r
        min_matches = main_port["settings"].get("min_team_matches")
        rows_html = []
        for f in fixtures:
            h, a = f["home"], f["away"]
            # Full ensemble, not bare Dixon-Coles: this must be the same number
            # the auto-bet path stakes on, or the board is quietly lying.
            try:
                hs = get_current_stats(df, h, elo_dict=elo_dict)
                as_ = get_current_stats(df, a, elo_dict=elo_dict)
                _dc, _blend, res = full_predict(
                    h, a, seeded_dc, seeded_draw, xgb_m, feat_cols,
                    draw_xgb_m, draw_fc, hs, as_)
                p = {"draw": res["draw"]}
            except Exception:
                p = predict_dixon_coles(h, a, seeded_dc)
            blocked = pf.should_skip_unrated(h, a, match_counts, min_matches)
            if blocked:
                low = [t for t in (h, a) if match_counts.get(t, 0) < (min_matches or 0)]
                badge = _preflight_chip("GATED", "stop")
                why = (f"{', '.join(low)} under {min_matches} Premier League "
                       f"matches — priced on the promoted prior, not staked")
            else:
                badge = _preflight_chip("ELIGIBLE", "go")
                why = "both sides rated"
            draw_pct = p["draw"] * 100
            rows_html.append(
                f'<div style="display:grid;grid-template-columns:5.5rem 1fr auto auto;'
                f'gap:0.7rem;align-items:center;padding:0.5rem 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.06)">'
                f'<span style="font-size:0.78rem;color:#b8c0d0">'
                f'{f["date"].strftime("%a %-d %b")}</span>'
                f'<span><span style="font-size:0.88rem;color:#e8eaf0;font-weight:700">'
                f'{h} v {a}</span><br>'
                f'<span style="font-size:0.78rem;color:#b8c0d0">'
                f'Elo {elo_dict.get(h, 0):.0f} v {elo_dict.get(a, 0):.0f} · '
                f'{why}</span></span>'
                f'<span style="font-size:0.95rem;font-weight:900;color:#ffd600;'
                f'font-variant-numeric:tabular-nums">{draw_pct:.1f}%</span>{badge}</div>')
        st.markdown(
            '<div class="metric-card" style="text-align:left">'
            '<div style="display:grid;grid-template-columns:5.5rem 1fr auto auto;'
            'gap:0.7rem;font-size:0.78rem;color:#b8c0d0;font-weight:800;'
            'letter-spacing:1px;text-transform:uppercase;padding-bottom:0.4rem">'
            '<span>Date</span><span>Fixture</span><span>Draw</span><span>Status</span>'
            '</div>' + "".join(rows_html) + '</div>',
            unsafe_allow_html=True)

        seeded = seeded_dc.get("seeded_teams", [])
        if seeded:
            priors = promoted_prior_for(seeded)
            rows = "".join(
                f'<div style="display:grid;'
                f'grid-template-columns:9rem 5.5rem 1fr 1fr 5rem;gap:0.6rem;'
                f'align-items:center;padding:0.35rem 0;'
                f'border-bottom:1px solid rgba(255,214,0,0.15)">'
                f'<span style="font-size:0.88rem;color:#e8eaf0;font-weight:800">'
                f'{team}</span>'
                f'<span style="font-size:0.86rem;color:#ffd600;font-weight:800;'
                f'font-variant-numeric:tabular-nums">'
                f'{priors[team].get("market_relegation_prob", 0):.0%}</span>'
                f'<span style="font-size:0.86rem;color:#e8eaf0;'
                f'font-variant-numeric:tabular-nums">'
                f'{priors[team]["attack"]:+.3f}</span>'
                f'<span style="font-size:0.86rem;color:#e8eaf0;'
                f'font-variant-numeric:tabular-nums">'
                f'{priors[team]["defense"]:+.3f}</span>'
                f'<span style="font-size:0.95rem;color:#00e5ff;font-weight:900;'
                f'font-variant-numeric:tabular-nums">{elo_dict.get(team, 0):.0f}'
                f'</span></div>'
                for team in sorted(seeded,
                                   key=lambda t: elo_dict.get(t, 0), reverse=True))
            # Nothing refreshes the relegation table, and when it ages out it
            # fails silently: next season's promoted clubs are absent from it,
            # all of them revert to the flat prior, and nothing errors.
            stale_note = ""
            if market_odds_are_stale():
                stale_note = (
                    f'<div style="margin-top:0.6rem;padding:0.5rem 0.7rem;'
                    f'border-radius:8px;background:rgba(255,64,129,0.12);'
                    f'border:1px solid rgba(255,64,129,0.45);font-size:0.82rem;'
                    f'color:#ff8fb1;font-weight:700">Relegation odds are '
                    f'{market_odds_age_days()} days old (captured '
                    f'{MARKET_ODDS_CAPTURED}). Refresh MARKET_RELEGATION_ODDS in '
                    f'models.py — until then promoted sides fall back to the flat '
                    f'prior and cannot be told apart.</div>')
            rated = sorted(((t, v) for t, v in elo_dict.items()
                            if t not in seeded and t in teams),
                           key=lambda kv: kv[1])[:2]
            scale = " · ".join(f"{t} {v:.0f}" for t, v in rated)
            st.markdown(
                f'<div style="margin-top:0.9rem;padding:0.9rem 1rem;'
                f'border-radius:12px;background:rgba(255,214,0,0.08);'
                f'border:1px solid rgba(255,214,0,0.28)">'
                f'<div style="font-size:0.86rem;color:#ffd600;font-weight:800;'
                f'margin-bottom:0.5rem">No Premier League history — rated by the '
                f'relegation market</div>'
                f'<div style="display:grid;'
                f'grid-template-columns:9rem 5.5rem 1fr 1fr 5rem;gap:0.6rem;'
                f'font-size:0.78rem;color:#b8c0d0;font-weight:800;'
                f'letter-spacing:1px;text-transform:uppercase;'
                f'padding-bottom:0.3rem">'
                f'<span>Team</span><span>Relegation</span><span>Attack</span>'
                f'<span>Defence</span><span>Elo</span></div>'
                f'{rows}'
                f'<div style="display:grid;'
                f'grid-template-columns:9rem 5.5rem 1fr 1fr 5rem;gap:0.6rem;'
                f'padding-top:0.35rem">'
                f'<span style="font-size:0.82rem;color:#b8c0d0">pooled prior</span>'
                f'<span style="font-size:0.82rem;color:#b8c0d0;'
                f'font-variant-numeric:tabular-nums">60%</span>'
                f'<span style="font-size:0.82rem;color:#b8c0d0;'
                f'font-variant-numeric:tabular-nums">'
                f'{PROMOTED_PRIOR["attack"]:+.3f}</span>'
                f'<span style="font-size:0.82rem;color:#b8c0d0;'
                f'font-variant-numeric:tabular-nums">'
                f'{PROMOTED_PRIOR["defense"]:+.3f}</span>'
                f'<span style="font-size:0.82rem;color:#b8c0d0;'
                f'font-variant-numeric:tabular-nums">1394</span></div>'
                f'<div style="font-size:0.82rem;color:#e8eaf0;line-height:1.5;'
                f'margin-top:0.6rem">The fitted prior is flat: no Championship '
                f'signal beat the pooled mean out of sample, so it cannot tell '
                f'these two apart. The relegation book can — 9 of the last 15 '
                f'promoted sides went straight back down, so 60% is average, and '
                f'each team is moved off the pooled value by half the observed '
                f'spread. Elo is anchored on where promoted teams actually finish '
                f'their first season, not the 1500 default — measured live from the '
                f'weakest clubs currently in the league, so it cannot go stale. '
                f'Lowest rated sides with real history: {scale}.</div>'
                f'{stale_note}</div>',
                unsafe_allow_html=True)

    # ── Runner ───────────────────────────────────────────────────────────
    _section_header("Headless runner",
                    "scripts/run_auto_bet.py places bets when the app is "
                    "closed. Same gates as the app.")
    state = _preflight_runner_state()
    job = (_preflight_chip("SCHEDULED", "go") if state["job_loaded"]
           else _preflight_chip("NOT SCHEDULED", "hold"))
    last = state["last_run"]
    when = _humanize_age(last["ts"]) if last else "never"
    st.markdown(
        f'<div class="metric-card" style="text-align:left">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:0.88rem;color:#e8eaf0;font-weight:700">'
        f'launchd job</span>{job}</div>'
        f'<div style="font-size:0.82rem;color:#b8c0d0;margin-top:0.4rem">'
        f'Last completed run: {when}</div></div>',
        unsafe_allow_html=True)

    if state["skips"]:
        with st.expander(f"Fixtures the runner passed over ({len(state['skips'])})"):
            for e in state["skips"]:
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#e8eaf0;padding:0.2rem 0">'
                    f'<b>{e.get("match", "?")}</b> — {e.get("reason", "?")}: '
                    f'{e.get("detail", "")}</div>', unsafe_allow_html=True)

def _render_top_bar(active_key: str) -> None:
    """Compact header for non-home views: small logo + tab title + Home button."""
    meta = next(((e, n, b) for k, e, n, b in TAB_REGISTRY if k == active_key), ("⚽", "—", ""))
    emoji, name, blurb = meta

    logo_col, title_col, btn_col = st.columns([1, 8, 2])
    with logo_col:
        if PL_SYMBOL_URI:
            st.markdown(
                f'<div class="top-bar"><img src="{PL_SYMBOL_URI}" alt="PL" /></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="top-bar"><span style="font-size:2rem">{emoji}</span></div>',
                        unsafe_allow_html=True)
    with title_col:
        st.markdown(f"""
        <div class="top-bar">
            <div>
                <div class="top-bar-title">{emoji}  {name}</div>
                <div class="top-bar-sub">{blurb}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with btn_col:
        st.markdown('<div style="padding-top:0.45rem">', unsafe_allow_html=True)
        if st.button("← Home", key="home_back", use_container_width=True):
            st.session_state["_active_view"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin:0.5rem 0 1rem"></div>',
                unsafe_allow_html=True)


def _log_activity_event(event_type: str, **kwargs) -> None:
    """Append a JSON-Lines event to data/activity.log (same format as
    scripts/run_auto_bet.py so the home-screen activity feed reads them
    interchangeably)."""
    log_path = Path("data/activity.log")
    log_path.parent.mkdir(exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "type": event_type, **kwargs}
    try:
        with log_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass  # Activity log is best-effort; never crash the app


def _session_auto_bet(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                      draw_xgb_m, draw_fc, teams, elo_dict) -> None:
    """Run the auto-bet pipeline once per Streamlit session.

    Mirrors `scripts/run_auto_bet.py` but runs in-process so opening the app
    or refreshing the page triggers settlements + auto-places. Uses session
    state to ensure it only fires once per browser session (not on every
    rerun from button clicks).

    Skips silently when the API key is missing or both portfolios have
    auto-bet disabled. Toasts the user when bets are actually placed.
    """
    # Re-fire only if last run was > 5 minutes ago in this session.
    # Streamlit reruns on every button/widget change; we don't want to
    # refetch live odds on each click.
    last_ts = st.session_state.get("_session_auto_bet_ts")
    now = datetime.now(timezone.utc)
    if last_ts and (now - last_ts).total_seconds() < 300:
        return
    st.session_state["_session_auto_bet_ts"] = now

    # Load both portfolios fresh — these reads are cheap (small JSONs)
    main_port = pf.load_portfolio()
    mt_port   = pf.load_portfolio_two()

    # Auto-settle both BEFORE we look at auto-betting (so closed bets don't
    # block fresh ones via exposure caps)
    n_settled_main = pf.auto_settle(main_port, df)
    n_settled_mt   = pf.auto_settle(mt_port, df)
    pf.backfill_clv_for_settled_bets(main_port, df)
    pf.backfill_clv_for_settled_bets(mt_port, df)

    api_key      = pf.resolve_odds_api_key(main_port.get("settings", {}).get("odds_api_key", ""))
    main_enabled = bool(main_port["settings"].get("auto_bet_enabled", False))
    mt_enabled   = bool(mt_port["settings"].get("auto_bet_enabled", False))

    if not api_key:
        _log_activity_event("warning",
                            message="No Odds API key — auto-bet skipped (app load)")
        # Still save settlements
        pf.save_portfolio(main_port)
        pf.save_portfolio_two(mt_port)
        return
    if not (main_enabled or mt_enabled):
        pf.save_portfolio(main_port)
        pf.save_portfolio_two(mt_port)
        return

    _log_activity_event("run_started", source="app_load")

    # Fit calibrators + K-N model once per session (cached)
    try:
        df_hash = len(df)
        calibrators = cached_calibrators(df_hash)
        dc_kn_r     = cached_dc_kn(f"{df_hash}_{df['Date'].max().date()}")
        bin_vars    = cached_bin_variances(df_hash)
    except Exception as e:
        _log_activity_event("error", stage="calibrate_load", error=str(e))
        pf.save_portfolio(main_port)
        pf.save_portfolio_two(mt_port)
        return

    # Live odds + fixtures
    try:
        fixtures = fetch_upcoming_fixtures(lookahead_days=14)
    except Exception as e:
        fixtures = []
        _log_activity_event("error", stage="fixtures", error=str(e))
    try:
        live_odds_map = pf.fetch_live_odds(api_key)
    except Exception as e:
        live_odds_map = {}
        _log_activity_event("error", stage="live_odds", error=str(e))

    if not fixtures or not live_odds_map:
        if not fixtures:
            _log_activity_event("warning", message="No upcoming fixtures (app load)")
        if not live_odds_map:
            _log_activity_event("warning", message="No live odds (app load)")
        pf.save_portfolio(main_port)
        pf.save_portfolio_two(mt_port)
        _log_activity_event("run_completed", source="app_load")
        return

    main_cands: list[dict] = []
    mt_cands:   list[dict] = []
    # The runner treats any club the model can rate as playable; the app used
    # `teams` = the most recent season's 20 sides, so it silently dropped
    # Coventry, Hull AND Ipswich — three of the ten openers, including the
    # largest edge on the board. Same rule in both now: rated means playable.
    rateable = set(teams) | set(dc_r.get("attacks", {}))
    for fix in fixtures:
        h, a = fix["home"], fix["away"]
        if h not in rateable or a not in rateable:
            # Was a bare continue, which made promoted sides invisible: in
            # 2026-27 that silently hid 2 of the 10 opening fixtures.
            unknown = [t for t in (h, a) if t not in rateable]
            _log_activity_event("fixture_skipped", reason="unknown_team",
                                match=f"{h} vs {a}",
                                detail=f"{', '.join(unknown)} not in the model's team list")
            continue
        api_o = live_odds_map.get((h, a), {})
        if not api_o or "H" not in api_o:
            continue
        try:
            hs  = get_current_stats(df, h, elo_dict=elo_dict)
            as_ = get_current_stats(df, a, elo_dict=elo_dict)
            dc_p, _dc_blend, main_res = full_predict(
                h, a, dc_r, dc_draw_r, xgb_m, feat_cols,
                draw_xgb_m, draw_fc, hs, as_,
            )
            _kn_p, _kn_blend, mt_res = full_predict_v2(
                h, a, dc_kn_r, dc_draw_r, xgb_m, feat_cols,
                draw_xgb_m, draw_fc, hs, as_,
            )
        except Exception as e:
            _log_activity_event("error", stage="predict",
                                match=f"{h} vs {a}", error=str(e))
            continue

        ds = fix["date"].isoformat() if hasattr(fix["date"], "isoformat") else str(fix["date"])
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

    placed_total = 0
    # No-history gate: promoted sides with too little top-flight history to be
    # rated. Counted once and shared by both lines.
    match_counts = team_match_counts(df)
    skip_log: list[dict] = []
    if main_enabled:
        thr = float(main_port["settings"].get("auto_bet_threshold", 0.40))
        placed = pf.auto_place_value_bets(main_port, main_cands, thr,
                                          calibrators=calibrators,
                                          match_counts=match_counts,
                                          skip_log=skip_log)
        for b in placed:
            _log_activity_event("auto_bet_placed", portfolio="main",
                                match=f"{b['home']} vs {b['away']}",
                                market=b["market"], selection=b["selection"],
                                stake=b["stake"], odds=b["odds"], ev=b["ev"])
        if placed:
            placed_total += len(placed)
            st.toast(f"🤖 Main: placed {len(placed)} bet"
                     f"{'s' if len(placed) > 1 else ''}")

    if mt_enabled:
        thr = float(mt_port["settings"].get("auto_bet_threshold", 0.40))
        placed = pf.auto_place_value_bets_v2(mt_port, mt_cands, thr,
                                             calibrators=calibrators,
                                             bin_variances=bin_vars,
                                             match_counts=match_counts,
                                             skip_log=skip_log)
        for b in placed:
            _log_activity_event("auto_bet_placed", portfolio="mt",
                                match=f"{b['home']} vs {b['away']}",
                                market=b["market"], selection=b["selection"],
                                stake=b["stake"], odds=b["odds"], ev=b["ev"])
        if placed:
            placed_total += len(placed)
            st.toast(f"🧪 Mock Two: placed {len(placed)} bet"
                     f"{'s' if len(placed) > 1 else ''}")

    # Flush gate skips, deduped: both lines scan the same fixtures, so an
    # unrated side would otherwise be logged once per market per portfolio.
    for key in dict.fromkeys(
            (e["reason"], e["home"], e["away"], e["detail"]) for e in skip_log):
        reason, home, away, detail = key
        _log_activity_event("fixture_skipped", reason=reason,
                            match=f"{home} vs {away}", detail=detail)

    pf.save_portfolio(main_port)
    pf.save_portfolio_two(mt_port)
    _log_activity_event("run_completed", source="app_load",
                        placed=placed_total)


def main():
    with st.spinner("Loading match data..."):
        try:
            df, df_features = cached_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    # Surface degraded xG state — Understat down means the model is silently
    # running on the shots-conversion proxy instead of real xG.
    try:
        from data import XG_COVERAGE
        if XG_COVERAGE:
            _cur_season = max(XG_COVERAGE)
            _cov = XG_COVERAGE[_cur_season]
            if _cov["total"] > 50 and _cov["matched"] / max(_cov["total"], 1) < 0.7:
                st.warning(
                    f"⚠️ Real xG coverage for {_cur_season} is only "
                    f"{_cov['matched']}/{_cov['total']} matches — Understat may be "
                    f"unreachable. Predictions are falling back to the "
                    f"shots-conversion proxy, which is weaker for draw detection."
                )
    except Exception:
        pass

    with st.spinner("Training models (Dixon-Coles MLE + XGBoost)..."):
        cache_key = f"{len(df)}_{df['Date'].max().date()}"
        poisson_r, dc_r, dc_draw_r, xgb_m, feat_cols, draw_xgb_m, draw_fc, elo_dict = cached_models(cache_key)

        # Promoted sides carry no rating, and an unrated team reads as league
        # average rather than as unknown. Seed them before anything predicts.
        try:
            _fx = cached_fixtures()
            _fx_key = "|".join(f"{f['home']}~{f['away']}" for f in _fx)
            dc_r, dc_draw_r = cached_promoted_seeding(
                cache_key, _fx_key, dc_r, dc_draw_r)
            # Elo too. Seeding on top of the finished dict only lasts until
            # the team plays: the series is recomputed from scratch each time
            # and would restart them at 1500, so a promoted side could LOSE its
            # opener and come out rated higher. Feed the market rating in as the
            # ENTRY rating instead, so it survives contact with results.
            _names = [t for f in _fx for t in (f["home"], f["away"])]
            elo_dict = get_current_elo(
                df, entry_offsets=promoted_elo_offsets(_names))
            # Before a promoted side has played it is absent from the series
            # entirely, so fill it in for display and for candidate metadata.
            elo_dict = seed_promoted_elo(
                elo_dict, _names, active=get_current_teams(df))
        except Exception:
            pass

    teams = get_current_teams(df)

    # Run auto-bet pipeline once per session (or every 5+ min on rerun).
    # This replaces the launchd schedule for users who prefer the in-app
    # trigger — opening the app or refreshing the page now settles + places.
    _session_auto_bet(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                      draw_xgb_m, draw_fc, teams, elo_dict)

    active_view = st.session_state.get("_active_view")

    # ── Home screen ──────────────────────────────────────────────────────
    if active_view is None:
        _render_home_screen()
        return

    # ── In-tab view: compact top bar + routed content ────────────────────
    _render_top_bar(active_view)

    if active_view == "predict":
        tab_predict(df, df_features, poisson_r, dc_r, dc_draw_r,
                    xgb_m, feat_cols, draw_xgb_m, draw_fc, teams, elo_dict)
    elif active_view == "weekend":
        tab_weekend(df, dc_r, dc_draw_r, xgb_m, feat_cols,
                    draw_xgb_m, draw_fc, teams, elo_dict)
    elif active_view == "results":
        tab_results(df, dc_r, dc_draw_r, xgb_m, feat_cols,
                    draw_xgb_m, draw_fc, teams, elo_dict)
    elif active_view == "backtest":
        tab_backtest(len(df))
    elif active_view == "season":
        tab_season(df, dc_r)
    elif active_view == "review":
        tab_season_review(df)
    elif active_view == "elo":
        tab_elo(df, teams)
    elif active_view == "teamdeep":
        tab_team_deepdive(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                          draw_xgb_m, draw_fc, teams, elo_dict)
    elif active_view == "portfolio":
        tab_portfolio(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                      draw_xgb_m, draw_fc, teams, elo_dict)
    elif active_view == "portfolio2":
        tab_portfolio_two(df, df_features, dc_r, dc_draw_r, xgb_m, feat_cols,
                          draw_xgb_m, draw_fc, teams, elo_dict)
    elif active_view == "preflight":
        tab_preflight(df, dc_r, dc_draw_r, xgb_m, feat_cols,
                      draw_xgb_m, draw_fc, teams, elo_dict)
    else:
        st.session_state["_active_view"] = None
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;font-size:0.78rem;color:#9aa6ba;letter-spacing:2px">'
        'DATA: FOOTBALL-DATA.CO.UK + UNDERSTAT · MODEL: DIXON-COLES + POISSON + XGBOOST · '
        'FOR ENTERTAINMENT PURPOSES</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
