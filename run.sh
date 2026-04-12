#!/bin/bash
set -e

echo ""
echo "  ⚽  Premier League Predictor"
echo "  ─────────────────────────────"
echo ""

# Install dependencies
echo "  Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "  Starting app at http://localhost:8501"
echo ""

streamlit run app.py --server.port 8501
