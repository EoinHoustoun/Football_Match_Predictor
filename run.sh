#!/bin/bash
# F_PRED app launcher.
#
# The UI runs from .venv (Python 3.11, Streamlit 1.64). The numerical stack is
# pinned to the exact versions the launchd runner uses on the framework Python
# 3.9 (numpy 1.24.3, pandas 2.0.3, scipy 1.13.1, scikit-learn 1.6.1, xgboost
# 2.1.4), so the app-load auto-bet and the hourly runner compute identical
# probabilities. Verified 25 Sep 2026: 380 fixture pairings, max difference 0.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "  Creating .venv (Python 3.11)..."
  uv venv .venv --python 3.11
  uv pip install --python .venv/bin/python -r requirements-app.txt
fi

echo ""
echo "  ⚽  Premier League Predictor · http://localhost:8501"
echo ""
exec .venv/bin/python -m streamlit run server.py --server.port 8501
