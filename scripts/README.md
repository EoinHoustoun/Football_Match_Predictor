# Background auto-bet runner

`run_auto_bet.py` runs the full F_PRED auto-bet pipeline outside of Streamlit so
bets get placed on schedule (not only when you have the app open).

## What it does each run
1. Loads the match data + fits Dixon-Coles, Draw Specialist, K-N γ, and XGBoost models.
2. Fits isotonic calibrators from the 10-week walk-forward backtest.
3. Auto-settles any pending bets that the football-data CSV has now resolved.
4. Backfills CLV (closing-line value) for newly-settled bets.
5. Fetches upcoming fixtures + live odds (UK + EU regions, includes Pinnacle).
6. Builds candidate bets and runs `auto_place_value_bets` (Main) and
   `auto_place_value_bets_v2` (Mock Two), respecting each portfolio's gates,
   filters, and `auto_bet_enabled` toggle.
7. Appends every action to `data/activity.log` (JSON Lines), which the
   home-screen activity feed reads.

The script never overwrites bet history or bankroll outside of the
`auto_settle` / `auto_place_value_bets` paths the Streamlit UI already uses —
both portfolios go through the same code, so this is functionally equivalent
to opening the app once an hour and letting it tick.

## Running it manually
```bash
cd /Users/eoinhoustoun/Desktop/Projects/Football\ Analytics/Claude/F_PRED
python3 scripts/run_auto_bet.py
```

Total run time ≈ 30–60 s (most of it is fitting the Dixon-Coles MLE).

## Scheduling on macOS (launchd)

The repo includes `com.eoinhoustoun.fpred.plist` configured to fire hourly.

```bash
# Copy the plist to the user-level LaunchAgents directory
cp scripts/com.eoinhoustoun.fpred.plist ~/Library/LaunchAgents/

# Load it
launchctl load ~/Library/LaunchAgents/com.eoinhoustoun.fpred.plist

# Verify
launchctl list | grep fpred

# To stop / unload later:
launchctl unload ~/Library/LaunchAgents/com.eoinhoustoun.fpred.plist
```

stdout goes to `~/Library/Logs/fpred/autobet.stdout.log`, errors to `~/Library/Logs/fpred/autobet.stderr.log` (not under `data/`: launchd cannot open log files inside ~/Desktop because of macOS privacy protection, and the job then dies with exit 78 EX_CONFIG before it starts).
The `data/activity.log` file (JSON Lines) is the canonical event stream — that
is what the Streamlit home screen reads.

## Scheduling on Linux/cron
```cron
# every hour at :05 past
5 * * * * cd /path/to/F_PRED && /usr/bin/env python3 scripts/run_auto_bet.py >> data/cron.log 2>&1
```

## Activity log format

`data/activity.log` is JSON Lines. Each line is one event:

```json
{"ts": "2026-05-10T14:05:12+00:00", "type": "run_started"}
{"ts": "2026-05-10T14:05:38+00:00", "type": "settled", "portfolio": "main", "match": "Liverpool vs Crystal Palace", "market": "D", "result": "lost", "profit": -924.61}
{"ts": "2026-05-10T14:05:52+00:00", "type": "auto_bet_placed", "portfolio": "main", "match": "Brentford vs Crystal Palace", "market": "D", "selection": "Draw", "stake": 540.0, "odds": 4.20, "ev": 0.41}
{"ts": "2026-05-10T14:05:53+00:00", "type": "clv_snapshot", "portfolio": "main", "median_clv": -0.0124, "pct_positive": 0.35, "n": 20}
{"ts": "2026-05-10T14:05:53+00:00", "type": "run_completed"}
```

Event types:
- `run_started` / `run_completed` — bookend each run
- `settled` — a pending bet was just resolved by the auto-settle path
- `auto_bet_placed` — a new bet was auto-placed by the value-scanner path
- `clv_snapshot` — end-of-run CLV summary per portfolio
- `warning` / `error` / `info` — diagnostics with a `message` field
- `fatal` — uncaught exception (run aborted)

The home-screen activity feed pulls the most-recent ~50 events to render
"Since last visit" tiles.

## Safety notes

- Both portfolios remain off-limits for unsupervised mutation outside the
  approved paths (`auto_settle`, `auto_place_value_bets`, `place_bet`,
  `backfill_clv_for_settled_bets`). The script doesn't touch the bet array
  except via these helpers.
- If the API key is missing or the portfolios both have `auto_bet_enabled`
  set to False, the script logs a warning and exits cleanly — no bets touched.
- Backups live next to `portfolio.json` / `portfolio_two.json`; restore via
  `cp data/portfolio.backup.YYYYMMDD_HHMMSS.json data/portfolio.json`.
