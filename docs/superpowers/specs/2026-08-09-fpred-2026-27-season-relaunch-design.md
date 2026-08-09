# F_PRED — 2026/27 Season Relaunch

**Date:** 2026-08-09
**Status:** Approved
**Kickoff:** Friday 21 August 2026, Arsenal v Coventry City (12 days out)

## Goal

Close 2025/26 as a preserved, browsable record. Start 2026/27 with two fresh
£10,000 paper-trading lines driven by the draw model and The Odds API, run on a
schedule rather than on page loads, with the promoted-team blind spot closed and
the config re-validated on five seasons of data.

## Where we are starting from

| | Start | End | P&L | Bets | Median CLV |
|---|---|---|---|---|---|
| Main | £10,000 | £18,251.75 | +£8,251.75 | 29 | -0.83% |
| Mock Two | £10,000 | £16,972.20 | +£6,972.20 | 16 | +1.72% |

Live Main config, label `2026-27 defaults`, validated 2026-06-09:
`min_prob=0.21, min_ev=0.40, kelly_fraction=1.0, max_stake_pct=0.25`,
Monday ban, `main_min_team_elo=1500`, `DRAW_PROB_CAP=0.45`,
`auto_markets=["D","under25"]`, `skip_late_season=True`.

Verified working on 2026-08-09:
- `data.py` season config auto-rolled to 2026-27. No manual edit needed.
- ESPN returns all 10 opening fixtures with names resolved, except Coventry City
  and Hull City which have no football-data counterpart in the loaded dataset.
- The Odds API returns the opening round from 39 books including Pinnacle.
  494 of 500 monthly credits remaining. One h2h pull over `regions=uk,eu` costs
  2 credits.
- `football-data.co.uk/mmz4281/2627/E0.csv` returns 404. No results file exists
  until after the first matchday.
- The app-load auto-bet path ran clean against live 2026/27 odds and placed
  nothing. No stray bets landed on the 2025/26 portfolio.

## Decisions taken

| Question | Decision |
|---|---|
| Archiving | Build a Season Archive with a season selector in Season Review |
| Structure | Main £10k + Mock Two £10k, both draws-only |
| Mock Two variant | Detect at Pinnacle, place at best book. Single variable. |
| Promoted teams | Explicit no-history gate at 6 matches, plus visible warnings |
| Starting config | Re-validate on five seasons before deploying |
| Risk posture | Compound aggressively, gates untouched, drawdown throttle on |
| Autobet cadence | Local launchd agent, Thursday / Friday / Saturday |
| Markets | Draws only, both portfolios |
| Validation rule | Tightening ships automatically. Loosening needs Eoin's sign-off. |

## Components

### A. Season archive (`season_archive.py`, new module)

A new top-level module rather than more surface area on `portfolio.py`, which is
already 98k.

Layout:

```
data/seasons/2025-26/
    portfolio.json        verbatim copy of the live main portfolio
    portfolio_two.json    verbatim copy of the live research portfolio
    manifest.json         closing summary, see below
```

`manifest.json` holds, per portfolio: `initial_bankroll`, `final_bankroll`,
`profit`, `roi_pct`, `n_bets`, `n_settled`, `win_rate`, `median_clv`,
`settings_label`. Plus `season`, `closed_at`, and `archived_by`.

Interface:

- `archive_season(season: str, *, overwrite: bool = False) -> dict`
  Copies both live portfolio files into the archive directory and computes the
  manifest. Refuses if the directory already exists unless `overwrite=True`.
  Returns the manifest.
- `list_archived_seasons() -> list[str]` — sorted, newest first.
- `load_archived_season(season: str) -> dict` — `{"manifest", "main", "mock_two"}`.
- `reset_for_new_season(season, initial_bankroll, main_settings, two_settings) -> None`
  Writes fresh live portfolio files. **Refuses to run unless the outgoing season
  is already archived.** This is the guard that makes the reset non-destructive.

Safety, per the portfolio-safety rule in CLAUDE.md: timestamped backups of both
live files before any write, then `diff` verification that the archived copies
are byte-identical to the pre-reset originals. `data/seasons/` is added to
`.gitignore` alongside the portfolio files.

### B. Season Review becomes multi-season

`tab_season_review(df)` currently calls `pf.load_portfolio()` /
`pf.load_portfolio_two()` directly and derives its season label from
`_offseason_info(df)`. It gains a season selector at the top listing the live
season first, then archived seasons.

When an archived season is selected, portfolios come from
`load_archived_season()` and the match dataframe is sliced to that season for
the final table and champion. Every downstream calculation in the tab already
works off portfolio dicts and a dataframe slice, so no chart logic changes.

The live season shows an in-progress state rather than a wrap-up until the
season ends.

### C. Promoted-team safety

Three tightening-only changes.

1. **No-history gate.** A team with fewer than 6 Premier League matches in the
   loaded dataset cannot be bet on. Implemented as a shared predicate so it
   applies identically in `auto_place_value_bets`, `auto_place_value_bets_v2`,
   `ev_backtest_simulate` and `ev_backtest_simulate_v2`. Validating the gate on
   historical data requires it to run in the simulators too.

2. **ELO floor excludes the unrated default.** Today `min_team_elo` rejects on
   `elo < 1500` while an unseen team defaults to exactly 1500, so unknown teams
   pass the filter that exists to exclude weak sides. The floor will require a
   team that is actually rated. This closes the same failure mode that produced
   last season's promoted-side calibrator artifact.

3. **Silent skips become visible.** `if h not in teams or a not in teams:
   continue` currently drops fixtures with no trace. It will record a structured
   skip reason, surfaced on the GW board and the pre-flight check.

For 2026/27 this means Coventry and Hull are excluded until roughly mid-October.
Ipswich is rated from 2024/25 and is unaffected.

### D. Re-validation (`scripts/validate_2026_27.py`, new)

Walk-forward across 2021-22 to 2025-26, draws-only, with the no-history gate
active. Compares the live config against tightened neighbours and reports
fold-by-fold out-of-sample P&L and CLV, not just aggregates.

Follows `cached_honest_calibrators` discipline: calibrators fit strictly before
each evaluation window. The June session established that the in-app backtest
default window matters enormously (27 weeks shows a loss where 40 weeks shows a
large profit on the same config), so the report states its window explicitly.

The `validating-model-changes` skill governs this step.

Outcome rule: changes that tighten ship. Anything that wants loosening is
brought to Eoin with the fold table and the failure mode it opens up.

### E. Mock Two: detect/place

Mock Two resets to £10,000, draws-only, gates identical to Main. The single
difference is `detect_source="PS"` and placement at best available price.

This requires plumbing detect/place into `auto_place_value_bets_v2`, which is
the only genuinely new betting-engine code in this spec. `fetch_live_odds`
already returns a `_pinnacle` sub-dict and a `_books` breakdown, so the data is
present. The work is candidate construction and graceful handling of fixtures
where Pinnacle has no price, which must skip rather than silently fall back to
the best-book price as its own benchmark.

The K-N and Baker-McHale stack comes off so the A/B has one variable. Rationale:
that stack already lost the out-of-sample head-to-head (+£12k vs +£14k), while
detect/place produced the cleanest edge signal on record (+2.68% median CLV,
77.7% of bets beating the close across 14 folds) and has never been live.

### F. Scheduled runner (`scripts/run_gameweek.py` + launchd)

Headless, no Streamlit import. Per run: auto-settle, CLV backfill, auto-bet both
portfolios, append to `data/activity.log`, write a summary to `data/gw_runs/`.

Driven by a launchd agent at
`~/Library/LaunchAgents/com.eoinhoustoun.fpred.gameweek.plist`, firing Thursday,
Friday and Saturday mornings.

Credit budget: one odds pull per run at 2 credits, shared by both portfolios.
Three runs a week is 6 credits per gameweek, roughly 24 a month against a 500
limit.

If the Mac is asleep the run does not fire. Missed runs are detected by the
pre-flight check comparing the last run timestamp against the fixture list,
rather than failing silently.

### G. UI

**GW board.** One row per fixture: model draw probability raw and calibrated,
best price and which book, the Pinnacle line, computed edge, Kelly stake, and a
pass/fail chip per gate (EV, min prob, ELO floor, no-history, day-of-week ban,
late season). Makes the autobet's reasoning legible instead of a black box.

**Pre-flight check.** One screen verifying, before each gameweek: fixtures
resolving, every fixture team rated, odds API responding with credits remaining,
calibrator freshness, promoted-gate status, and last scheduled run timestamp.
This is the screen that would have caught the Coventry and Hull problem.

**Home screen for the new season.** Countdown to Arsenal v Coventry, fresh £10k
tiles, promoted and relegated sides, and the 2025/26 result moved into an
archive card linking to Season Review.

Design system is binding: vibrant dark palette, Inter, minimum font size
0.78rem, badges over emoji, new features become tiles rather than sidebar items.

## Build order

1. Season archive and reset. First, so nothing can touch 2025/26 afterwards.
2. Promoted-team safety. Before validation, so the validation is honest.
3. Re-validation, and deploy the resulting config.
4. Mock Two detect/place.
5. Scheduled runner.
6. UI: pre-flight, GW board, home screen.

Steps 1 through 5 complete before 21 August. UI polish may continue past kickoff.

## Testing

Currently 111 passing. New coverage:

- Archive round-trip: archive, verify manifest arithmetic, load back, confirm
  the reset guard refuses when the season is not archived.
- No-history gate: below threshold blocks, at threshold passes, and it behaves
  identically in the live path and the simulators.
- ELO floor: an unrated 1500 default is rejected where a genuinely rated 1500 is
  not.
- Detect/place: Pinnacle present sizes off the Pinnacle benchmark, Pinnacle
  absent skips rather than falling back.
- Headless runner: runs end to end against fixtures without a Streamlit context.

## Out of scope

- **Multi-season track record page.** The archive produces the data for a
  lifetime bankroll curve and per-season ROI. Deliberately deferred.
- Motivation features into the draw XGB, Pinnacle-residual feature, portfolio-tab
  dedup refactor. These remain on the research queue.
- Real money. This stays paper trading throughout.
