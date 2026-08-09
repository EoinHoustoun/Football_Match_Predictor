# 2026-27 re-validation: findings

Run: `scripts/validate_2026_27.py`, 2026-08-09.
Raw output: `data/diagnostics/validate_2026_27_20260809_203335.json`.

Four seasons (2022-23 … 2025-26), honest calibrators fit strictly before each
season, 14 configs on identical per-season packs.

## Headline

**Do not switch auto-bet on at full Kelly.** The deployed config is profitable in
the backtest, but the profit is one club in one season, and the bet counts are
too small for the season totals to mean what they look like.

## What the run showed

| variant | 2022-23 | 2023-24 | 2024-25 | 2025-26 | worst | bets | CLV | max 1-club share |
|---|---|---|---|---|---|---|---|---|
| deployed (gate on) | +6,809 | +923 | +17,529 | +51,222 | +923 | 43 | +2.92% | **79%** |
| deployed, gate off | +6,809 | +923 | +17,529 | +95,943 | +923 | 46 | +2.92% | 82% |
| EV gate 30% | -4,444 | +5,474 | +46,574 | +56,754 | -4,444 | 66 | +2.88% | 65% |
| EV gate 50% | +526 | -1,367 | +4,358 | +114,406 | -1,367 | 27 | +2.94% | 73% |
| min_prob 0.25 | +6,809 | -4,902 | +17,529 | +51,222 | -4,902 | 40 | +2.92% | **100%** |
| Elo floor 1450 | +1,946 | -2,454 | +23,982 | +112,394 | -2,454 | 71 | +3.02% | 67% |
| half Kelly | +5,968 | +1,052 | +8,913 | +31,560 | +1,052 | 43 | +2.92% | 79% |
| club cap 5 | -39 | +923 | +17,529 | +10,031 | -39 | 36 | +3.00% | 62% |
| club cap 3 | +4,314 | +923 | +4,565 | +1,778 | +923 | 28 | +3.01% | 50% |
| club cap 7 | +6,809 | +923 | +17,529 | +1,706 | +923 | 39 | +2.92% | 70% |
| club cap 5 + half Kelly | +1,883 | +1,052 | +8,913 | +9,443 | **+1,052** | 36 | +3.00% | 62% |

## 1. The 2025-26 number is Sunderland

Of 14 bets placed in 2025-26, **11 were Sunderland matches**. Sunderland
returned **+£66,577** against a season total of **+£51,222**, so every other club
combined **lost £15k**. This is the same shape as the £100k sweep artifact
already documented in `sweep_100k_findings.md`, and it is the same club.

The other seasons are milder but not clean: the top club takes 43% (Brighton,
2022-23), 50% (Arsenal, 2023-24) and 44% (Liverpool, 2024-25) of that season's
bets.

## 2. The no-history gate helps, but it is not the answer

The gate cut 2025-26 from +£95,943 to +£51,222 by blocking three September
Sunderland bets. It does not touch the other eight, because after six matches
Sunderland is "rated" and the run continues. Worst season and losing-season
count are unchanged, so the gate costs nothing in survival terms and removes the
most speculative end of the run. Keep it. Do not mistake it for concentration
control.

## 3. The season totals are path-dependent, not stable

**Club cap 7 returns less in 2025-26 (+£1,706) than club cap 5 (+£10,031),
despite being the looser constraint.** A looser filter producing a worse result
is not a signal about the filter. It means the season total is decided by which
specific bets land and how the bankroll happens to compound through them. At
full Kelly with 25% max stake, one January bet in the gate-on log staked £13,149
and returned +£36,816 on a £10,000 starting bankroll.

Anything quoted to the nearest thousand from 6-14 bets a season is noise
dressed as precision.

## 4. CLV is the one stable, encouraging signal

Median CLV sits between **+2.85% and +3.05% across every single variant**,
including the ones that lose money. The model is picking bets that beat the
Pinnacle close. That is the leading indicator and it says the *selection* has
real value. The instability is in the *sizing*, not the picks.

## What was applied

- **`max_bets_per_club: 5` added to Main.** A risk control, chosen a priori (no
  club should carry more than roughly a third of a season at 6-14 bets a
  season), not by picking the best-backtesting number. It is a tightening, so it
  is allowed direct to live under the no-tinker rule. Backup taken first.
- **Auto-bet stays OFF.** Nothing here earns switching it on.

## What was not applied, and needs Eoin's call

- **Kelly fraction 1.0 → 0.5.** Half Kelly has the best worst season of any
  variant tested (+£1,052, no losing season), roughly halves the headline
  numbers, and leaves CLV untouched. It is a sizing change, which the
  no-tinker rule sends to sandbox rather than straight to live, so it is not
  applied. Given that section 3 shows the sizing is what makes the P&L a
  lottery, this is the change most likely to matter.

## What this does not cover

- Four seasons and 43 bets is a small sample. It cannot separate a 3% edge from
  a 0% edge.
- Every variant shares the same model, so a blind spot in Dixon-Coles or in the
  draw specialist is invisible here.
- The promoted-team prior is fitted, not observed: no promoted side has played a
  match under it yet.
- 2026-27 has no closing-odds history yet, so live CLV cannot be checked until
  bets settle.
