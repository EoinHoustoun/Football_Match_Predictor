# Can we rate a promoted side before it kicks a ball?

Three independent attempts, 2026-08-09. All three lose to a flat prior.

| Attempt | Information used | Strictly pre-season? | Attack | Defence |
|---|---|---|---|---|
| Championship form (`promoted_team_prior.py`) | DC attack/defence, GD, points, promotion route | yes | **lose** 0.236 vs 0.226 | **lose** 0.265 vs 0.236 |
| Market implied PPG, 1 match (`promoted_market_prior.py`) | opening-fixture 1X2 odds | yes | **lose** 0.235 vs 0.226 | **lose** 0.241 vs 0.236 |
| Market-solved DC rating (`promoted_opener_prior.py`) | opening-fixture 1X2, conditioned on the opponent's actual rating | yes | **lose** 0.232 vs 0.198 | **lose** 0.239 vs 0.235 |
| Market implied PPG, 3 matches | odds for matches 1-3 | **no** | win 0.208 vs 0.226 | win 0.189 vs 0.236 |

Leave-one-out throughout, against the incumbent of predicting the pooled mean.

## Why the pre-season attempts fail

**Championship form does not transfer.** Every one of the fifteen promoted teams
got worse in both phases, attack by 1.12 and defence by 1.30 on average, but
nothing about *how much* worse tracks anything the Championship table records.
Coventry's +52 goal difference bought them nothing over Hull's +4. This matches
the published work, which finds promoted sides drop sharply on xG and rise on
xGA but does not claim Championship stats separate the survivors.

**One match's price identifies strength but not its split.** Solving the
Dixon-Coles equations against a single opening fixture is exactly determined, two
unknowns against two independent probabilities, but badly conditioned: a strong
attack with a weak defence produces nearly the same home/draw/away split as the
reverse. The numbers show precisely that failure — attack correlates +0.058 with
the outcome, essentially nothing, while defence manages -0.335. The market knows
roughly how good the team is; the 1X2 price cannot say where that quality sits.

The solved attack ratings are also far too high in level, +0.1 to +0.6 against
actual season ratings of -0.3 to -0.9, which is the same ill-conditioning showing
up as bias rather than variance.

## What this licenses

**Keep the flat prior.** Every promoted side gets attack -0.645, defence +0.817
until it has played. This is not laziness, it is three failed attempts to do
better with the information available before kickoff.

**Keep the no-history gate.** Six matches before a promoted side can be staked.

## The one real opportunity

The three-match market view genuinely beats the flat prior, by 8% on attack and
20% on defence. It was excluded above because match 2's odds already know how
match 1 went, which disqualifies it as a *pre-season* prior.

That objection disappears once the season is running. At matchday 3 those results
are not leakage, they are information. So there is a validated improvement
available in the window that currently has nothing: **between matchday 3 and
matchday 6, a promoted side could carry a market-informed rating instead of the
flat prior**, bridging the gap until the model has enough history of its own.

Worth building in September. It changes model behaviour, so it needs Eoin's
sign-off rather than being applied quietly.

## What was ruled out as a data source

- **FBref** has Championship xG for Hull and Coventry but returns 403 to
  automated fetches, the same block hit on the transfer-intelligence project.
- **FootyStats** has it behind a paid API.
- **Understat** does not cover the Championship at all.
- **Historical pre-season relegation odds** exist only across scattered news
  articles using different bookmakers, formats and dates. Assembling fifteen
  teams from those would inject more inconsistency than the signal could survive.

None of these would have changed the conclusion, because the failure is not a
shortage of Championship detail. It is that the Championship does not predict the
Premier League.
