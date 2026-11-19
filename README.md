# makefun-gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/danikluue-max/makefun-gta6-trophy-forecast-01-r3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/danikluue-max/makefun-gta6-trophy-forecast-01-r3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles a trophy, scored by a pinball loss on the logit scale. The program works in counts, not rates: Exophase's page only moves when it refreshes, and a refresh adds newly tracked players (who have earned almost nothing yet) while recording the progress of the players already tracked. So it reads every refresh step in the case's history, splits each step's new earners into old-player progress and new-cohort earns with a ridge-regularised two-parameter fit — the cohort prior is the aggregate new-cohort rate measured from the steps' own `trophies_earned` totals, and the fitted step totals are rescaled to match those observed totals exactly (count consistency). Players-tracked at the target comes from two estimates: a recency-weighted power law in log-time, and a discrete-refresh estimate that compounds the recent per-refresh growth over the refreshes the horizon holds; the program takes the LOWER of the two, because both estimators fail by overstating growth when a launch decelerates. When the history is flat it uses the first refresh the competition rules document (638 → 3,868 players, 943 → 4,276 trophies earned) with a small new-cohort rate (4% of each trophy's current rate) and a tight spread (0.17 logit). A log-time trend line is blended in as a hedge, and the spread comes from how much the steps and the trend disagree.

Round 3 vs round 2 (live loss 0.131, rank #375): practice loss across nine plausible allocations of the first refresh's 103 leftover earns improves 0.107 → 0.092, and on simulated final cases (two days of refresh history, 4.5-day horizon) it improves 0.115 → 0.082 when players keep growing as a power law and 0.420 → 0.160 when growth stalls — the discrete-refresh cross-check is what saves the stalled case.

## Run it

```sh
python3 -P -s /bot/main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=0zdpscn2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
