# Challenger test: top-5 European leagues, seasons 2011/12 to 2026/27

Every number below is out of sample: each season was predicted with settings chosen on earlier seasons only. Lower log loss and Brier are better. Prices are Bet365 pre-match prices from football-data.co.uk.

## Match result (1X2): 27,163 matches

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.9676 | 0.5750 | 0.0030 | 1.00 |
| B. Existing model, sim only | 1.0299 | 0.6136 | 0.0451 | 0.66 |
| B. Existing model, 70/30 blend | 0.9741 | 0.5795 | 0.0068 | 0.97 |
| C. New model alone | 0.9851 | 0.5868 | 0.0134 | 1.12 |
| D. Market + new model | 0.9676 | 0.5751 | 0.0033 | 0.99 |

Log-loss differences (negative = first model better):

- B existing blend minus A market: +0.0065 (95% interval +0.0054 to +0.0077); better in 0/16 seasons
- C new minus A market: +0.0176 (95% interval +0.0155 to +0.0196); better in 0/16 seasons
- D combo minus A market: +0.0000 (95% interval -0.0001 to +0.0002); better in 6/16 seasons
- C new minus B existing raw: -0.0448 (95% interval -0.0502 to -0.0398); better in 16/16 seasons
- D combo minus B existing blend: -0.0065 (95% interval -0.0077 to -0.0053); better in 16/16 seasons

## Over/under 2.5 goals: 27,163 matches

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6725 | 0.4799 | 0.0015 | 1.01 |
| B. Existing model, sim only | 0.7249 | 0.5224 | 0.0947 | 0.27 |
| B. Existing model, 70/30 blend | 0.6774 | 0.4845 | 0.0164 | 0.82 |
| C. New model alone | 0.6790 | 0.4861 | 0.0085 | 0.89 |
| D. Market + new model | 0.6726 | 0.4800 | 0.0038 | 0.98 |

Log-loss differences (negative = first model better):

- B existing blend minus A market: +0.0049 (95% interval +0.0039 to +0.0058); better in 0/16 seasons
- C new minus A market: +0.0064 (95% interval +0.0052 to +0.0078); better in 0/16 seasons
- D combo minus A market: +0.0001 (95% interval -0.0000 to +0.0002); better in 6/16 seasons
- C new minus B existing raw: -0.0459 (95% interval -0.0503 to -0.0420); better in 16/16 seasons
- D combo minus B existing blend: -0.0048 (95% interval -0.0057 to -0.0039); better in 16/16 seasons

## Which parts of the new model help

Change in the new model's log loss when one part is removed (positive = that part helps):

| Removed part | 1X2 | Over/under 2.5 |
|---|---|---|
| Strength changing over time | +0.0106 (95% interval +0.0092 to +0.0120) | +0.0020 (95% interval +0.0010 to +0.0028) |
| Pooling toward league baseline | +0.0056 (95% interval +0.0041 to +0.0071) | +0.0094 (95% interval +0.0077 to +0.0111) |
| Low-score dependence | +0.0010 (95% interval +0.0006 to +0.0014) | +0.0000 (95% interval +0.0000 to +0.0000) |
| Parameter uncertainty (outer draws) | -0.0005 (95% interval -0.0006 to -0.0004) | +0.0001 (95% interval +0.0000 to +0.0003) |

## Flat one-unit stakes at the quoted price

| Rule | Bets | Return per bet | Profit (units) |
|---|---|---|---|
| B. Existing rule (blended EV > 0, not flagged) | 3,434 | -6.8% (95% interval -12.3% to -1.4%) | -234.5 |
| C. New model, thresholds from earlier seasons | 0 | no selections | 0 |
| D. Combination, thresholds from earlier seasons | 0 | no selections | 0 |

Selections searched in the evaluation window: 135,815 for the combination. Seasons where the rule chose no selection at all: C 16, D 16 of 16.

## Verdict

- Combination versus the existing 70/30 blend: **better** in both markets, every season, intervals excluding zero.
- Why: the combination learned to give the new model almost no weight (typical weight 0.04 on 1X2 and 0.03 on totals, against 1.00 for the market). In practice it is the margin-free market price. The gain over the existing blend comes from dropping the fixed 30% model weight, not from the new model knowing more than the market.
- Better than the market's own probabilities: **no**. No tested model beat the margin-free market price.
- Any betting rule with a return interval above zero: **none**.
