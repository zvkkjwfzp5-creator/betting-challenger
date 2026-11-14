# Challenger test: NHL

Every number is out of sample: each season was predicted with settings chosen on earlier seasons only. Lower log loss and Brier are better. Intervals resample whole match weeks.

## Moneyline at the opening price (OT included)

7,584 games, seasons 2016 to 2021.

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6682 | 0.4753 | 0.0109 | 1.06 |
| B. Existing model, sim only | 0.7014 | 0.5037 | 0.0693 | 0.42 |
| B. Existing model, 70/30 blend | 0.6710 | 0.4782 | 0.0080 | 0.92 |
| C. New model alone | 0.6713 | 0.4785 | 0.0191 | 1.24 |
| D. Market + new model | 0.6681 | 0.4751 | 0.0073 | 1.00 |
| Closing price (reference only, not available in time) | 0.6658 | 0.4732 | 0.0073 | 1.07 |

Log-loss differences (negative = first model better):

- B existing blend minus A market: +0.0028 (95% interval +0.0012 to +0.0044); better in 0/6 seasons
- C new minus A market: +0.0031 (95% interval +0.0006 to +0.0055); better in 1/6 seasons
- D combo minus A market: -0.0001 (95% interval -0.0005 to +0.0004); better in 4/6 seasons
- C new minus B existing raw: -0.0301 (95% interval -0.0372 to -0.0234); better in 6/6 seasons
- D combo minus B existing blend: -0.0028 (95% interval -0.0044 to -0.0013); better in 6/6 seasons

Change in the new model's log loss when one part is removed (positive = that part helps):

- Strength changing over time: +0.0116 (95% interval +0.0087 to +0.0145)
- Pooling toward league baseline: -0.0010 (95% interval -0.0019 to -0.0001)
- Extra regulation ties: -0.0003 (95% interval -0.0006 to -0.0001)
- Parameter uncertainty (outer draws): -0.0001 (95% interval -0.0001 to -0.0000)

| Flat one-unit stakes | Bets | Return per bet | Profit (units) |
|---|---|---|---|
| B. Existing rule (blended edge > 0, not flagged) | 398 | -5.5% (95% interval -17.4% to +7.3%) | -21.8 |
| B. Blended edge > 0, flag ignored | 4,157 | -4.3% (95% interval -7.5% to -1.2%) | -180.6 |
| C. New model, thresholds from earlier seasons | 39 | +29.9% (95% interval -44.7% to +123.0%) | +11.7 |
| D. Combination, thresholds from earlier seasons | 0 | no selections | 0 |

Selections searched: 15,168. Seasons with no selection at all: C 5, D 6 of 6.

Verdict: the combination gives the new model a typical weight of 0.06 (market 1.00). Beats the market: no. Beats the existing 70/30 blend: yes. Rule with a return interval above zero: none.
