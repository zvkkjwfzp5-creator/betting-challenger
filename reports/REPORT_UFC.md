# Challenger test: UFC

Every number is out of sample: each season was predicted with settings chosen on earlier seasons only. Lower log loss and Brier are better. Intervals resample whole match weeks.

## Fight moneyline, corners randomised (the credible run)

4,719 games, seasons 2016 to 2026.

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6075 | 0.4202 | 0.0190 | 1.16 |
| C. New model alone | 0.6644 | 0.4719 | 0.0216 | 1.24 |
| D. Market + new model | 0.6072 | 0.4202 | 0.0132 | 1.08 |

Log-loss differences (negative = first model better):

- C new minus A market: +0.0569 (95% interval +0.0486 to +0.0645); better in 0/11 seasons
- D combo minus A market: -0.0003 (95% interval -0.0012 to +0.0006); better in 8/11 seasons

Change in the new model's log loss when one part is removed (positive = that part helps):

- Strength changing over time: +0.0001 (95% interval -0.0009 to +0.0010)
- Pooling toward league baseline: +0.0276 (95% interval +0.0197 to +0.0350)
- Age and reach: +0.0125 (95% interval +0.0095 to +0.0156)
- Parameter uncertainty (outer draws): -0.0008 (95% interval -0.0020 to +0.0004)

| Flat one-unit stakes | Bets | Return per bet | Profit (units) |
|---|---|---|---|
| C. New model, thresholds from earlier seasons | 0 | no selections | 0 |
| D. Combination, thresholds from earlier seasons | 0 | no selections | 0 |

Selections searched: 9,438. Seasons with no selection at all: C 11, D 11 of 11.

Verdict: the combination gives the new model a typical weight of 0.03 (market 1.00). Beats the market: no. Rule with a return interval above zero: none.

## Fight moneyline, corners as recorded (red-corner label is suspect)

4,719 games, seasons 2016 to 2026.

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6075 | 0.4202 | 0.0197 | 1.16 |
| C. New model alone | 0.6555 | 0.4635 | 0.0165 | 1.18 |
| D. Market + new model | 0.6063 | 0.4195 | 0.0161 | 1.07 |

Log-loss differences (negative = first model better):

- C new minus A market: +0.0480 (95% interval +0.0401 to +0.0556); better in 0/11 seasons
- D combo minus A market: -0.0012 (95% interval -0.0030 to +0.0005); better in 7/11 seasons

Change in the new model's log loss when one part is removed (positive = that part helps):

- Strength changing over time: -0.0001 (95% interval -0.0003 to +0.0000)
- Pooling toward league baseline: +0.0349 (95% interval +0.0262 to +0.0434)
- Age and reach: +0.0172 (95% interval +0.0130 to +0.0214)
- Parameter uncertainty (outer draws): -0.0006 (95% interval -0.0014 to +0.0002)

| Flat one-unit stakes | Bets | Return per bet | Profit (units) |
|---|---|---|---|
| C. New model, thresholds from earlier seasons | 0 | no selections | 0 |
| D. Combination, thresholds from earlier seasons | 515 | +10.4% (95% interval +0.4% to +20.3%) | +53.5 |

Selections searched: 9,438. Seasons with no selection at all: C 11, D 1 of 11.

Verdict: the combination gives the new model a typical weight of 0.11 (market 1.00). Beats the market: no. Rule with a return interval above zero: D_combo_rule.
