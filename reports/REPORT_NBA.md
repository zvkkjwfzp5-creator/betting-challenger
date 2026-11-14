# Challenger test: NBA

Every number is out of sample: each season was predicted with settings chosen on earlier seasons only. Lower log loss and Brier are better. Intervals resample whole match weeks.

## Spread at the opening line (price assumed -110)

7,436 games, seasons 2016 to 2021.

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6931 | 0.5000 | 0.0003 | 1.00 |
| B. Existing model, sim only | 0.7414 | 0.5407 | 0.1182 | 0.00 |
| B. Existing model, 70/30 blend | 0.6969 | 0.5037 | 0.0397 | -0.02 |
| C. New model alone | 0.7118 | 0.5174 | 0.0696 | -0.01 |
| D. Market + new model | 0.6932 | 0.5000 | 0.0144 | 0.22 |

Log-loss differences (negative = first model better):

- B existing blend minus A market: +0.0038 (95% interval +0.0019 to +0.0058); better in 0/6 seasons
- C new minus A market: +0.0187 (95% interval +0.0139 to +0.0239); better in 0/6 seasons
- D combo minus A market: +0.0000 (95% interval -0.0002 to +0.0003); better in 4/6 seasons
- C new minus B existing raw: -0.0296 (95% interval -0.0366 to -0.0227); better in 6/6 seasons
- D combo minus B existing blend: -0.0037 (95% interval -0.0057 to -0.0018); better in 6/6 seasons

Change in the new model's log loss when one part is removed (positive = that part helps):

- Strength changing over time: +0.0614 (95% interval +0.0507 to +0.0728)
- Pooling toward league baseline: +0.0015 (95% interval +0.0004 to +0.0027)
- Back-to-back adjustment: +0.0025 (95% interval +0.0013 to +0.0037)
- Parameter uncertainty (outer draws): +0.0018 (95% interval +0.0016 to +0.0021)

| Flat one-unit stakes | Bets | Return per bet | Profit (units) |
|---|---|---|---|
| B. Existing rule (blended edge > 0, not flagged) | 0 | no selections | 0 |
| B. Blended edge > 0, flag ignored | 4,286 | -4.5% (95% interval -7.2% to -1.7%) | -192.5 |
| C. New model, thresholds from earlier seasons | 0 | no selections | 0 |
| D. Combination, thresholds from earlier seasons | 0 | no selections | 0 |

Selections searched: 15,124. Seasons with no selection at all: C 6, D 6 of 6.

Verdict: the combination gives the new model a typical weight of 0.01 (market 1.00). Beats the market: no. Beats the existing 70/30 blend: yes. Rule with a return interval above zero: none.

## Total at the opening line (price assumed -110)

7,474 games, seasons 2016 to 2021.

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6931 | 0.5000 | 0.0003 | 1.00 |
| B. Existing model, sim only | 0.7529 | 0.5468 | 0.1224 | 0.02 |
| B. Existing model, 70/30 blend | 0.6967 | 0.5035 | 0.0341 | 0.11 |
| C. New model alone | 0.7007 | 0.5072 | 0.0489 | 0.19 |
| D. Market + new model | 0.6927 | 0.4996 | 0.0138 | 1.13 |

Log-loss differences (negative = first model better):

- B existing blend minus A market: +0.0035 (95% interval +0.0010 to +0.0061); better in 0/6 seasons
- C new minus A market: +0.0076 (95% interval +0.0036 to +0.0114); better in 0/6 seasons
- D combo minus A market: -0.0004 (95% interval -0.0011 to +0.0002); better in 5/6 seasons
- C new minus B existing raw: -0.0522 (95% interval -0.0655 to -0.0399); better in 6/6 seasons
- D combo minus B existing blend: -0.0040 (95% interval -0.0065 to -0.0016); better in 6/6 seasons

Change in the new model's log loss when one part is removed (positive = that part helps):

- Strength changing over time: +0.0431 (95% interval +0.0359 to +0.0502)
- Pooling toward league baseline: +0.0029 (95% interval +0.0016 to +0.0041)
- Back-to-back adjustment: +0.0000 (95% interval -0.0000 to +0.0000)
- Parameter uncertainty (outer draws): +0.0007 (95% interval +0.0005 to +0.0008)

| Flat one-unit stakes | Bets | Return per bet | Profit (units) |
|---|---|---|---|
| B. Existing rule (blended edge > 0, not flagged) | 0 | no selections | 0 |
| B. Blended edge > 0, flag ignored | 4,604 | -3.3% (95% interval -6.1% to -0.5%) | -151.5 |
| C. New model, thresholds from earlier seasons | 0 | no selections | 0 |
| D. Combination, thresholds from earlier seasons | 0 | no selections | 0 |

Selections searched: 15,106. Seasons with no selection at all: C 6, D 6 of 6.

Verdict: the combination gives the new model a typical weight of 0.13 (market 1.00). Beats the market: no. Beats the existing 70/30 blend: yes. Rule with a return interval above zero: none.

## Win probability (no opening moneyline in the data: scored, not bet)

7,562 games, seasons 2016 to 2021.

| Model | Log loss | Brier | Calibration error | Calibration slope |
|---|---|---|---|---|
| A. Market (margin removed) | 0.6100 | 0.4218 | 0.0206 | 0.91 |
| B. Existing model, sim only | 0.6516 | 0.4540 | 0.0490 | 0.64 |
| B. Existing model, 70/30 blend | 0.6129 | 0.4242 | 0.0135 | 0.92 |
| C. New model alone | 0.6262 | 0.4365 | 0.0102 | 0.95 |
| D. Market + new model | 0.6105 | 0.4222 | 0.0164 | 0.91 |
| Closing price (reference only, not available in time) | 0.6052 | 0.4180 | 0.0083 | 0.94 |

Log-loss differences (negative = first model better):

- B existing blend minus A market: +0.0029 (95% interval +0.0011 to +0.0048); better in 1/6 seasons
- C new minus A market: +0.0162 (95% interval +0.0116 to +0.0210); better in 0/6 seasons
- D combo minus A market: +0.0005 (95% interval -0.0001 to +0.0011); better in 4/6 seasons
- C new minus B existing raw: -0.0254 (95% interval -0.0325 to -0.0189); better in 6/6 seasons
- D combo minus B existing blend: -0.0024 (95% interval -0.0041 to -0.0008); better in 5/6 seasons

Change in the new model's log loss when one part is removed (positive = that part helps):

- Strength changing over time: +0.0534 (95% interval +0.0446 to +0.0627)
- Pooling toward league baseline: +0.0005 (95% interval -0.0005 to +0.0015)
- Back-to-back adjustment: +0.0019 (95% interval +0.0009 to +0.0029)
- Parameter uncertainty (outer draws): +0.0006 (95% interval +0.0001 to +0.0010)

Verdict: the combination gives the new model a typical weight of 0.09 (market 1.00). Beats the market: no. Beats the existing 70/30 blend: yes.
