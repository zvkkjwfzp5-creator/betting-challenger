# Challenger test: summary across sports

Out-of-sample log loss (lower is better). The last column is the combination minus the market with its 95% interval; an interval that includes zero means no demonstrated improvement.

| Sport and market | Games | Market | Existing 70/30 blend | New model alone | Market + new model | Combination vs market |
|---|---|---|---|---|---|---|
| Soccer: match result | 27,163 | 0.9676 | 0.9741 | 0.9851 | 0.9676 | +0.0000 (-0.0001 to +0.0002) |
| Soccer: over/under 2.5 | 27,163 | 0.6725 | 0.6774 | 0.6790 | 0.6726 | +0.0001 (-0.0000 to +0.0002) |
| NHL: moneyline | 7,584 | 0.6682 | 0.6710 | 0.6713 | 0.6681 | -0.0001 (-0.0005 to +0.0004) |
| NBA: spread | 7,436 | 0.6931 | 0.6969 | 0.7118 | 0.6932 | +0.0000 (-0.0002 to +0.0003) |
| NBA: total | 7,474 | 0.6931 | 0.6967 | 0.7007 | 0.6927 | -0.0004 (-0.0011 to +0.0002) |
| NBA: win probability | 7,562 | 0.6100 | 0.6129 | 0.6262 | 0.6105 | +0.0005 (-0.0001 to +0.0011) |
| UFC: fight moneyline, corners randomised | 4,719 | 0.6075 | none | 0.6644 | 0.6072 | -0.0003 (-0.0012 to +0.0006) |
| UFC: fight moneyline, corners as recorded | 4,719 | 0.6075 | none | 0.6555 | 0.6063 | -0.0012 (-0.0030 to +0.0005) |

## Flat-stake returns of each betting rule

| Sport and market | Rule | Bets | Return per bet (95% interval) |
|---|---|---|---|
| Soccer: both markets | Existing rule (edge > 0, not flagged) | 3,434 | -6.8% (-12.3% to -1.4%) |
| Soccer: both markets | New model rule | 0 | no selections |
| Soccer: both markets | Combination rule | 0 | no selections |
| NHL: moneyline | Existing rule (edge > 0, not flagged) | 398 | -5.5% (-17.4% to +7.3%) |
| NHL: moneyline | Existing blend, edge > 0, flag ignored | 4,157 | -4.3% (-7.5% to -1.2%) |
| NHL: moneyline | New model rule | 39 | +29.9% (-44.7% to +123.0%) |
| NHL: moneyline | Combination rule | 0 | no selections |
| NBA: spread | Existing rule (edge > 0, not flagged) | 0 | no selections |
| NBA: spread | Existing blend, edge > 0, flag ignored | 4,286 | -4.5% (-7.2% to -1.7%) |
| NBA: spread | New model rule | 0 | no selections |
| NBA: spread | Combination rule | 0 | no selections |
| NBA: total | Existing rule (edge > 0, not flagged) | 0 | no selections |
| NBA: total | Existing blend, edge > 0, flag ignored | 4,604 | -3.3% (-6.1% to -0.5%) |
| NBA: total | New model rule | 0 | no selections |
| NBA: total | Combination rule | 0 | no selections |
| UFC: fight moneyline, corners randomised | New model rule | 0 | no selections |
| UFC: fight moneyline, corners randomised | Combination rule | 0 | no selections |
| UFC: fight moneyline, corners as recorded | New model rule | 0 | no selections |
| UFC: fight moneyline, corners as recorded | Combination rule | 515 | +10.4% (+0.4% to +20.3%) |

Notes: NBA spread and total bets are scored at an assumed -110. The UFC row with corners as recorded is not credible (the corner label appears to carry information about the result); the corner-randomised row is the one to rely on. NHL and NBA history ends in June 2022.
