# Betting challenger

A challenger to the existing prediction model, built so it can be tested
honestly before anyone relies on it. It does not place, size or recommend
bets on its own: betting decisions stay manual.

## Scope

| Sport | Model (scoring distribution) | Markets tested | History and price used |
|---|---|---|---|
| Soccer, top-5 leagues | Dynamic Poisson attack/defence, low-score dependence | Match result, over/under 2.5 | 45,559 matches 2000-2026, Bet365 pre-match |
| NHL | Dynamic Poisson on regulation goals, extra ties, fitted overtime step | Moneyline incl. OT | 13,667 games 2011/12-2021/22, opening moneyline |
| NBA | Dynamic Gaussian points model (margin and total from one pair of scores), back-to-back effect | Spread and total at the opening line; win probability (scored only) | 13,893 games 2011/12-2021/22, opening lines, price assumed -110 |
| UFC | Dynamic paired-comparison fighter ratings with age and reach | Fight moneyline | 7,169 fights 2010-2026, one pre-fight price |

The existing model (Poisson or normal sim from last-10 and season rates, then
a 70/30 market blend, with the 7-point flag) is written down in
`challenger/champion.py` so it can be tested. UFC has no existing model: the
instructions use a "structured estimate" that restates the market.

Priced by the same score distribution but **not validated**, so not eligible
for selection: Asian handicap, NHL totals and puck line, any combination.

## Layout

```
challenger/
  data.py        load results and prices
  market.py      margin removal from complete outcome sets (3 methods)
  champion.py    the existing model, as code
  dynamic.py     dynamic hierarchical attack/defence state-space model
  tuning.py      chronological choice of pooling and evolution rates
  scoregrid.py   one score matrix -> every market, settlement incl. pushes
  predictive.py  nested pricing: parameter draws x conditional outcomes
  combine.py     regularised market + model combination
  selection.py   expected return across draws, thresholds, "no selection"
  evaluate.py    log loss, Brier, calibration, block bootstrap
  backtest.py    soccer: nested walk-forward comparison and ablations
  data_us.py     NHL / NBA / UFC loaders (date repair, name clean-up, price checks)
  gaussian.py    NBA points model (exact Kalman filter)
  rating.py      UFC fighter-rating model
  nhl.py nba.py ufc.py   per-sport back-tests
  twoway.py      shared walk-forward harness for two-outcome markets
  mathx.py       normal CDF, logistic fit
  live_nhl.py    daily NHL model numbers from the NHL schedule feed
  report.py      short summaries;  cli.py  entry points
tests/           47 tests (leak checks, settlement, pushes, shrinkage, selection)
config.toml      every tunable and search grid
```

## Run

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python -m challenger.cli backtest
```

```bash
python -m challenger.cli backtest --sport nhl
```

```bash
python -m challenger.cli backtest --sport nba
```

```bash
python -m challenger.cli backtest --sport ufc
```

```bash
python -m challenger.cli price --league E0 --home Arsenal --away Chelsea --odds-1x2 1.80,3.80,4.50 --odds-ou 1.75,2.10
```

Daily NHL model numbers from this season's results (needs network access to
`api-web.nhle.com`; results since 2022 are kept in `live/nhl_results.csv`):

```bash
python -m challenger.cli live --date 2026-11-14 --prices '{"Red Wings@Lightning": [1.50, 2.76]}'
```

Prices are optional decimal prices as (home, away), keyed "Away@Home". The
output is information for a manual decision: the back-test found no edge.
NBA has no live step yet: the official NBA feed refuses automated access.

First runs fit every candidate setting and cache it under `data/` (soccer
about 11 minutes, NHL 2, NBA 1, UFC about 25 on 4 cores). Data files are
downloaded if missing. Results go to `reports/` (`SUMMARY.md` covers all four
sports). The `price` command covers soccer only.

## How the test is kept honest

- Season `s` is predicted with settings chosen on seasons before `s`: the
  model's pooling and evolution rates, the low-score parameter, the margin
  removal method, the combination weights and their ridge, and the selection
  thresholds.
- All matches on a date are predicted before any result from that date is used.
- The price used is one bookmaker's pre-match price. Closing prices are not in
  the data and are not used. The source does not record the capture time of
  each price, so "pre-match" is as precise as it gets.
- One bookmaker only, so several books are never counted as separate evidence.
- Intervals resample whole match weeks, not individual bets.
- Thresholds are accepted only if the lower bound of their earlier return is
  above zero; otherwise the rule for that season is "no selection".

## Missing data (not modelled, on purpose)

- All sports: line-ups, injuries, minutes, and the capture time of each price.
  Scenarios can be passed to `price` as shifts of the goal rates; without an
  evidence-based weight they are shown as sensitivity cases, never averaged in.
- Soccer: rest and travel (cup and European fixtures are absent), xG.
- NHL: goalies, xG, a complete price set for totals, opening puck-line prices.
- NBA: prices for spread and total (assumed -110), an opening moneyline, pace,
  travel. About 7% of closing lines in the archive are corrupted; closing
  numbers are used only as a reference row, never as an input.
- UFC: the corner label is suspect (see `load_ufc`); method-of-victory prices
  are incomplete sets. Ratings use UFC fights only, a median of four per fighter.
- Player-level history for props: none, for any sport.
- NHL and NBA history ends in June 2022; nothing here tests recent seasons.
