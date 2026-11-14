"""Loaders for NHL, NBA (sportsbookreview archive, 2011/12 to 2021/22) and UFC.

Known gaps, so nothing downstream pretends otherwise:
  NHL  opening and closing moneylines (both sides); totals carry one side's
       price only, so no complete outcome set -> totals are not tested.
  NBA  opening and closing spread and total LINES but no prices for them, and
       a moneyline at the close only. Spread/total bets are scored at an
       assumed standard -110 on each side. ~7% of closing lines are corrupted
       in the source and are used for nothing except a reference row.
  UFC  one pre-fight moneyline per fighter, capture time not recorded.
  None of the three has line-ups, injuries, minutes, goalies, pace or xG.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SBR_URL = "https://raw.githubusercontent.com/FinnedAI/sportsbookreview-scraper/main/data/{}_archive_10Y.json"
UFC_URL = "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv"

NAME_FIX = {
    "nhl": {"Arizonas": "Coyotes", "Phoenix": "Coyotes", "NY Islanders": "Islanders", "SeattleKraken": "Kraken",
            "Tampa": "Lightning", "Tampa Bay": "Lightning", "WinnipegJets": "Jets", "St.Louis": "Blues"},
    "nba": {"NewJersey": "Nets", "Golden State": "Warriors", "LA Clippers": "Clippers", "Oklahoma City": "Thunder"},
}


def american_to_decimal(a) -> np.ndarray:
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / np.abs(a))
    return np.where((a == 0) | np.isnan(a) | (np.abs(a) < 100), np.nan, d)


def repair_dates(dates: pd.Series, seasons: pd.Series) -> pd.Series:
    """The archive gives month and day reliably but mislabels the year for
    seasons that ran off the usual calendar (2019/20 bubble, 2020/21). Rows are
    in playing order within a season, so walk them in file order and move a
    date one year forward whenever it would otherwise go back in time."""
    out = dates.copy()
    for s in seasons.unique():
        idx = np.flatnonzero((seasons == s).to_numpy())
        cur = pd.Timestamp(int(s), 9, 1)
        for i in idx:
            d = dates.iloc[i]
            if d < cur - pd.Timedelta(days=3):
                d = d + pd.DateOffset(years=1)
            out.iloc[i] = d
            cur = max(cur, d)
    return out


def _download(path: Path, url: str):
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    iso = df["date"].dt.isocalendar()
    df["block"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    return df


def load_sbr(path: str | Path, sport: str) -> pd.DataFrame:
    path = Path(path)
    _download(path, SBR_URL.format(sport))
    raw = pd.DataFrame(json.loads(path.read_text()))
    for c in raw.columns:
        if c in ("home_team", "away_team"):
            raw[c] = raw[c].astype(str).replace(NAME_FIX[sport])
        else:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[(raw.home_team != "0") & (raw.away_team != "0")].dropna(subset=["home_final", "away_final"])
    raw = raw.reset_index(drop=True)
    raw["date"] = repair_dates(pd.to_datetime(raw.date.astype(int).astype(str)), raw.season)
    df = pd.DataFrame({"league": sport.upper(), "season": raw.season.astype(int), "date": raw.date,
                       "home": raw.home_team, "away": raw.away_team,
                       "fh": raw.home_final.astype(int), "fa": raw.away_final.astype(int)})
    if sport == "nhl":
        df["hg"] = raw[["home_1stPeriod", "home_2ndPeriod", "home_3rdPeriod"]].sum(axis=1).astype(int)
        df["ag"] = raw[["away_1stPeriod", "away_2ndPeriod", "away_3rdPeriod"]].sum(axis=1).astype(int)
        for side in ("home", "away"):
            for when in ("open", "close"):
                df[f"o_{side}_{when}"] = american_to_decimal(raw[f"{side}_{when}_ml"])
        for when in ("open", "close"):          # a price pair must be a plausible complete set
            ov = 1 / df[f"o_home_{when}"] + 1 / df[f"o_away_{when}"]
            bad = ~((ov > 1.0) & (ov < 1.12))
            df.loc[bad, [f"o_home_{when}", f"o_away_{when}"]] = np.nan
        df = df[df.fh != df.fa]                 # two archive rows have no winner
    else:
        df["hg"], df["ag"] = df.fh, df.fa
        sp, tot = raw.home_open_spread, raw.open_over_under
        df["spread_open"] = sp.where((sp.abs() <= 30) & (sp == -raw.away_open_spread))
        df["total_open"] = tot.where((tot >= 150) & (tot <= 280))
        spc, totc = raw.home_close_spread, raw.close_over_under
        df["spread_close"] = spc.where((spc.abs() <= 30) & (spc == -raw.away_close_spread))
        df["total_close"] = totc.where((totc >= 150) & (totc <= 280))
        df["o_home_close"] = american_to_decimal(raw.home_close_ml)
        df["o_away_close"] = american_to_decimal(raw.away_close_ml)
        ov = 1 / df.o_home_close + 1 / df.o_away_close
        df.loc[~((ov > 1.0) & (ov < 1.12)), ["o_home_close", "o_away_close"]] = np.nan
    df = _finish(df)
    if sport == "nba":                          # back-to-back: played the previous calendar day
        long = pd.concat([df[["match_id", "date"]].assign(team=df.home), df[["match_id", "date"]].assign(team=df.away)])
        long = long.sort_values(["team", "date"])
        long["b2b"] = (long.groupby("team").date.diff().dt.days == 1).astype(float)
        b = long.set_index(["match_id", "team"]).b2b
        df["b2b_h"] = b.reindex(pd.MultiIndex.from_arrays([df.match_id, df.home])).to_numpy()
        df["b2b_a"] = b.reindex(pd.MultiIndex.from_arrays([df.match_id, df.away])).to_numpy()
    return df


def load_ufc(path: str | Path, neutral_corners: bool = False, seed: int = 0) -> pd.DataFrame:
    """neutral_corners=True swaps the two fighters in a random half of the
    fights, so that which corner a fighter is listed in carries no information.
    In this dataset the red corner wins about two points more often than its
    price implies in almost every year, and backing every red fighter blindly
    nearly breaks even, which a real market would not allow for long. The
    corner label is therefore treated as suspect (it may partly encode the
    result) and the corner-neutral run is the one to trust."""
    path = Path(path)
    _download(path, UFC_URL)
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw.Winner.isin(["Red", "Blue"])]
    df = pd.DataFrame({
        "league": "UFC", "date": pd.to_datetime(raw.date), "home": raw.R_fighter.str.strip(),
        "away": raw.B_fighter.str.strip(), "red_win": (raw.Winner == "Red").astype(int),
        "o_red": american_to_decimal(raw.R_odds), "o_blue": american_to_decimal(raw.B_odds),
        "age_diff": (raw.R_age - raw.B_age) / 5.0, "reach_diff": (raw.R_Reach_cms - raw.B_Reach_cms) / 10.0})
    df[["age_diff", "reach_diff"]] = df[["age_diff", "reach_diff"]].fillna(0.0).clip(-4, 4)
    ov = 1 / df.o_red + 1 / df.o_blue
    df.loc[~((ov > 1.0) & (ov < 1.12)), ["o_red", "o_blue"]] = np.nan
    df["season"] = df.date.dt.year
    df = _finish(df)
    if neutral_corners:
        sw = np.random.default_rng(seed).random(len(df)) < 0.5
        h, a, r, b = df.home.to_numpy().copy(), df.away.to_numpy().copy(), df.o_red.to_numpy().copy(), df.o_blue.to_numpy().copy()
        df["home"], df["away"] = np.where(sw, a, h), np.where(sw, h, a)
        df["o_red"], df["o_blue"] = np.where(sw, b, r), np.where(sw, r, b)
        df["red_win"] = np.where(sw, 1 - df.red_win, df.red_win)
        df["age_diff"] = np.where(sw, -df.age_diff, df.age_diff)
        df["reach_diff"] = np.where(sw, -df.reach_diff, df.reach_diff)
    return df
