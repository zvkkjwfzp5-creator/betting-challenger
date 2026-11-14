"""Load match results and the Bet365 pre-match prices.

Source: football-data.co.uk results and odds, as mirrored in the public
xgabora/Club-Football-Match-Data-2000-2025 dataset.

What the data does NOT contain (so the model does not pretend to use it):
line-ups, injuries, minutes, xG, travel, and the capture time of each price.
Rest days cannot be derived reliably because cup and European games are absent.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

COLS = {
    "Division": "league", "MatchDate": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHome": "hg", "FTAway": "ag",
    "OddHome": "o_h", "OddDraw": "o_d", "OddAway": "o_a",
    "Over25": "o_over", "Under25": "o_under",
    "HandiSize": "ah_line", "HandiHome": "o_ah_h", "HandiAway": "o_ah_a",
}


def season_of(date: pd.Series) -> pd.Series:
    """Season label = calendar year in which the season starts (July cut-over)."""
    return date.dt.year - (date.dt.month < 7).astype(int)


def snap_quarter(x: pd.Series) -> pd.Series:
    """The source rounds Asian lines to one decimal (-0.3 means -0.25)."""
    return (x * 4).round() / 4


def load_matches(path: str | Path, leagues: list[str], url: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if url is None:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    raw = pd.read_csv(path, usecols=list(COLS), low_memory=False).rename(columns=COLS)
    df = raw[raw.league.isin(leagues)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["hg", "ag", "home", "away", "date"])
    df[["hg", "ag"]] = df[["hg", "ag"]].astype(int)
    df["season"] = season_of(df["date"])
    df["ah_line"] = snap_quarter(df["ah_line"])
    # A price must be a real decimal price; anything else is treated as missing.
    for c in ["o_h", "o_d", "o_a", "o_over", "o_under", "o_ah_h", "o_ah_a"]:
        df.loc[~(df[c] > 1.0), c] = np.nan
    df = df.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    # Block id for resampling: matches in the same ISO week are resampled together.
    iso = df["date"].dt.isocalendar()
    df["block"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    return df
