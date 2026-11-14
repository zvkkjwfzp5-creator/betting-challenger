"""The existing model ("champion"), written down as code so it can be tested.

The existing model lives in the scheduled-task instructions, not in a code
base. For soccer they say: simulate each team's goals as independent Poisson
from expected goals built on recent form (last 10) and season ratings with
home field, then blend 70% no-vig market / 30% model, and flag (do not trust)
games where model and market disagree by 7+ points.

No xG history is available, so goals stand in for it. That choice, and the
equal weighting of last-10 form and season-to-date rates, are this file's
reading of the instructions and are stated as such in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoregrid import ScoreGrid


def champion_rates(df: pd.DataFrame, form_games: int = 10, clip=(0.2, 4.5), default=(1.5, 1.1)) -> pd.DataFrame:
    """Pre-match expected goals for every match, using earlier matches only."""
    long = pd.concat([
        df[["match_id", "date", "season", "league"]].assign(team=df.home, gf=df.hg, ga=df.ag),
        df[["match_id", "date", "season", "league"]].assign(team=df.away, gf=df.ag, ga=df.hg),
    ]).sort_values(["team", "date"], kind="stable")
    g = long.groupby("team")
    for c in ("gf", "ga"):
        long[f"{c}_l10"] = g[c].transform(lambda s: s.shift(1).rolling(form_games, min_periods=3).mean())
    gs = long.groupby(["team", "season"])
    for c in ("gf", "ga"):
        long[f"{c}_std"] = gs[c].transform(lambda s: s.shift(1).expanding().mean())
    for c in ("gf", "ga"):
        both = 0.5 * long[f"{c}_l10"] + 0.5 * long[f"{c}_std"]
        long[f"{c}_rate"] = both.fillna(long[f"{c}_l10"])
    rate = long.set_index("match_id")

    # League scoring level over the previous 365 days, strictly before the match date.
    daily = df.groupby(["league", "date"]).agg(h=("hg", "sum"), a=("ag", "sum"), n=("hg", "size")).reset_index()
    parts = []
    for lg, d in daily.groupby("league"):
        r = d.set_index("date")[["h", "a", "n"]].rolling("365D", closed="left").sum()
        parts.append(pd.DataFrame({"league": lg, "date": r.index,
                                   "avg_h": (r.h / r.n).to_numpy(), "avg_a": (r.a / r.n).to_numpy()}))
    lg_avg = pd.concat(parts)
    out = df[["match_id", "league", "date", "home", "away"]].merge(lg_avg, on=["league", "date"], how="left")
    out["avg_h"] = out["avg_h"].fillna(default[0])
    out["avg_a"] = out["avg_a"].fillna(default[1])
    avg_team = 0.5 * (out.avg_h + out.avg_a)

    home = rate[rate.team.to_numpy() == df.set_index("match_id").home.reindex(rate.index).to_numpy()]
    away = rate[rate.team.to_numpy() == df.set_index("match_id").away.reindex(rate.index).to_numpy()]
    hm = home.reindex(out.match_id)
    aw = away.reindex(out.match_id)
    att_h = (hm.gf_rate.to_numpy() / avg_team).fillna(1.0)
    def_h = (hm.ga_rate.to_numpy() / avg_team).fillna(1.0)
    att_a = (aw.gf_rate.to_numpy() / avg_team).fillna(1.0)
    def_a = (aw.ga_rate.to_numpy() / avg_team).fillna(1.0)
    out["lam_h"] = np.clip(out.avg_h * att_h * def_a, *clip)
    out["lam_a"] = np.clip(out.avg_a * att_a * def_h, *clip)
    return out[["match_id", "lam_h", "lam_a"]]


def champion_probs(rates: pd.DataFrame, grid: ScoreGrid):
    """Independent-Poisson probabilities (the exact limit of the 10,000-game sim)."""
    M = grid.pmf(rates.lam_h.to_numpy(), rates.lam_a.to_numpy(), 0.0)
    return grid.p_1x2(M), grid.p_over(M, 2.5)


def blend(p_market: np.ndarray, p_model: np.ndarray, market_weight: float = 0.70) -> np.ndarray:
    """The existing linear blend: 70% no-vig market + 30% model."""
    return market_weight * p_market + (1.0 - market_weight) * p_model
