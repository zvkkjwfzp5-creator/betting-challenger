"""Chronological tuning of the dynamic model.

Each candidate setting is run once, causally, over the whole history. That
yields a one-step-ahead predictive log-likelihood for every match. To predict
season s we pick the setting with the best total over seasons before s only.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from math import lgamma

from .dynamic import FilterParams, run_filter, PRED_COLS

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(7)
_GH_W = _GH_W / _GH_W.sum()
_LF = np.array([lgamma(k + 1.0) for k in range(40)])


def goal_loglik(preds: pd.DataFrame, hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    """Log predictive probability of the observed score, integrating over the
    posterior of the two log rates (7x7 Gauss-Hermite)."""
    from .predictive import rate_draws
    z = np.array([(a, b) for a in _GH_X for b in _GH_X])
    w = np.array([wa * wb for wa in _GH_W for wb in _GH_W])
    lh, la = rate_draws(*[preds[c].to_numpy() for c in PRED_COLS], z)
    hg = np.asarray(hg)[:, None]
    ag = np.asarray(ag)[:, None]
    lp = hg * np.log(lh) - lh - _LF[hg] + ag * np.log(la) - la - _LF[ag]
    mx = lp.max(axis=1, keepdims=True)
    return (mx[:, 0] + np.log((np.exp(lp - mx) * w).sum(axis=1)))


def build_grid(cfg: dict) -> list[FilterParams]:
    g = cfg["filter_grid"]
    grid = [FilterParams(tau=t, q_year=q, season_jump=j, promo_delta=d)
            for t, q, j, d in itertools.product(g["tau"], g["q_year"], g["season_jump"], g["promo_delta"])]
    rw = cfg.get("filter_grid_rw")
    if rw:
        grid += [FilterParams(tau=t, q_year=q, season_jump=j, promo_delta=d, revert=False)
                 for t, q, j, d in itertools.product(rw["tau"], rw["q_year"], rw["season_jump"], rw["promo_delta"])]
    # ablation: no pooling (very wide prior), same dynamics choices
    grid += [FilterParams(tau=g["nopool_tau"], q_year=q, season_jump=j, promo_delta=g["promo_delta"][0])
             for q, j in itertools.product(g["q_year"], g["season_jump"])]
    return grid


def _one(args):
    df, params, cache = args
    f = Path(cache) / (params.key().replace("|", "_") + ".npz")
    if f.exists():
        z = np.load(f)
        return params, z["preds"], z["ll"]
    preds = run_filter(df, params)
    ll = goal_loglik(preds, df.hg.to_numpy(), df.ag.to_numpy())
    np.savez_compressed(f, preds=preds.to_numpy(), ll=ll)
    return params, preds.to_numpy(), ll


def run_grid(df: pd.DataFrame, grid: list[FilterParams], cache: str, workers: int = 4):
    Path(cache).mkdir(parents=True, exist_ok=True)
    slim = df[["match_id", "league", "season", "date", "home", "away", "hg", "ag"]]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one, [(slim, p, cache) for p in grid]))
    return {p.key(): (p, preds, ll) for p, preds, ll in res}


def select_per_season(results: dict, seasons: np.ndarray, eval_seasons: list[int],
                      first_score_season: int, allowed=lambda p: True) -> dict[int, str]:
    """For each season s: the allowed setting with the best log-likelihood over
    seasons in [first_score_season, s)."""
    keys = [k for k, (p, _, _) in results.items() if allowed(p)]
    per_season = {k: pd.Series(results[k][2]).groupby(seasons).sum() for k in keys}
    choice = {}
    for s in eval_seasons:
        tot = {k: v[(v.index >= first_score_season) & (v.index < s)].sum() for k, v in per_season.items()}
        choice[s] = max(tot, key=tot.get)
    return choice


def nested_predictions(results: dict, choice: dict[int, str], seasons: np.ndarray) -> pd.DataFrame:
    """Stitch together, season by season, the predictions of the setting that
    was chosen using earlier seasons only."""
    width = next(iter(results.values()))[1].shape[1]
    out = np.full((len(seasons), width), np.nan)
    for s, k in choice.items():
        m = seasons == s
        out[m] = results[k][1][m]
    return pd.DataFrame(out, columns=PRED_COLS) if width == 5 else out


def _one_generic(args):
    runner, df, params, cache = args
    f = Path(cache) / (params.key().replace("|", "_") + ".npz")
    if f.exists():
        z = np.load(f)
        return params, z["preds"], z["ll"]
    preds, ll = runner(df, params)
    np.savez_compressed(f, preds=preds, ll=ll)
    return params, preds, ll


def grid_search(runner, df: pd.DataFrame, grid: list, cache: str, workers: int = 4) -> dict:
    """Run `runner(df, params) -> (predictions, one-step-ahead log-likelihood)`
    once per setting, in parallel, with an on-disk cache."""
    Path(cache).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one_generic, [(runner, df, p, cache) for p in grid]))
    return {p.key(): (p, preds, ll) for p, preds, ll in res}
