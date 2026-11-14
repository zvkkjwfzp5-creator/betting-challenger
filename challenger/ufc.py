"""UFC back-test: fight moneyline at the recorded pre-fight price.

The existing process has no model for UFC (the instructions say "no game sim;
use a structured estimate", which in practice restates the market), so there
is no existing-model row here: the comparison is market vs new model vs the
combination.
"""
from __future__ import annotations

import itertools

import numpy as np

from .data_us import load_ufc
from .market import DEVIG
from .mathx import sigmoid
from .predictive import standard_draws
from .rating import RatingParams, run_rating_filter
from .tuning import grid_search, nested_predictions, select_per_season
from .twoway import evaluate_twoway


def build_grid(c: dict) -> list[RatingParams]:
    g = [RatingParams(tau=t, q_year=q, debut_delta=d) for t, q, d in
         itertools.product(c["tau"], c["q_year"], c["debut_delta"])]
    g += [RatingParams(tau=c["nopool_tau"], q_year=q, debut_delta=0.0) for q in c["q_year"]]
    g += [RatingParams(tau=t, q_year=q, debut_delta=c["debut_delta"][1], use_cov=False)
          for t, q in itertools.product(c["tau"], c["q_year"])]
    return g


def run(cfg: dict, workers=4, verbose=True) -> dict:
    """Runs twice: with corners as recorded, and with corners randomised (see
    load_ufc). Only a result that survives the second run is credible."""
    a = _run_one(cfg, False, "data/cache_ufc", workers, verbose)
    b = _run_one(cfg, True, "data/cache_ufc_neutral", workers, verbose)
    return {"sport": "UFC", "markets": {
        "Fight moneyline, corners randomised (the credible run)": b["res"],
        "Fight moneyline, corners as recorded (red-corner label is suspect)": a["res"]},
        "meta": {"settings_by_season": b["settings"], "matches": b["n"], "outer_draws": b["k"]}}


def _run_one(cfg: dict, neutral: bool, cache: str, workers: int, verbose: bool) -> dict:
    c, st = cfg["ufc"], cfg["ufc"]["stage"]
    df = load_ufc(c["path"], neutral_corners=neutral, seed=cfg["predictive"]["seed"])
    seasons = df.season.to_numpy()
    all_seasons = sorted(int(s) for s in np.unique(seasons) if s >= st["first_pred_season"])
    z = standard_draws(cfg["predictive"]["outer_draws"], cfg["predictive"]["seed"])[:, 0]
    results = grid_search(run_rating_filter, df[["match_id", "date", "home", "away", "red_win", "age_diff", "reach_diff"]],
                          build_grid(c), cache, workers)
    variants = {"full": lambda p: p.tau != c["nopool_tau"] and p.use_cov,
                "static": lambda p: p.tau != c["nopool_tau"] and p.use_cov and p.q_year == 0,
                "nopool": lambda p: p.tau == c["nopool_tau"],
                "nocov": lambda p: not p.use_cov}
    first_score = int(df.season.min()) + 1
    choice = {v: select_per_season(results, seasons, all_seasons, first_score, allowed=f) for v, f in variants.items()}
    preds = {v: nested_predictions(results, choice[v], seasons) for v in variants}

    def draws(pr):
        return sigmoid(pr[:, [0]] + np.sqrt(pr[:, [1]]) * z[None, :])

    pc = draws(preds["full"])
    abl = {"C_static": draws(preds["static"]).mean(axis=1), "C_nopool": draws(preds["nopool"]).mean(axis=1),
           "C_no_age_reach": draws(preds["nocov"]).mean(axis=1), "C_plugin": sigmoid(preds["full"][:, 0])}
    odds = df[["o_red", "o_blue"]].to_numpy()
    okm = ~np.isnan(odds).any(axis=1)
    p_mkt = np.full(len(df), np.nan)
    p_mkt[okm] = DEVIG["multiplicative"](odds[okm])[:, 0]
    y = 1 - df.red_win.to_numpy()
    res = evaluate_twoway("moneyline", df, y, np.zeros(len(df), bool), p_mkt, odds, pc, st, cfg, ablations=abl)
    if verbose:
        print("UFC settings:", {s: choice["full"][s] for s in all_seasons}, flush=True)
    return {"res": res, "settings": choice["full"], "n": int(len(df)), "k": len(z)}
