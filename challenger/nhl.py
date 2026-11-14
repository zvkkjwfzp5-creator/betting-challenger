"""NHL back-test: moneyline (overtime and shoot-out included) at the opening price.

Model: the dynamic hierarchical Poisson model on REGULATION goals, with extra
mass on level scores (teams protect a tie late), then an overtime step whose
home-win probability depends on the strength gap and is fitted on earlier
seasons. Totals and the puck line are priced by the same score matrix but are
not tested: the archive has one side's total price only, and puck-line prices
at the close only.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .champion import champion_rates
from .data_us import load_sbr
from .dynamic import FilterParams, PRED_COLS, run_filter
from .market import DEVIG
from .mathx import fit_logistic, sigmoid
from .predictive import rate_draws, standard_draws
from .scoregrid import ScoreGrid
from .tuning import goal_loglik, grid_search, nested_predictions, select_per_season
from .twoway import evaluate_twoway


def _runner(df, params):
    preds = run_filter(df, params)
    return preds.to_numpy(), goal_loglik(preds, df.hg.to_numpy(), df.ag.to_numpy())


def build_grid(c: dict) -> list[FilterParams]:
    kw = dict(mu0=c["mu0"], h0=c["h0"], q_league_year=c["q_league_year"])
    g = [FilterParams(tau=t, q_year=q, season_jump=j, promo_delta=d, **kw)
         for t, q, j, d in itertools.product(c["tau"], c["q_year"], c["season_jump"], c["promo_delta"])]
    g += [FilterParams(tau=c["nopool_tau"], q_year=q, season_jump=j, promo_delta=0.0, **kw)
          for q, j in itertools.product(c["q_year"], c["season_jump"])]
    return g


def home_win_draws(preds: np.ndarray, z, grid, diag, ot, chunk=1000):
    """Per-draw P(home wins incl. OT/SO): regulation win + tie * P(home wins OT)."""
    out = np.empty((len(preds), len(z)))
    for i in range(0, len(preds), chunk):
        a = preds[i:i + chunk]
        lh, la = rate_draws(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], z)
        p3 = grid.p_1x2(grid.pmf(lh, la, 0.0, diag))
        out[i:i + chunk] = p3[..., 0] + p3[..., 1] * sigmoid(ot[0] + ot[1] * (np.log(lh) - np.log(la)))
    return out


def run(cfg: dict, cache="data/cache_nhl", workers=4, verbose=True) -> dict:
    c, st = cfg["nhl"], cfg["nhl"]["stage"]
    df = load_sbr(c["path"], "nhl")
    seasons = df.season.to_numpy()
    all_seasons = sorted(int(s) for s in np.unique(seasons) if s >= st["first_pred_season"])
    grid = ScoreGrid(12)
    z = standard_draws(cfg["predictive"]["outer_draws"], cfg["predictive"]["seed"])
    results = grid_search(_runner, df[["match_id", "league", "season", "date", "home", "away", "hg", "ag"]],
                          build_grid(c), cache, workers)
    variants = {"full": lambda p: p.tau != c["nopool_tau"],
                "static": lambda p: p.tau != c["nopool_tau"] and p.q_year == 0 and p.season_jump == 0,
                "nopool": lambda p: p.tau == c["nopool_tau"]}
    first_score = int(df.season.min()) + 1
    choice = {v: select_per_season(results, seasons, all_seasons, first_score, allowed=f) for v, f in variants.items()}
    preds = {v: nested_predictions(results, choice[v], seasons).to_numpy() for v in variants}

    hg, ag = np.minimum(df.hg.to_numpy(), 12), np.minimum(df.ag.to_numpy(), 12)
    tie = (df.hg == df.ag).to_numpy()
    home_win = (df.fh > df.fa).to_numpy()
    pf = preds["full"]
    lam_h, lam_a = np.exp(pf[:, 0] + 0.5 * pf[:, 2]), np.exp(pf[:, 1] + 0.5 * pf[:, 3])
    n = len(df)
    pc = np.full((n, len(z)), np.nan)
    abl = {k: np.full(n, np.nan) for k in ("C_static", "C_nopool", "C_no_tie_inflation", "C_plugin")}
    settings = {}
    for s in all_seasons:
        past = (seasons < s) & (seasons >= st["first_pred_season"])
        diag = 1.0
        if past.sum() > 500:                         # tie inflation: best regulation-score likelihood so far
            score = {d: np.log(grid.pmf(lam_h[past], lam_a[past], 0.0, d)[np.arange(past.sum()), hg[past], ag[past]]).sum()
                     for d in c["tie_inflation_grid"]}
            diag = max(score, key=score.get)
        ot = np.zeros(2)
        pt = past & tie
        if pt.sum() > 200:                           # overtime step, fitted on earlier tied games
            X = np.column_stack([np.ones(pt.sum()), pf[pt, 0] - pf[pt, 1]])
            ot = fit_logistic(X, home_win[pt].astype(float), ridge=1.0)
        m = seasons == s
        pc[m] = home_win_draws(pf[m], z, grid, diag, ot)
        abl["C_static"][m] = home_win_draws(preds["static"][m], z, grid, diag, ot).mean(axis=1)
        abl["C_nopool"][m] = home_win_draws(preds["nopool"][m], z, grid, diag, ot).mean(axis=1)
        abl["C_no_tie_inflation"][m] = home_win_draws(pf[m], z, grid, 1.0, ot).mean(axis=1)
        abl["C_plugin"][m] = home_win_draws(pf[m] * np.array([1, 1, 0, 0, 0]), np.zeros((1, 2)), grid, diag, ot)[:, 0]
        settings[s] = {"filter": choice["full"][s], "tie_inflation": diag, "ot": np.round(ot, 3).tolist()}
        if verbose:
            print(f"NHL {s}: n={m.sum()} {settings[s]}", flush=True)

    odds = df[["o_home_open", "o_away_open"]].to_numpy()
    okm = ~np.isnan(odds).any(axis=1)
    p_mkt = np.full(n, np.nan)
    p_mkt[okm] = DEVIG["multiplicative"](odds[okm])[:, 0]
    oc = df[["o_home_close", "o_away_close"]].to_numpy()
    okc = ~np.isnan(oc).any(axis=1)
    p_close = np.full(n, np.nan)
    p_close[okc] = DEVIG["multiplicative"](oc[okc])[:, 0]

    # existing model: independent Poisson from last-10 + season goal rates, overtime as a coin flip
    rates = champion_rates(df, cfg["champion"]["form_games"], clip=(1.0, 6.0), default=(2.9, 2.6))
    p3 = grid.p_1x2(grid.pmf(rates.lam_h.to_numpy(), rates.lam_a.to_numpy()))
    p_b = p3[:, 0] + 0.5 * p3[:, 1]

    y = (~home_win).astype(int)
    res = evaluate_twoway("moneyline", df, y, np.zeros(n, bool), p_mkt, odds, pc, st, cfg, p_b_raw=p_b,
                          ablations=abl, reference={"Z_closing_price_reference": p_close})
    return {"sport": "NHL", "markets": {"Moneyline at the opening price (OT included)": res},
            "meta": {"settings_by_season": settings, "matches": int(n), "outer_draws": len(z)}}
