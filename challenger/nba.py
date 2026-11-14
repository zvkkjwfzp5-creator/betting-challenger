"""NBA back-test: spread and total at the OPENING line, plus win probability.

Margin and total come from the same pair of predicted scores (Gaussian
model), so spread, total and win probability are mutually consistent.

Limits of the archive, reflected in what is claimed:
  * spread and total have lines but no prices -> bets are scored at an assumed
    -110 on each side, and the market baseline for those two markets is 50/50;
  * there is no opening moneyline -> win probability is scored but not bet.
    Its market baseline is the opening spread converted to a win probability
    with a spread-to-win curve fitted on earlier seasons;
  * the closing moneyline appears only as a reference row. It is never an input.
"""
from __future__ import annotations

import itertools

import numpy as np

from .champion import champion_rates
from .data_us import load_sbr
from .gaussian import GaussParams, run_gauss_filter
from .market import DEVIG
from .mathx import ndtr
from .predictive import standard_draws
from .tuning import grid_search, nested_predictions, select_per_season
from .twoway import evaluate_twoway


def build_grid(c: dict) -> list[GaussParams]:
    base = list(itertools.product(c["tau"], c["q_year"], c["season_jump"]))
    g = [GaussParams(tau=t, q_year=q, season_jump=j) for t, q, j in base]
    g += [GaussParams(tau=t, q_year=q, season_jump=j, use_b2b=False) for t, q, j in base]
    g += [GaussParams(tau=c["nopool_tau"], q_year=q, season_jump=j) for q, j in
          itertools.product(c["q_year"], c["season_jump"])]
    return g


def line_probs(mean_draws: np.ndarray, sd: float, line: np.ndarray):
    """P(score quantity > line) and P(push) for an integer-valued quantity
    (margin or total) modelled as normal. Half-point lines cannot push."""
    line = np.asarray(line, float)[:, None]
    whole = np.isclose(line, np.round(line))
    hi = np.where(whole, line + 0.5, line)
    lo = np.where(whole, line - 0.5, line)
    win = 1.0 - ndtr((hi - mean_draws) / sd)
    push = ndtr((hi - mean_draws) / sd) - ndtr((lo - mean_draws) / sd)
    return win, push


def _cond(win, push):
    """Probability of winning given the bet is not a push."""
    return win / np.clip(1.0 - push, 1e-9, None)


def run(cfg: dict, cache="data/cache_nba", workers=4, verbose=True) -> dict:
    c, st = cfg["nba"], cfg["nba"]["stage"]
    df = load_sbr(c["path"], "nba")
    seasons = df.season.to_numpy()
    all_seasons = sorted(int(s) for s in np.unique(seasons) if s >= st["first_pred_season"])
    z = standard_draws(cfg["predictive"]["outer_draws"], cfg["predictive"]["seed"])
    results = grid_search(run_gauss_filter, df[["match_id", "season", "date", "home", "away", "hg", "ag", "b2b_h", "b2b_a"]],
                          build_grid(c), cache, workers)
    np_tau = c["nopool_tau"]
    variants = {"full": lambda p: p.tau != np_tau and p.use_b2b,
                "static": lambda p: p.tau != np_tau and p.use_b2b and p.q_year == 0 and p.season_jump == 0,
                "nopool": lambda p: p.tau == np_tau,
                "nob2b": lambda p: p.tau != np_tau and not p.use_b2b}
    first_score = int(df.season.min()) + 1
    choice = {v: select_per_season(results, seasons, all_seasons, first_score, allowed=f) for v, f in variants.items()}
    preds = {v: np.asarray(nested_predictions(results, choice[v], seasons)) for v in variants}

    margin = (df.hg - df.ag).to_numpy(float)
    total = (df.hg + df.ag).to_numpy(float)
    sp = df.spread_open.to_numpy()          # home handicap: home covers if margin > -sp
    tl = df.total_open.to_numpy()
    n = len(df)
    names = ("spread", "total", "win")
    pc = {k: np.full((n, len(z)), np.nan) for k in names}
    abl = {k: {a: np.full(n, np.nan) for a in ("C_static", "C_nopool", "C_no_back_to_back", "C_plugin")} for k in names}
    p_mkt_win = np.full(n, np.nan)
    settings = {}

    def price(pr, sd_m, sd_t, zz):
        l11 = np.sqrt(np.clip(pr[:, 2], 1e-9, None))
        l21 = pr[:, 4] / l11
        l22 = np.sqrt(np.clip(pr[:, 3] - l21 ** 2, 1e-9, None))
        md = pr[:, [0]] + l11[:, None] * zz[None, :, 0]
        td = pr[:, [1]] + l21[:, None] * zz[None, :, 0] + l22[:, None] * zz[None, :, 1]
        return (_cond(*line_probs(md, sd_m, -sp_s)), _cond(*line_probs(td, sd_t, tl_s)), 1.0 - ndtr((0.0 - md) / sd_m))

    for s in all_seasons:
        past = (seasons < s) & (seasons >= st["first_pred_season"])
        pf = preds["full"]
        sd_m, sd_t, sd_curve = 12.5, 18.0, 12.0
        if past.sum() > 500:
            # game noise = spread of earlier one-step-ahead errors minus the parameter variance already in them
            sd_m = float(np.sqrt(max(np.var(margin[past] - pf[past, 0]) - pf[past, 2].mean(), 25.0)))
            sd_t = float(np.sqrt(max(np.var(total[past] - pf[past, 1]) - pf[past, 3].mean(), 25.0)))
            ok = past & ~np.isnan(sp)
            cand = np.arange(9.0, 16.01, 0.25)        # spread-to-win curve for the market baseline
            won = margin[ok] > 0
            loss = [-(np.log(np.clip(np.where(won, ndtr(-sp[ok] / q), 1 - ndtr(-sp[ok] / q)), 1e-9, None))).mean() for q in cand]
            sd_curve = float(cand[int(np.argmin(loss))])
        m = seasons == s
        sp_s, tl_s = sp[m], tl[m]
        out = price(pf[m], sd_m, sd_t, z)
        for k, v in zip(names, out):
            pc[k][m] = v
        for a, pr, zz in (("C_static", preds["static"][m], z), ("C_nopool", preds["nopool"][m], z),
                          ("C_no_back_to_back", preds["nob2b"][m], z), ("C_plugin", pf[m], np.zeros((1, 2)))):
            for k, v in zip(names, price(pr, sd_m, sd_t, zz)):
                abl[k][a][m] = v.mean(axis=1)
        p_mkt_win[m] = ndtr(-sp_s / sd_curve)
        settings[s] = {"filter": choice["full"][s], "margin_sd": round(sd_m, 2), "total_sd": round(sd_t, 2),
                       "spread_to_win_sd": sd_curve}
        if verbose:
            print(f"NBA {s}: n={m.sum()} {settings[s]}", flush=True)

    # existing model: points from last-10 + season scoring rates; margin SD 12, total SD 14 as instructed
    rates = champion_rates(df, cfg["champion"]["form_games"], clip=(70.0, 150.0), default=(100.0, 97.0))
    bm = (rates.lam_h - rates.lam_a).to_numpy()[:, None]
    bt = (rates.lam_h + rates.lam_a).to_numpy()[:, None]
    b_spread = _cond(*line_probs(bm, 12.0, -sp))[:, 0]
    b_total = _cond(*line_probs(bt, 14.0, tl))[:, 0]
    b_win = (1.0 - ndtr(-bm / 12.0))[:, 0]

    price_ = c["assumed_price"]
    odds = np.full((n, 2), price_)
    half = np.full(n, 0.5)
    oc = df[["o_home_close", "o_away_close"]].to_numpy()
    okc = ~np.isnan(oc).any(axis=1)
    p_close = np.full(n, np.nan)
    p_close[okc] = DEVIG["multiplicative"](oc[okc])[:, 0]

    out = {}
    cover, push_s = margin + sp > 0, margin + sp == 0
    out["Spread at the opening line (price assumed -110)"] = evaluate_twoway(
        "spread", df, (~cover).astype(int), push_s | np.isnan(sp), np.where(np.isnan(sp), np.nan, half),
        np.where(np.isnan(sp)[:, None], np.nan, odds), pc["spread"], st, cfg, p_b_raw=b_spread, ablations=abl["spread"])
    over, push_t = total > tl, total == tl
    out["Total at the opening line (price assumed -110)"] = evaluate_twoway(
        "total", df, (~over).astype(int), push_t | np.isnan(tl), np.where(np.isnan(tl), np.nan, half),
        np.where(np.isnan(tl)[:, None], np.nan, odds), pc["total"], st, cfg, p_b_raw=b_total, ablations=abl["total"])
    out["Win probability (no opening moneyline in the data: scored, not bet)"] = evaluate_twoway(
        "win", df, (margin < 0).astype(int), np.isnan(sp), p_mkt_win, odds, pc["win"], st, cfg, p_b_raw=b_win,
        ablations=abl["win"], reference={"Z_closing_price_reference": p_close}, bettable=False)
    return {"sport": "NBA", "markets": out, "meta": {"settings_by_season": settings, "matches": int(n), "outer_draws": len(z)}}
