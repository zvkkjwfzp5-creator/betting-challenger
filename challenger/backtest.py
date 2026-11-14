"""Nested walk-forward comparison: market (A), existing model (B), new model (C),
regularised combination (D), plus ablations of C.

Everything used to predict season s (filter setting, Dixon-Coles rho, margin
removal method, combination weights and ridge, selection thresholds) is chosen
on seasons before s. Matches on one date are predicted before any of that
date's results are used. Intervals resample whole match weeks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from .champion import blend, champion_probs, champion_rates
from .combine import fit_pool_walk_forward
from .data import load_matches
from .dynamic import PRED_COLS
from .market import DEVIG
from .predictive import rate_draws, standard_draws
from .scoregrid import ScoreGrid
from .selection import apply_rule, choose_thresholds, ev_summary
from .tuning import build_grid, nested_predictions, run_grid, select_per_season


def _market_probs(df, cols, first_season):
    """Margin-free market probabilities; the removal method for season s is the
    one with the best log loss on earlier seasons."""
    odds = df[cols].to_numpy()
    ok = ~np.isnan(odds).any(axis=1)
    y = df["_y"].to_numpy()
    seasons = df.season.to_numpy()
    by = {}
    for name, fn in DEVIG.items():
        p = np.full(odds.shape, np.nan)
        p[ok] = fn(odds[ok])
        by[name] = p
    out = np.full(odds.shape, np.nan)
    chosen = {}
    for s in np.unique(seasons):
        past = ok & (seasons < s) & (seasons >= first_season)
        name = "multiplicative"
        if past.sum() > 500:
            name = min(by, key=lambda k: ev.log_loss(by[k][past], y[past]).sum())
        out[seasons == s] = by[name][seasons == s]
        chosen[int(s)] = name
    return out, ok, chosen


def _choose_rho(preds, df, grid, rhos, first_season):
    lam_h = np.exp(preds.eta_h + 0.5 * preds.s_hh).to_numpy()
    lam_a = np.exp(preds.eta_a + 0.5 * preds.s_aa).to_numpy()
    hg = np.minimum(df.hg.to_numpy(), grid.G)
    ag = np.minimum(df.ag.to_numpy(), grid.G)
    seasons = df.season.to_numpy()
    ok = ~np.isnan(lam_h)
    table = {}
    for r in rhos:
        ll = np.full(len(df), np.nan)
        M = grid.pmf(lam_h[ok], lam_a[ok], r)
        ll[ok] = np.log(M[np.arange(ok.sum()), hg[ok], ag[ok]])
        table[r] = pd.Series(ll).groupby(seasons).sum()
    choice = {}
    for s in np.unique(seasons):
        tot = {r: v[(v.index >= first_season) & (v.index < s)].sum() for r, v in table.items()}
        choice[int(s)] = max(tot, key=tot.get) if any(t != 0 for t in tot.values()) else 0.0
    return choice


def _draw_probs(preds, rho, z, grid, chunk=400):
    """Per-draw 1X2 and over-2.5 probabilities: (n, k, 3) and (n, k)."""
    n = len(preds)
    p1 = np.empty((n, len(z), 3))
    po = np.empty((n, len(z)))
    arr = preds[PRED_COLS].to_numpy()
    for i in range(0, n, chunk):
        a = arr[i:i + chunk]
        lh, la = rate_draws(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], z)
        M = grid.pmf(lh, la, rho)
        p1[i:i + chunk] = grid.p_1x2(M)
        po[i:i + chunk] = grid.p_over(M, 2.5)
    return p1, po


def _candidates(d, model, p1_draws, po_draws):
    """Long table of selections with expected-return summaries across draws."""
    rows = []
    y1, yo = d["_y"].to_numpy(), d["_yo"].to_numpy()
    for market, sels, cols, pd_, y in (
        ("1x2", ["home", "draw", "away"], ["o_h", "o_d", "o_a"], p1_draws, y1),
        ("ou25", ["over", "under"], ["o_over", "o_under"],
         np.stack([po_draws, 1 - po_draws], axis=-1), yo),
    ):
        for j, (sel, col) in enumerate(zip(sels, cols)):
            odds = d[col].to_numpy()
            s = ev_summary(pd_[:, :, j] if pd_.ndim == 3 else pd_[:, None, j], np.nan_to_num(odds, nan=1.0))
            won = (y == j)
            rows.append(pd.DataFrame({
                "match_id": d.match_id.to_numpy(), "season": d.season.to_numpy(), "block": d.block.to_numpy(),
                "model": model, "market": market, "sel": sel, "odds": odds, **s,
                "won": won, "ret": np.where(won, odds - 1.0, -1.0)}))
    out = pd.concat(rows, ignore_index=True)
    return out[out.odds.notna()]


def run_backtest(cfg: dict, cache: str = "data/cache", workers: int = 4, verbose: bool = True) -> dict:
    wf = cfg["walk_forward"]
    df = load_matches(cfg["data"]["path"], cfg["data"]["leagues"], cfg["data"].get("url"))
    df["_y"] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    df["_yo"] = np.where(df.hg + df.ag > 2.5, 0, 1)
    seasons = df.season.to_numpy()
    all_seasons = [int(s) for s in np.unique(seasons) if s >= wf["first_pred_season"]]
    grid = ScoreGrid(cfg["score"]["max_goals"])
    z = standard_draws(cfg["predictive"]["outer_draws"], cfg["predictive"]["seed"])

    # ---- C: dynamic hierarchical model, setting chosen on earlier seasons ----
    results = run_grid(df, build_grid(cfg), cache, workers)
    nopool = cfg["filter_grid"]["nopool_tau"]
    variants = {
        "full": lambda p: p.tau != nopool,
        "static": lambda p: p.tau != nopool and p.q_year == 0 and p.season_jump == 0,
        "nopool": lambda p: p.tau == nopool,
    }
    first_score = df.season.min() + 2
    choice = {v: select_per_season(results, seasons, all_seasons, first_score, allowed=f)
              for v, f in variants.items()}
    preds = {v: nested_predictions(results, choice[v], seasons) for v in variants}
    rho = _choose_rho(preds["full"], df, grid, cfg["score"]["dc_rho_grid"], wf["first_pred_season"])

    # ---- A: market baseline ----
    pm1, ok1, dv1 = _market_probs(df, ["o_h", "o_d", "o_a"], wf["first_pred_season"])
    df2 = df.assign(_y=df["_yo"])
    pmo, oko, dvo = _market_probs(df2, ["o_over", "o_under"], wf["first_pred_season"])

    # ---- B: existing model ----
    ch = cfg["champion"]
    rates = champion_rates(df, ch["form_games"])
    b1_raw, bo_raw = champion_probs(rates, grid)
    bo_raw = np.stack([bo_raw, 1 - bo_raw], axis=1)
    b1 = blend(pm1, b1_raw, ch["market_weight"])
    bo = blend(pmo, bo_raw, ch["market_weight"])

    n = len(df)
    P = {k: np.full((n, 3), np.nan) for k in ("C", "D", "C_static", "C_nopool", "C_nodc", "C_plugin")}
    O = {k: np.full((n, 2), np.nan) for k in P}
    cands, pools = [], {}
    y1, yo = df["_y"].to_numpy(), df["_yo"].to_numpy()

    for s in all_seasons:
        m = seasons == s
        d = df[m]
        r = rho[s]
        p1d, pod = _draw_probs(preds["full"][m], r, z, grid)
        P["C"][m] = p1d.mean(axis=1)
        O["C"][m] = np.stack([pod.mean(axis=1), 1 - pod.mean(axis=1)], axis=1)
        for name, pr, rr in (("C_static", preds["static"][m], r), ("C_nopool", preds["nopool"][m], r),
                             ("C_nodc", preds["full"][m], 0.0)):
            a1, ao = _draw_probs(pr, rr, z, grid)
            P[name][m] = a1.mean(axis=1)
            O[name][m] = np.stack([ao.mean(axis=1), 1 - ao.mean(axis=1)], axis=1)
        pf = preds["full"][m]
        Mp = grid.pmf(np.exp(pf.eta_h.to_numpy()), np.exp(pf.eta_a.to_numpy()), r)
        P["C_plugin"][m] = grid.p_1x2(Mp)
        po_pl = grid.p_over(Mp, 2.5)
        O["C_plugin"][m] = np.stack([po_pl, 1 - po_pl], axis=1)
        cands.append(_candidates(d, "C", p1d, pod))

        # ---- D: combination fitted on earlier seasons only ----
        if s >= wf["first_combo_season"]:
            past1 = ok1 & (seasons < s) & (seasons >= wf["first_pred_season"])
            pasto = oko & (seasons < s) & (seasons >= wf["first_pred_season"])
            pool1 = fit_pool_walk_forward(pm1[past1], P["C"][past1], y1[past1], seasons[past1], s, cfg["combine"]["ridge"])
            poolo = fit_pool_walk_forward(pmo[pasto], O["C"][pasto], yo[pasto], seasons[pasto], s, cfg["combine"]["ridge"])
            pools[s] = {"1x2": {"theta": pool1.theta.round(4).tolist(), "ridge": pool1.ridge},
                        "ou25": {"theta": poolo.theta.round(4).tolist(), "ridge": poolo.ridge}}
            d1 = np.full(p1d.shape, np.nan)
            mk1 = ok1[m]
            d1[mk1] = pool1.predict(pm1[m][mk1], p1d[mk1])
            pod2 = np.stack([pod, 1 - pod], axis=-1)
            do = np.full(pod2.shape, np.nan)
            mko = oko[m]
            do[mko] = poolo.predict(pmo[m][mko], pod2[mko])
            P["D"][m] = d1.mean(axis=1)
            O["D"][m] = do.mean(axis=1)
            cands.append(_candidates(d, "D", d1, do[:, :, 0]))
        if verbose:
            print(f"season {s}: n={m.sum()} setting={choice['full'][s]} rho={r}", flush=True)

    cands = pd.concat(cands, ignore_index=True)

    # ---- B selections: the existing rule (blended EV > 0, not flagged) ----
    bc = []
    for market, sels, cols, pb, praw, pmk, y in (
        ("1x2", ["home", "draw", "away"], ["o_h", "o_d", "o_a"], b1, b1_raw, pm1, y1),
        ("ou25", ["over", "under"], ["o_over", "o_under"], bo, bo_raw, pmo, yo)):
        for j, (sel, col) in enumerate(zip(sels, cols)):
            odds = df[col].to_numpy()
            bc.append(pd.DataFrame({
                "match_id": df.match_id, "season": seasons, "block": df.block, "model": "B", "market": market,
                "sel": sel, "odds": odds, "p": pb[:, j], "ev": pb[:, j] * odds - 1.0,
                "gap": np.abs(praw[:, j] - pmk[:, j]), "won": y == j, "ret": np.where(y == j, odds - 1.0, -1.0)}))
    bc = pd.concat(bc, ignore_index=True).dropna(subset=["odds", "ev"])
    b_bets = bc[(bc.ev > 0) & (bc.gap < ch["flag_gap"])]
    b_bets = b_bets.loc[b_bets.groupby(["match_id", "market"]).ev.idxmax()]

    # ---- thresholds for C and D, chosen on earlier out-of-sample seasons ----
    sel = cfg["selection"]
    eval_seasons = [s for s in all_seasons if s >= wf["first_eval_season"]]
    bets, thr_log = {"C": [], "D": []}, {"C": {}, "D": {}}
    for model in ("C", "D"):
        cm = cands[cands.model == model]
        for s in eval_seasons:
            hist = cm[(cm.season < s) & (cm.season >= s - wf["threshold_lookback"])]
            thr, lcb, tried = choose_thresholds(hist, sel["ev_min"], sel["p_pos_min"], sel["min_bets"],
                                                sel["lcb_quantile"])
            thr_log[model][s] = {"thresholds": thr, "earlier_roi_lower_bound": round(lcb, 4), "rules_tried": tried}
            if thr is not None:
                bets[model].append(apply_rule(cm[cm.season == s], *thr))
        bets[model] = pd.concat(bets[model]) if bets[model] else cm.iloc[0:0]

    # ---- scores on the evaluation window ----
    bs = cfg["bootstrap"]
    em = np.isin(seasons, eval_seasons)
    blocks = df.block.to_numpy()
    models1 = {"A_market": pm1, "B_existing_raw": b1_raw, "B_existing_blend": b1, "C_new": P["C"], "D_combo": P["D"]}
    modelso = {"A_market": pmo, "B_existing_raw": bo_raw, "B_existing_blend": bo, "C_new": O["C"], "D_combo": O["D"]}
    out = {"markets": {}, "ablations": {}, "returns": {}}
    for mname, models, ok, y in (("1x2", models1, ok1, y1), ("ou25", modelso, oko, yo)):
        msk = em & ok & ~np.isnan(models["D_combo"]).any(axis=1)
        res = {"n_matches": int(msk.sum())}
        ll = {k: ev.log_loss(p[msk], y[msk]) for k, p in models.items()}
        for k, p in models.items():
            res[k] = {"log_loss": float(ll[k].mean()), "brier": float(ev.brier(p[msk], y[msk]).mean()),
                      "ece": ev.ece(p[msk], y[msk]), "calibration_slope": ev.calibration_slope(p[msk], y[msk])}
        diffs = {}
        for a, b in (("B_existing_blend", "A_market"), ("C_new", "A_market"), ("D_combo", "A_market"),
                     ("C_new", "B_existing_raw"), ("D_combo", "B_existing_blend")):
            dd = ll[a] - ll[b]
            r_ = ev.block_bootstrap_mean(dd, blocks[msk], bs["n"], bs["seed"])
            per = pd.Series(dd).groupby(seasons[msk]).mean()
            r_["seasons_better"] = f"{int((per < 0).sum())}/{len(per)}"
            diffs[f"{a} minus {b}"] = r_
        res["log_loss_differences"] = diffs
        out["markets"][mname] = res
        # ablations of C (difference in log loss vs the full model; negative = ablation better)
        Pm = P if mname == "1x2" else O
        full = ev.log_loss(Pm["C"][msk], y[msk])
        out["ablations"][mname] = {
            k: ev.block_bootstrap_mean(ev.log_loss(Pm[k][msk], y[msk]) - full, blocks[msk], bs["n"], bs["seed"])
            for k in ("C_static", "C_nopool", "C_nodc", "C_plugin")}

    def roi(b):
        b = b[b.season.isin(eval_seasons)]
        r_ = ev.block_bootstrap_mean(b.ret.to_numpy(), b.block.to_numpy(), bs["n"], bs["seed"])
        r_["profit_units"] = float(b.ret.sum())
        r_["by_market"] = {k: {"bets": int(len(g)), "roi": float(g.ret.mean())} for k, g in b.groupby("market")}
        return r_

    out["returns"] = {"B_existing_rule": roi(b_bets), "C_new_rule": roi(bets["C"]), "D_combo_rule": roi(bets["D"])}
    cd = cands[cands.season.isin(eval_seasons)]
    out["selection"] = {
        "candidates_searched": {m: int((cd.model == m).sum()) for m in ("C", "D")},
        "seasons_with_no_selection": {m: [s for s, v in thr_log[m].items() if v["thresholds"] is None] for m in ("C", "D")},
        "thresholds": thr_log,
    }
    out["meta"] = {
        "eval_seasons": [eval_seasons[0], eval_seasons[-1]], "leagues": cfg["data"]["leagues"],
        "filter_setting_by_season": choice["full"], "rho_by_season": rho,
        "devig_by_season": {"1x2": dv1, "ou25": dvo}, "pool_by_season": pools,
        "outer_draws": len(z),
    }
    return out


def write_reports(out: dict, folder: str = "reports"):
    Path(folder).mkdir(exist_ok=True)
    Path(folder, "backtest.json").write_text(json.dumps(out, indent=1, default=str))
