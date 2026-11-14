"""Walk-forward comparison for any two-outcome market (moneyline, spread,
total). Used by the NHL, NBA and UFC back-tests.

For season s the combination weights, their ridge and the selection
thresholds come from seasons before s. Pushes return the stake and are left
out of the probability scores. Intervals resample whole match weeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as ev
from .champion import blend
from .combine import fit_pool_walk_forward
from .selection import apply_rule, choose_thresholds, ev_summary


def _two(p):
    return np.stack([p, 1 - p], axis=-1)


def _cands(meta, model, market, p_draws, odds, y, push):
    rows = []
    for j in (0, 1):
        pj = p_draws if j == 0 else 1 - p_draws
        o = odds[:, j]
        s = ev_summary(pj, np.nan_to_num(o, nan=1.0))
        won = (y == j) & ~push
        rows.append(pd.DataFrame({
            "match_id": meta.match_id.to_numpy(), "season": meta.season.to_numpy(), "block": meta.block.to_numpy(),
            "model": model, "market": market, "sel": j, "odds": o, **s, "won": won,
            "ret": np.where(push, 0.0, np.where(won, o - 1.0, -1.0))}))
    out = pd.concat(rows, ignore_index=True)
    return out[out.odds.notna() & out.p.notna()]


def evaluate_twoway(market: str, meta: pd.DataFrame, y: np.ndarray, push: np.ndarray, p_mkt: np.ndarray,
                    odds: np.ndarray, pc_draws: np.ndarray, stage: dict, cfg: dict,
                    p_b_raw: np.ndarray | None = None, ablations: dict | None = None,
                    reference: dict | None = None, bettable: bool = True) -> dict:
    """y: 1 where outcome 0 (home / red / over) did NOT happen, i.e. the index
    of the realised outcome. p_mkt, p_b_raw: probability of outcome 0.
    pc_draws: (n, draws) conditional probability of outcome 0."""
    seasons = meta.season.to_numpy()
    blocks = meta.block.to_numpy()
    all_seasons = sorted(int(s) for s in np.unique(seasons) if s >= stage["first_pred_season"])
    eval_seasons = [s for s in all_seasons if s >= stage["first_eval_season"]]
    pc = pc_draws.mean(axis=1)
    if not bettable:                       # no price in the data: probabilities are scored, nothing is bet
        odds = np.full((len(y), 2), np.nan)
    ok = ~np.isnan(p_mkt) & ~np.isnan(pc) & ~push
    if bettable:
        ok &= ~np.isnan(odds).any(axis=1)
    pd_draws = np.full(pc_draws.shape, np.nan)
    pools = {}
    for s in all_seasons:
        if s < stage["first_combo_season"]:
            continue
        past = ok & (seasons < s) & (seasons >= stage["first_pred_season"])
        pool = fit_pool_walk_forward(_two(p_mkt[past]), _two(pc[past]), y[past], seasons[past], s, cfg["combine"]["ridge"])
        pools[s] = {"theta": pool.theta.round(4).tolist(), "ridge": pool.ridge}
        m = (seasons == s) & ~np.isnan(p_mkt) & ~np.isnan(pc)
        pd_draws[m] = pool.predict(_two(p_mkt[m]), _two(pc_draws[m]))[:, :, 0]
    pdm = pd_draws.mean(axis=1)

    cands = pd.concat([_cands(meta, "C", market, pc_draws, odds, y, push),
                       _cands(meta, "D", market, pd_draws, odds, y, push)], ignore_index=True)
    if not bettable:
        cands = cands.assign(ret=np.nan).iloc[0:0]
    sel = cfg["selection"]
    bets, thr_log = {}, {"C": {}, "D": {}}
    for model in ("C", "D"):
        cm = cands[cands.model == model]
        got = []
        for s in eval_seasons:
            hist = cm[(cm.season < s) & (cm.season >= s - stage["threshold_lookback"])]
            thr, lcb, tried = choose_thresholds(hist, sel["ev_min"], sel["p_pos_min"], sel["min_bets"], sel["lcb_quantile"])
            thr_log[model][s] = {"thresholds": thr, "earlier_roi_lower_bound": round(lcb, 4), "rules_tried": tried}
            if thr is not None:
                got.append(apply_rule(cm[cm.season == s], *thr))
        bets[model] = pd.concat(got) if got else cm.iloc[0:0]

    bs = cfg["bootstrap"]

    def roi(b):
        b = b[b.season.isin(eval_seasons)]
        r_ = ev.block_bootstrap_mean(b.ret.to_numpy(), b.block.to_numpy(), bs["n"], bs["seed"])
        r_["profit_units"] = float(b.ret.sum())
        return r_

    msk = ok & np.isin(seasons, eval_seasons) & ~np.isnan(pdm)
    models = {"A_market": p_mkt, "C_new": pc, "D_combo": pdm}
    returns = {"C_new_rule": roi(bets["C"]), "D_combo_rule": roi(bets["D"])}
    pairs = [("C_new", "A_market"), ("D_combo", "A_market")]
    if p_b_raw is not None:
        ch = cfg["champion"]
        pb = blend(p_mkt, p_b_raw, ch["market_weight"])
        models = {"A_market": p_mkt, "B_existing_raw": p_b_raw, "B_existing_blend": pb, "C_new": pc, "D_combo": pdm}
        pairs = [("B_existing_blend", "A_market")] + pairs + [("C_new", "B_existing_raw"), ("D_combo", "B_existing_blend")]
        bc = []
        for j in (0, 1):
            pj = pb if j == 0 else 1 - pb
            o = odds[:, j]
            won = (y == j) & ~push
            bc.append(pd.DataFrame({"match_id": meta.match_id.to_numpy(), "season": seasons, "block": blocks,
                                    "market": market, "odds": o, "ev": pj * o - 1.0, "gap": np.abs(p_b_raw - p_mkt),
                                    "ret": np.where(push, 0.0, np.where(won, o - 1.0, -1.0))}))
        bc = pd.concat(bc, ignore_index=True).dropna(subset=["odds", "ev"])
        pos = bc[bc.ev > 0]
        returns = {"B_existing_rule": roi(pos[pos.gap < ch["flag_gap"]]),
                   "B_blend_edge_ignoring_flag": roi(pos), **returns}
    if not bettable:
        returns = {}
    if reference:
        models.update(reference)
    res = {"n_matches": int(msk.sum())}
    ll = {}
    for k, p in models.items():
        mk = msk & ~np.isnan(p)
        ll[k] = ev.log_loss(_two(p[mk]), y[mk]) if mk.sum() == msk.sum() else None
        P2 = _two(p[mk])
        res[k] = {"log_loss": float(ev.log_loss(P2, y[mk]).mean()), "brier": float(ev.brier(P2, y[mk]).mean()),
                  "ece": ev.ece(P2, y[mk]), "calibration_slope": ev.calibration_slope(P2, y[mk]), "n": int(mk.sum())}
    diffs = {}
    for a, b in pairs:
        dd = ll[a] - ll[b]
        r_ = ev.block_bootstrap_mean(dd, blocks[msk], bs["n"], bs["seed"])
        per = pd.Series(dd).groupby(seasons[msk]).mean()
        r_["seasons_better"] = f"{int((per < 0).sum())}/{len(per)}"
        diffs[f"{a} minus {b}"] = r_
    res["log_loss_differences"] = diffs
    abl = {}
    for k, p in (ablations or {}).items():
        abl[k] = ev.block_bootstrap_mean(ev.log_loss(_two(p[msk]), y[msk]) - ll["C_new"], blocks[msk], bs["n"], bs["seed"])
    cd = cands[cands.season.isin(eval_seasons)]
    return {"market": res, "ablations": abl, "returns": returns,
            "selection": {"candidates_searched": int((cd.model == "D").sum()),
                          "seasons_with_no_selection": {m: [s for s, v in thr_log[m].items() if v["thresholds"] is None]
                                                        for m in ("C", "D")},
                          "n_eval_seasons": len(eval_seasons), "thresholds": thr_log},
            "pool_by_season": pools, "eval_seasons": [eval_seasons[0], eval_seasons[-1]],
            "_bets": {k: b[b.season.isin(eval_seasons)] for k, b in bets.items()}}


def strip_private(obj):
    """Drop keys starting with '_' (in-memory detail such as bet tables) before saving."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    return obj
