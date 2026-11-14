"""Selection under uncertainty.

For each eligible selection the expected return is computed for every outer
draw, giving a mean, an interval and the model-conditional probability that
the expected return is positive. These are statements about the model, not
guarantees about the world.

Thresholds are chosen on earlier, out-of-sample seasons only, by the lower
bound of the block-bootstrap ROI. Searching many markets inflates the best
apparent edge, so a threshold pair is accepted only if that lower bound is
positive; otherwise the rule is "no selection".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import block_bootstrap_mean


def ev_summary(p_draws: np.ndarray, odds: np.ndarray) -> dict:
    """p_draws (n, k): conditional win probability per outer draw. Win/lose
    markets only (no push); use ScoreGrid payoffs for push/quarter lines."""
    ev = p_draws * odds[:, None] - 1.0
    return {
        "p": p_draws.mean(axis=1),
        "ev": ev.mean(axis=1),
        "ev_lo": np.quantile(ev, 0.05, axis=1),
        "ev_hi": np.quantile(ev, 0.95, axis=1),
        "p_pos": (ev > 0).mean(axis=1),
    }


def apply_rule(c: pd.DataFrame, ev_min: float, p_pos_min: float) -> pd.DataFrame:
    """Qualifying selections; at most one per match and market (the best)."""
    q = c[(c.ev >= ev_min) & (c.p_pos >= p_pos_min) & c.odds.notna()]
    if q.empty:
        return q
    return q.loc[q.groupby(["match_id", "market"]).ev.idxmax()]


def choose_thresholds(cands: pd.DataFrame, ev_grid, p_grid, min_bets: int, lcb_q: float = 0.05,
                      n_boot: int = 400, seed: int = 0):
    """Best (ev_min, p_pos_min) on earlier seasons, or None for 'no selection'."""
    best, best_lcb, tried = None, 0.0, 0
    for e in ev_grid:
        for p in p_grid:
            bets = apply_rule(cands, e, p)
            if len(bets) < min_bets:
                continue
            tried += 1
            bs = block_bootstrap_mean(bets.ret.to_numpy(), bets.block.to_numpy(), n_boot, seed)
            lcb = bs["q05"] if lcb_q == 0.05 else bs["lo"]
            if lcb > best_lcb:
                best, best_lcb = (e, p), lcb
    return best, best_lcb, tried
