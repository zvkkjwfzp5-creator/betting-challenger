"""Dynamic hierarchical fighter ratings for UFC (binary outcome).

    logit P(red wins) = c + b_age * age_diff + b_reach * reach_diff + skill[red] - skill[blue]

A fight has no score, so this is a paired-comparison (Bradley-Terry) model
rather than a scoring-distribution model. Each fighter's skill starts from a
pooled prior N(-debut_delta, tau^2) and gains variance `q_year` per year, so a
long lay-off widens the uncertainty. With a median of four fights per
fighter, pooling does most of the work. c is the red-corner edge (the red
corner is usually the higher-ranked fighter); age and reach are known before
the fight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mathx import sigmoid

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / _GH_W.sum()


@dataclass(frozen=True)
class RatingParams:
    tau: float = 0.6
    q_year: float = 0.03
    debut_delta: float = 0.0
    use_cov: bool = True

    def key(self) -> str:
        return f"tau={self.tau}|q={self.q_year}|debut={self.debut_delta}|cov={int(self.use_cov)}"


def run_rating_filter(df: pd.DataFrame, p: RatingParams):
    """Causal pass. Returns (preds, loglik) with preds = [logit mean, logit variance]."""
    # Slots in order of first appearance: fighters not seen yet are independent
    # of everything, so each update only needs the block of fighters seen so far.
    order = pd.unique(df.sort_values("date", kind="stable")[["home", "away"]].to_numpy().ravel())
    fighters = list(order)
    slot = {f: 3 + i for i, f in enumerate(fighters)}
    n = 3 + len(fighters)
    na = 3
    m = np.zeros(n)
    m[3:] = -p.debut_delta
    var0 = [1.0, 0.25, 0.25] if p.use_cov else [1.0, 1e-12, 1e-12]
    P = np.diag(var0 + [p.tau ** 2] * len(fighters))
    seen_at: dict[str, pd.Timestamp] = {}
    out = np.full((len(df), 2), np.nan)
    ll = np.full(len(df), np.nan)
    pos = {mid: i for i, mid in enumerate(df.match_id.to_numpy())}

    def x_of(r):
        return np.array([0, 1, 2, slot[r.home], slot[r.away]]), np.array([1.0, r.age_diff, r.reach_diff, 1.0, -1.0])

    for date, day in df.groupby("date", sort=True):
        rows = list(day.itertuples(index=False))
        for r in rows:                       # lay-off widens uncertainty (random walk in time)
            for f in (r.home, r.away):
                if f in seen_at and p.q_year > 0:
                    P[slot[f], slot[f]] += p.q_year * (date - seen_at[f]).days / 365.25
                seen_at[f] = date
                na = max(na, slot[f] + 1)
        for r in rows:                       # predict the whole card first
            ix, x = x_of(r)
            eta0 = m[ix] @ x
            s = x @ P[np.ix_(ix, ix)] @ x
            out[pos[r.match_id]] = [eta0, s]
            pr = (sigmoid(eta0 + np.sqrt(s) * _GH_X) * _GH_W).sum()
            ll[pos[r.match_id]] = np.log(pr if r.red_win else 1 - pr)
        for r in rows:                       # then learn from its results
            ix, x = x_of(r)
            eta0 = m[ix] @ x
            Px = P[:na, ix] @ x
            s = x @ Px[ix]
            eta = eta0
            for _ in range(8):               # Laplace: mode of the 1-D posterior of the logit
                pr = sigmoid(eta)
                g = (r.red_win - pr) - (eta - eta0) / s
                eta += np.clip(g / (pr * (1 - pr) + 1 / s), -2, 2)
            pr = sigmoid(eta)
            s_post = 1.0 / (1.0 / s + pr * (1 - pr))
            K = Px / s
            m[:na] += K * (eta - eta0)
            P[:na, :na] -= np.outer(K, K * (s - s_post))
    return out, ll
