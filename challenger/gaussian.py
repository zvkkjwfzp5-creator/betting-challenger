"""Dynamic hierarchical points model for basketball (exact Kalman filter).

    home points = mu + h/2 + off[home] - dfn[away] - b/2 * (B_home - B_away) + noise
    away points = mu - h/2 + off[away] - dfn[home] + b/2 * (B_home - B_away) + noise

B_* flags a back-to-back (played the previous calendar day); b is a league
level state learned from data. Points are roughly normal with correlated
noise between the two teams (shared pace), so a Gaussian model is used here
rather than the Poisson model used for goals. Margin and total are both read
from the same pair of predicted scores.

off and dfn are pooled toward the league baseline with prior sd `tau`
(points) and gain variance `q_year` per year plus `season_jump` each summer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GaussParams:
    tau: float = 3.0
    q_year: float = 2.0
    season_jump: float = 2.0
    use_b2b: bool = True
    sigma: float = 10.8          # per-team noise sd used for the filter gain
    rho: float = 0.33            # correlation of the two teams' noise (pace)
    q_mu_year: float = 4.0       # league scoring level drifts
    mu0: float = 100.0

    def key(self) -> str:
        return f"tau={self.tau}|q={self.q_year}|jump={self.season_jump}|b2b={int(self.use_b2b)}"


def run_gauss_filter(df: pd.DataFrame, p: GaussParams):
    """Causal pass. Returns (preds, loglik): preds columns are the predictive
    mean of (margin, total) and their parameter variances/covariance:
    [m_margin, m_total, v_margin, v_total, v_mt]."""
    teams = sorted(set(df.home) | set(df.away))
    slot = {t: 3 + 2 * i for i, t in enumerate(teams)}
    n = 3 + 2 * len(teams)
    m = np.zeros(n)
    m[0], m[1], m[2] = p.mu0, 3.0, 0.0
    P = np.diag([100.0, 9.0, 4.0 if p.use_b2b else 1e-12] + [p.tau ** 2] * (2 * len(teams)))
    R = p.sigma ** 2 * np.array([[1.0, p.rho], [p.rho, 1.0]])
    tidx = np.arange(3, n)
    T = np.array([[1.0, -1.0], [1.0, 1.0]])           # (home, away) -> (margin, total)
    out = np.full((len(df), 5), np.nan)
    ll = np.full(len(df), np.nan)
    pos = {mid: i for i, mid in enumerate(df.match_id.to_numpy())}
    last_date, last_season = None, None

    def add_var(v_team, v_mu):
        nonlocal m, P
        if v_team > 0:
            tau2 = p.tau ** 2
            f2 = max(0.0, 1.0 - v_team / tau2)
            f = np.ones(n)
            f[3:] = np.sqrt(f2)
            m = m * f
            P = P * np.outer(f, f)
            P[tidx, tidx] += tau2 * (1 - f2)
        P[0, 0] += v_mu

    def design(r):
        db = (r.b2b_h - r.b2b_a) if p.use_b2b else 0.0
        H = np.zeros((2, n))
        H[0, [0, 1, 2, slot[r.home], slot[r.away] + 1]] = [1.0, 0.5, -0.5 * db, 1.0, -1.0]
        H[1, [0, 1, 2, slot[r.away], slot[r.home] + 1]] = [1.0, -0.5, 0.5 * db, 1.0, -1.0]
        return H

    for (season, date), day in df.groupby(["season", "date"], sort=True):
        if last_season is not None and season != last_season:
            add_var(p.season_jump, 0.0)
        if last_date is not None:
            yrs = (date - last_date).days / 365.25
            add_var(p.q_year * yrs, p.q_mu_year * yrs)
        last_date, last_season = date, season
        rows = list(day.itertuples(index=False))
        for r in rows:                                   # predict the whole date first
            H = design(r)
            mu = H @ m
            S = H @ P @ H.T
            mt, St = T @ mu, T @ S @ T.T
            out[pos[r.match_id]] = [mt[0], mt[1], St[0, 0], St[1, 1], St[0, 1]]
            C = S + R
            e = np.array([r.hg, r.ag]) - mu
            ll[pos[r.match_id]] = -0.5 * (e @ np.linalg.solve(C, e) + np.log(np.linalg.det(C)) + 2 * np.log(2 * np.pi))
        for r in rows:                                   # then learn from its results
            H = design(r)
            PHt = P @ H.T
            C = H @ PHt + R
            K = PHt @ np.linalg.inv(C)
            m = m + K @ (np.array([r.hg, r.ag]) - H @ m)
            P = P - K @ C @ K.T
            P = 0.5 * (P + P.T)
    return out, ll
