"""Regularised combination of market and model (log-linear pooling).

    logit_k = w_market * log p_market_k + w_model * log p_model_k + b_k

The ridge penalty pulls toward (w_market, w_model, b) = (1, 0, 0), i.e. toward
"market alone", so the model only gets weight it has earned on earlier data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class Pool:
    theta: np.ndarray  # [w_market, w_model, b_1..b_{K-1}]
    ridge: float

    def predict(self, p_market: np.ndarray, p_model: np.ndarray) -> np.ndarray:
        """p_market (n, K); p_model (n, K) or (n, draws, K)."""
        lm = np.log(np.clip(p_market, EPS, None))
        lc = np.log(np.clip(p_model, EPS, None))
        if lc.ndim == 3:
            lm = lm[:, None, :]
        z = self.theta[0] * lm + self.theta[1] * lc
        z[..., 1:] = z[..., 1:] + self.theta[2:]
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


def fit_pool(p_market: np.ndarray, p_model: np.ndarray, y: np.ndarray, ridge: float, iters: int = 50) -> Pool:
    """y: integer outcome index (n,). Newton's method on the penalised log loss."""
    n, K = p_market.shape
    P = 2 + K - 1
    F = np.zeros((n, K, P))
    F[:, :, 0] = np.log(np.clip(p_market, EPS, None))
    F[:, :, 1] = np.log(np.clip(p_model, EPS, None))
    for j in range(1, K):
        F[:, j, 1 + j] = 1.0
    Y = np.eye(K)[y]
    theta0 = np.zeros(P)
    theta0[0] = 1.0
    theta = theta0.copy()
    for _ in range(iters):
        z = F @ theta
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad = np.einsum("nk,nkp->p", p - Y, F) + ridge * (theta - theta0)
        Fbar = np.einsum("nk,nkp->np", p, F)
        H = np.einsum("nk,nkp,nkq->pq", p, F, F) - Fbar.T @ Fbar + ridge * np.eye(P) + 1e-9 * np.eye(P)
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.abs(step).max() < 1e-9:
            break
    return Pool(theta, ridge)


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, None))


def fit_pool_walk_forward(p_market, p_model, y, seasons, target_season: int, ridges: list[float],
                          inner_seasons: int = 3) -> Pool:
    """Fit for `target_season` using strictly earlier seasons. The ridge is
    itself chosen by an inner walk-forward over the last `inner_seasons`."""
    past = seasons < target_season
    if not past.any():
        raise ValueError("no earlier data")
    prior = np.unique(seasons[past])
    best, best_score = ridges[-1], np.inf
    if len(prior) >= 2:
        best_score = np.inf
        for r in ridges:
            tot = 0.0
            for s in prior[-inner_seasons:]:
                tr, te = seasons < s, seasons == s
                if not tr.any():
                    continue
                pool = fit_pool(p_market[tr], p_model[tr], y[tr], r)
                tot += log_loss(pool.predict(p_market[te], p_model[te]), y[te]).sum()
            if tot < best_score:
                best, best_score = r, tot
    return fit_pool(p_market[past], p_model[past], y[past], best)
