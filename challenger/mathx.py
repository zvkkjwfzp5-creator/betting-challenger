"""Small numeric helpers (no SciPy dependency)."""
from __future__ import annotations

import numpy as np


def ndtr(x) -> np.ndarray:
    """Standard normal CDF, vectorised (Abramowitz-Stegun 7.1.26 erf; error < 1.5e-7)."""
    x = np.asarray(x, float)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 30) -> np.ndarray:
    """Plain ridge logistic regression by Newton's method. X already includes
    any intercept column."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ b)
        g = X.T @ (p - y) + ridge * b
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-10:
            break
    return b
