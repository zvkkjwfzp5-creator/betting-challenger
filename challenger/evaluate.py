"""Scoring rules, calibration and block-bootstrap intervals."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, None))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    Y = np.eye(p.shape[1])[y]
    return ((p - Y) ** 2).sum(axis=1)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over all outcome classes (equal-count bins)."""
    Y = np.eye(p.shape[1])[y].ravel()
    q = p.ravel()
    order = np.argsort(q)
    q, Y = q[order], Y[order]
    parts = np.array_split(np.arange(len(q)), bins)
    return float(sum(len(ix) * abs(q[ix].mean() - Y[ix].mean()) for ix in parts) / len(q))


def calibration_slope(p: np.ndarray, y: np.ndarray, iters: int = 30) -> float:
    """Slope a in softmax(a * log p): 1 = calibrated, <1 = over-confident."""
    lp = np.log(np.clip(p, EPS, None))
    Y = np.eye(p.shape[1])[y]
    a = 1.0
    for _ in range(iters):
        z = a * lp
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        g = ((q - Y) * lp).sum()
        m = (q * lp).sum(axis=1)
        h = ((q * lp ** 2).sum(axis=1) - m ** 2).sum()
        a -= g / max(h, 1e-9)
    return float(a)


def block_bootstrap_mean(values: np.ndarray, blocks: np.ndarray, n_boot: int = 2000, seed: int = 0,
                         weights: np.ndarray | None = None):
    """Mean of `values` (optionally sum(values)/sum(weights)) with a percentile
    interval from resampling whole blocks (match weeks), so results that share
    a week are never treated as independent."""
    values = np.asarray(values, float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, float)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    _, inv = np.unique(blocks, return_inverse=True)
    nb = inv.max() + 1
    s = np.bincount(inv, values, nb)
    c = np.bincount(inv, w, nb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    den = c[idx].sum(axis=1)
    boot = s[idx].sum(axis=1) / np.where(den > 0, den, np.nan)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"mean": float(s.sum() / c.sum()), "lo": float(lo), "hi": float(hi), "n": int(len(values)),
            "p_gt0": float(np.nanmean(boot > 0)), "q05": float(np.nanpercentile(boot, 5))}
