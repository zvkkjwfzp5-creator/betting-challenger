"""Margin-adjusted market baseline from complete outcome sets.

One bookmaker snapshot per match is used. Several books quoting the same game
are not independent evidence, so they are never stacked as extra observations.
"""
from __future__ import annotations

import numpy as np


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    """odds: (n, k) decimal prices of a complete outcome set."""
    q = 1.0 / np.asarray(odds, float)
    return q / q.sum(axis=1, keepdims=True)


def devig_power(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    """Find c with sum(q**c) = 1. Removes more margin from longshots."""
    q = 1.0 / np.asarray(odds, float)
    lo = np.full(len(q), 0.5)
    hi = np.full(len(q), 3.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = (q ** mid[:, None]).sum(axis=1)
        hi = np.where(s < 1.0, mid, hi)
        lo = np.where(s >= 1.0, mid, lo)
    p = q ** (0.5 * (lo + hi))[:, None]
    return p / p.sum(axis=1, keepdims=True)


def devig_shin(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    """Shin (1993): margin explained by a share z of informed money."""
    q = 1.0 / np.asarray(odds, float)
    b = q.sum(axis=1, keepdims=True)

    def probs(z):
        z = z[:, None]
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / b) - z) / (2 * (1 - z))

    lo = np.zeros(len(q))
    hi = np.full(len(q), 0.4)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = probs(mid).sum(axis=1)
        lo = np.where(s > 1.0, mid, lo)
        hi = np.where(s <= 1.0, mid, hi)
    p = probs(0.5 * (lo + hi))
    return p / p.sum(axis=1, keepdims=True)


DEVIG = {"multiplicative": devig_multiplicative, "power": devig_power, "shin": devig_shin}


def overround(odds: np.ndarray) -> np.ndarray:
    return (1.0 / np.asarray(odds, float)).sum(axis=1) - 1.0
