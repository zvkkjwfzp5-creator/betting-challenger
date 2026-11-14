"""Nested posterior predictive pricing.

Outer level: draw plausible parameters (the match's two log goal rates) from
the filter's posterior, optionally under pre-game scenarios.
Inner level: the outcome distribution conditional on each draw. It is computed
exactly from the score matrix, which is the limit of simulating infinitely
many games per draw; `simulate_scores` does the literal simulation and is used
by the tests to confirm the two agree.

Keeping the conditional probability of every draw separates three things:
  * game randomness            -> the probability itself
  * parameter uncertainty      -> spread of the conditional probabilities
  * Monte Carlo sampling error -> that spread / sqrt(number of draws)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoregrid import ScoreGrid


@dataclass(frozen=True)
class Scenario:
    """A pre-game scenario as a shift of the log goal rates.

    weight=None means no evidence-based probability is available: the scenario
    is then reported as a sensitivity case and never averaged in."""
    name: str
    d_home: float = 0.0
    d_away: float = 0.0
    weight: float | None = None


def standard_draws(k: int, seed: int) -> np.ndarray:
    """(k, 2) standard normal draws, antithetic pairs (exactly mean zero)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((k // 2, 2))
    return np.concatenate([z, -z])


def rate_draws(eta_h, eta_a, s_hh, s_aa, s_ha, z: np.ndarray):
    """Outer draws of (home rate, away rate). Inputs shape (n,), z (k, 2).
    Returns two arrays of shape (n, k)."""
    eta_h, eta_a, s_hh, s_aa, s_ha = (np.asarray(x, float) for x in (eta_h, eta_a, s_hh, s_aa, s_ha))
    l11 = np.sqrt(np.clip(s_hh, 1e-12, None))
    l21 = s_ha / l11
    l22 = np.sqrt(np.clip(s_aa - l21 ** 2, 1e-12, None))
    eh = eta_h[:, None] + l11[:, None] * z[None, :, 0]
    ea = eta_a[:, None] + l21[:, None] * z[None, :, 0] + l22[:, None] * z[None, :, 1]
    # Guard: keep rates inside 0.02..12 goals so a very diffuse posterior
    # (e.g. the no-pooling ablation) cannot overflow the score matrix.
    return np.exp(np.clip(eh, -4.0, 2.5)), np.exp(np.clip(ea, -4.0, 2.5))


def conditional_probs(grid: ScoreGrid, lam_h: np.ndarray, lam_a: np.ndarray, rho: float):
    """Per-draw probabilities of the standard markets, all from one score matrix.
    Returns dict of arrays shaped (n, k, ...)."""
    M = grid.pmf(lam_h, lam_a, rho)
    return {"1x2": grid.p_1x2(M), "over25": grid.p_over(M, 2.5), "M": M}


def summarize(p_draws: np.ndarray, axis: int = -1) -> dict:
    """Mean, parameter-uncertainty sd and Monte Carlo standard error."""
    k = p_draws.shape[axis]
    sd = p_draws.std(axis=axis, ddof=1)
    return {"mean": p_draws.mean(axis=axis), "param_sd": sd, "mc_se": sd / np.sqrt(k)}


def simulate_scores(lam_h: float, lam_a: float, n: int, rng: np.random.Generator,
                    grid: ScoreGrid, rho: float = 0.0):
    """Literal inner loop: simulate n games for one parameter draw."""
    M = grid.pmf(np.array(lam_h), np.array(lam_a), rho)
    flat = rng.choice(M.size, size=n, p=M.ravel())
    return np.unravel_index(flat, M.shape)
