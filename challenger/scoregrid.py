"""Joint outcome generation: every market is read off one score distribution.

Conditional on a pair of goal rates the score matrix is exact, so 1X2, totals
and handicaps always agree with each other, and the probability of any related
combination is computed directly rather than with correlation multipliers.
"""
from __future__ import annotations

import numpy as np
from math import lgamma


def _log_fact(n: int) -> np.ndarray:
    return np.array([lgamma(k + 1.0) for k in range(n + 1)])


class ScoreGrid:
    def __init__(self, max_goals: int = 10):
        self.G = max_goals
        g = np.arange(max_goals + 1)
        self.g = g
        self.lf = _log_fact(max_goals)
        self.hg = g[:, None] * np.ones_like(g)[None, :]   # home goals on axis 0
        self.ag = np.ones_like(g)[:, None] * g[None, :]
        self.diff = self.hg - self.ag
        self.total = self.hg + self.ag
        self.m_home = (self.diff > 0).astype(float)
        self.m_draw = (self.diff == 0).astype(float)
        self.m_away = (self.diff < 0).astype(float)

    def pmf(self, lam_h: np.ndarray, lam_a: np.ndarray, rho: float = 0.0, diag: float = 1.0) -> np.ndarray:
        """Score matrix for each rate pair. Input shape (...,), output (..., G+1, G+1).

        rho is the Dixon-Coles low-score dependence (0 = independent Poisson),
        used for soccer. diag multiplies every level score (1 = none), used for
        hockey, where teams protect a tie late in regulation."""
        lam_h = np.asarray(lam_h, float)[..., None]
        lam_a = np.asarray(lam_a, float)[..., None]
        ph = np.exp(self.g * np.log(lam_h) - lam_h - self.lf)
        pa = np.exp(self.g * np.log(lam_a) - lam_a - self.lf)
        M = ph[..., :, None] * pa[..., None, :]
        if rho != 0.0:
            lh, la = lam_h[..., 0], lam_a[..., 0]
            M[..., 0, 0] *= np.clip(1.0 - lh * la * rho, 1e-6, None)
            M[..., 0, 1] *= np.clip(1.0 + lh * rho, 1e-6, None)
            M[..., 1, 0] *= np.clip(1.0 + la * rho, 1e-6, None)
            M[..., 1, 1] *= np.clip(1.0 - rho, 1e-6, None)
        if diag != 1.0:
            M = M * (1.0 + (diag - 1.0) * self.m_draw)
        return M / M.sum(axis=(-2, -1), keepdims=True)

    # -- market probabilities ------------------------------------------------
    def p_1x2(self, M: np.ndarray) -> np.ndarray:
        return np.stack([(M * m).sum(axis=(-2, -1)) for m in (self.m_home, self.m_draw, self.m_away)], axis=-1)

    def p_over(self, M: np.ndarray, line: float = 2.5) -> np.ndarray:
        return (M * (self.total > line)).sum(axis=(-2, -1))

    def p_event(self, M: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Probability of any event defined on the score, e.g. a combination
        such as (home win AND over 2.5): mask = m_home * (total > 2.5)."""
        return (M * mask).sum(axis=(-2, -1))

    # -- settlement ------------------------------------------------------------
    def total_payoff(self, line: float, odds: float, side: str) -> np.ndarray:
        """Net return per unit stake on over/under for every score, incl. pushes
        and quarter lines (half the stake on each neighbouring half line)."""
        return _split_line_payoff(self.total - line if side == "over" else line - self.total, odds, line)

    def ah_payoff(self, line: float, odds: float, side: str) -> np.ndarray:
        """Asian handicap. `line` is the home handicap (negative = home favoured)."""
        margin = self.diff + line if side == "home" else -(self.diff + line)
        return _split_line_payoff(margin, odds, line)

    def expected_return(self, M: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        return (M * payoff).sum(axis=(-2, -1))


def _settle(margin: np.ndarray, odds: float) -> np.ndarray:
    return np.where(margin > 1e-9, odds - 1.0, np.where(margin < -1e-9, -1.0, 0.0))


def _split_line_payoff(margin: np.ndarray, odds: float, line: float) -> np.ndarray:
    q = round(line * 4) % 2  # 1 for quarter lines (x.25, x.75)
    if q == 1:
        return 0.5 * (_settle(margin - 0.25, odds) + _settle(margin + 0.25, odds))
    return _settle(margin, odds)
