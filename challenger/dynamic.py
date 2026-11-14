"""Dynamic hierarchical attack/defence model (one state-space model per league).

    log(home goal rate) = mu + h + att[home] - dfn[away]
    log(away goal rate) = mu     + att[away] - dfn[home]

Partial pooling: every team's att and dfn has a stationary prior N(0, tau^2)
around the league baseline, so short samples shrink toward it.

Dynamics: each strength gains evolution variance `q_year` per year (plus
`season_jump` at each season break). Two forms are searched: a mean-reverting
(Ornstein-Uhlenbeck) process whose stationary variance is tau^2, and a random
walk where pooling acts through the prior a team enters with. q_year = 0 and
season_jump = 0 give a static model. Form and rates are chosen by chronological
one-step-ahead likelihood, not by a form window.

Inference is a sequential Gaussian filter with a Laplace update in the
two-dimensional space of the match's log rates. It is strictly causal:
every match on a date is predicted before any result of that date is used.

Home advantage `h` and the baseline `mu` are league-level states. Opponent
strength enters through the model structure itself, so it is not added again
as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterParams:
    tau: float = 0.30            # pooling sd of att/dfn
    q_year: float = 0.015        # evolution variance per year
    season_jump: float = 0.01    # extra variance at a season break
    promo_delta: float = 0.20    # entering teams start at att = dfn = -promo_delta
    revert: bool = True          # True: strengths are pulled back toward the league baseline (OU)
                                 # False: random walk; pooling acts through the entry prior only
    q_league_year: float = 0.002  # evolution variance of mu and h per year
    mu0: float = 0.15
    h0: float = 0.30
    newton_iters: int = 6

    def key(self) -> str:
        k = f"tau={self.tau}|q={self.q_year}|jump={self.season_jump}|promo={self.promo_delta}"
        return k if self.revert else k + "|rw"

    @staticmethod
    def from_key(key: str) -> "FilterParams":
        parts = key.split("|")
        kv = dict(x.split("=") for x in parts if "=" in x)
        return FilterParams(tau=float(kv["tau"]), q_year=float(kv["q"]), season_jump=float(kv["jump"]),
                            promo_delta=float(kv["promo"]), revert="rw" not in parts)


PRED_COLS = ["eta_h", "eta_a", "s_hh", "s_aa", "s_ha"]


def _laplace_2d(y: np.ndarray, eta0: np.ndarray, S: np.ndarray, iters: int):
    """Mode and covariance of p(eta | y) for two Poisson counts with log link."""
    Sinv = np.linalg.inv(S)
    eta = eta0.copy()
    for _ in range(iters):
        lam = np.exp(eta)
        g = y - lam - Sinv @ (eta - eta0)
        H = Sinv + np.diag(lam)
        step = np.linalg.solve(H, g)
        eta = eta + np.clip(step, -1.5, 1.5)
    S_post = np.linalg.inv(Sinv + np.diag(np.exp(eta)))
    return eta, S_post


class LeagueFilter:
    """Filter state for one league. State = [mu, h, att_0, dfn_0, att_1, dfn_1, ...]."""

    def __init__(self, params: FilterParams):
        self.p = params
        self.slot: dict[str, int] = {}
        self.m = np.array([params.mu0, params.h0])
        self.P = np.diag([1.0, 0.25])
        self.last_date: pd.Timestamp | None = None
        self.last_season: int | None = None
        self.prev_teams: set[str] = set()

    # -- state bookkeeping -------------------------------------------------
    def _add_team(self, team: str, mean: float):
        n = len(self.m)
        self.slot[team] = n
        self.m = np.concatenate([self.m, [mean, mean]])
        P = np.zeros((n + 2, n + 2))
        P[:n, :n] = self.P
        P[n, n] = P[n + 1, n + 1] = self.p.tau ** 2
        self.P = P

    def _reset_team(self, team: str, mean: float):
        a = self.slot[team]
        for i in (a, a + 1):
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = self.p.tau ** 2
            self.m[i] = mean

    def start_season(self, season: int, teams: set[str]):
        """Register the season's teams. Teams entering the league start from the
        'entering team' prior; teams present last season carry their state."""
        first = self.last_season is None
        for t in sorted(teams):
            mean = 0.0 if first else -self.p.promo_delta
            if t not in self.slot:
                self._add_team(t, mean)
            elif t not in self.prev_teams:
                self._reset_team(t, mean)
        if not first and self.p.season_jump > 0:
            self._evolve_var(self.p.season_jump)
        self.prev_teams = set(teams)
        self.last_season = season

    def _evolve_var(self, v_team: float, v_league: float = 0.0):
        """Advance all team states by an OU step that adds variance v_team."""
        n = len(self.m)
        if n > 2 and v_team > 0 and not self.p.revert:
            idx = np.arange(2, n)
            self.P[idx, idx] += v_team
        elif n > 2 and v_team > 0:
            tau2 = self.p.tau ** 2
            # OU: var' = f^2 var + tau2 (1 - f^2); small-step added variance = v_team.
            f2 = max(0.0, 1.0 - v_team / tau2)
            f = np.ones(n)
            f[2:] = np.sqrt(f2)
            self.m = self.m * f
            self.P = self.P * np.outer(f, f)
            idx = np.arange(2, n)
            self.P[idx, idx] += tau2 * (1.0 - f2)
        if v_league > 0:
            self.P[0, 0] += v_league
            self.P[1, 1] += v_league

    def advance_to(self, date: pd.Timestamp):
        if self.last_date is not None:
            years = (date - self.last_date).days / 365.25
            if years > 0:
                self._evolve_var(self.p.q_year * years, self.p.q_league_year * years)
        self.last_date = date

    # -- prediction and update ----------------------------------------------
    def _design(self, home: str, away: str):
        ah, aa = self.slot[home], self.slot[away]
        ih = np.array([0, 1, ah, aa + 1])
        sh = np.array([1.0, 1.0, 1.0, -1.0])
        ia = np.array([0, aa, ah + 1])
        sa = np.array([1.0, 1.0, -1.0])
        return ih, sh, ia, sa

    def predict(self, home: str, away: str):
        """Predictive mean and covariance of (log home rate, log away rate)."""
        ih, sh, ia, sa = self._design(home, away)
        eta0 = np.array([self.m[ih] @ sh, self.m[ia] @ sa])
        PXt = np.column_stack([self.P[:, ih] @ sh, self.P[:, ia] @ sa])
        S = np.array([[PXt[ih, 0] @ sh, PXt[ih, 1] @ sh],
                      [PXt[ia, 0] @ sa, PXt[ia, 1] @ sa]])
        S = 0.5 * (S + S.T)
        return eta0, S, PXt

    def update(self, home: str, away: str, hg: int, ag: int):
        eta0, S, PXt = self.predict(home, away)
        eta, S_post = _laplace_2d(np.array([hg, ag], float), eta0, S, self.p.newton_iters)
        K = PXt @ np.linalg.inv(S)
        self.m = self.m + K @ (eta - eta0)
        self.P = self.P - K @ (S - S_post) @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def strengths(self) -> pd.DataFrame:
        rows = []
        for t, a in self.slot.items():
            if t in self.prev_teams:
                rows.append((t, self.m[a], np.sqrt(self.P[a, a]), self.m[a + 1], np.sqrt(self.P[a + 1, a + 1])))
        return pd.DataFrame(rows, columns=["team", "att", "att_sd", "dfn", "dfn_sd"]).sort_values(
            "att", ascending=False)


def run_filter(df: pd.DataFrame, params: FilterParams, return_filters: bool = False):
    """One causal pass over all leagues.

    Returns a frame indexed like `df` with the pre-match predictive moments of
    (log home rate, log away rate): eta_h, eta_a, s_hh, s_aa, s_ha.
    """
    out = np.full((len(df), 5), np.nan)
    filters: dict[str, LeagueFilter] = {}
    pos = {mid: i for i, mid in enumerate(df["match_id"].to_numpy())}
    for league, g in df.groupby("league", sort=False):
        f = LeagueFilter(params)
        filters[league] = f
        teams_by_season = {s: set(x.home) | set(x.away) for s, x in g.groupby("season")}
        for (season, date), day in g.groupby(["season", "date"], sort=True):
            if f.last_season != season:
                f.start_season(season, teams_by_season[season])
            f.advance_to(date)
            rows = list(day.itertuples(index=False))
            for r in rows:                      # predict the whole date first
                eta0, S, _ = f.predict(r.home, r.away)
                out[pos[r.match_id]] = [eta0[0], eta0[1], S[0, 0], S[1, 1], S[0, 1]]
            for r in rows:                      # only then learn from its results
                f.update(r.home, r.away, r.hg, r.ag)
    preds = pd.DataFrame(out, columns=PRED_COLS, index=df.index)
    return (preds, filters) if return_filters else preds
