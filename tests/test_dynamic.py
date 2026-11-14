import numpy as np
import pandas as pd

from challenger.dynamic import FilterParams, run_filter
from challenger.tuning import goal_loglik
from tests.conftest import make_league


def test_filter_recovers_known_strengths(league):
    df, att, dfn = league
    _, filters = run_filter(df, FilterParams(tau=0.3, q_year=0.0, season_jump=0.0), return_filters=True)
    s = filters["X"].strengths().set_index("team")
    est_a = s.att.to_numpy()
    true_a = np.array([att[t] for t in s.index])
    est_d = s.dfn.to_numpy()
    true_d = np.array([dfn[t] for t in s.index])
    # at least as good as the direct benchmark (log goals scored / conceded per team)
    gf = np.log(df.groupby("home").hg.sum() + df.groupby("away").ag.sum()).reindex(s.index)
    ga = -np.log(df.groupby("home").ag.sum() + df.groupby("away").hg.sum()).reindex(s.index)
    ca, cd = np.corrcoef(est_a, true_a)[0, 1], np.corrcoef(est_d, true_d)[0, 1]
    assert ca > 0.85 and cd > 0.85
    assert ca > np.corrcoef(gf, true_a)[0, 1] - 0.02
    assert cd > np.corrcoef(ga, true_d)[0, 1] - 0.02
    # stated uncertainty is honest: centred errors are about one posterior sd
    zs = ((est_a - est_a.mean()) - (true_a - true_a.mean())) / s.att_sd.to_numpy()
    assert 0.4 < zs.std() < 1.6
    assert (s.att_sd < 0.3).all()          # posterior tighter than the prior


def test_no_future_information_leaks_into_predictions(league):
    df, _, _ = league
    p0 = run_filter(df, FilterParams())
    cut = df.date.iloc[len(df) // 2]
    df2 = df.copy()
    later = df2.date >= cut
    df2.loc[later, ["hg", "ag"]] = 9      # rewrite the future, incl. the cut date itself
    p1 = run_filter(df2, FilterParams())
    upto = (df.date <= cut).to_numpy()    # same-day matches are predicted before same-day results
    assert np.allclose(p0[upto].to_numpy(), p1[upto].to_numpy())
    assert not np.allclose(p0[~upto].to_numpy(), p1[~upto].to_numpy())


def test_small_samples_shrink_toward_league_baseline():
    df, _, _ = make_league(n_teams=10, seasons=1, seed=5)
    early = df[df.date <= df.date.unique()[2]]          # three match days only
    _, f_pool = run_filter(early, FilterParams(tau=0.3, q_year=0), return_filters=True)
    _, f_wide = run_filter(early, FilterParams(tau=5.0, q_year=0), return_filters=True)
    spread_pool = f_pool["X"].strengths().att.abs().mean()
    spread_wide = f_wide["X"].strengths().att.abs().mean()
    assert spread_pool < 0.5 * spread_wide


def test_chronological_likelihood_prefers_dynamics_only_when_strengths_move():
    def ll(df, q):
        p = run_filter(df, FilterParams(tau=0.3, q_year=q, season_jump=0.0))
        burn = (df.season > df.season.min()).to_numpy()
        return goal_loglik(p[burn], df.hg[burn].to_numpy(), df.ag[burn].to_numpy()).sum()

    moving, _, _ = make_league(seasons=6, seed=11, drift=0.25)
    assert ll(moving, 0.04) > ll(moving, 0.0)
    fixed, _, _ = make_league(seasons=6, seed=11, drift=0.0)
    assert ll(fixed, 0.0) > ll(fixed, 0.04)


def test_entering_team_starts_from_the_entering_team_prior():
    df, _, _ = make_league(n_teams=8, seasons=2, seed=2)
    s2 = df.season == df.season.max()
    df.loc[s2 & (df.home == "T00"), "home"] = "NEW"
    df.loc[s2 & (df.away == "T00"), "away"] = "NEW"
    p = run_filter(df, FilterParams(promo_delta=0.3, tau=0.3))
    first = df[s2 & (df.home == "NEW")].index[0]
    others = df[s2 & (df.home != "NEW") & (df.away != "NEW") & (df.date == df.date[first])].index
    assert p.s_hh[first] > p.s_hh[others].mean()        # more uncertainty about the newcomer


def test_setting_key_round_trips():
    for p in (FilterParams(tau=0.3, q_year=0.005, season_jump=0.01, promo_delta=0.3),
              FilterParams(tau=0.45, q_year=0.002, revert=False)):
        assert FilterParams.from_key(p.key()) == p


def test_random_walk_form_does_not_pull_strengths_to_baseline():
    from challenger.dynamic import LeagueFilter
    for revert, expect in ((True, True), (False, False)):
        f = LeagueFilter(FilterParams(tau=0.3, q_year=0.02, revert=revert))
        f.start_season(2020, {"A", "B"})
        f.m[2] = 0.5
        f.advance_to(pd.Timestamp("2020-08-01"))
        f.advance_to(pd.Timestamp("2022-08-01"))
        assert (f.m[2] < 0.5) == expect
