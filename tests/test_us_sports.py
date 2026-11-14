import math

import numpy as np
import pandas as pd
import pytest

from challenger.data_us import american_to_decimal, repair_dates
from challenger.gaussian import GaussParams, run_gauss_filter
from challenger.mathx import fit_logistic, ndtr, sigmoid
from challenger.nba import line_probs
from challenger.nhl import home_win_draws
from challenger.rating import RatingParams, run_rating_filter
from challenger.scoregrid import ScoreGrid
from challenger.twoway import evaluate_twoway


def test_american_prices():
    d = american_to_decimal([-110, 150, -200, 0, np.nan, 50])
    assert d[:3] == pytest.approx([1.9090909, 2.5, 1.5])
    assert np.isnan(d[3:]).all()                      # 0, missing and impossible prices are not prices


def test_repair_dates_restores_playing_order():
    # 2019/20 as the archive has it: autumn 2019, winter 2020, then the 2020 bubble labelled 2019
    raw = pd.to_datetime(["2019-10-05", "2020-01-10", "2020-03-10", "2019-08-02", "2019-09-20", "2019-10-01"])
    fixed = repair_dates(pd.Series(raw), pd.Series([2019] * 6))
    assert fixed.is_monotonic_increasing
    assert fixed.iloc[3] == pd.Timestamp("2020-08-02") and fixed.iloc[5] == pd.Timestamp("2020-10-01")
    # 2020/21 started in January 2021 but is labelled 2020
    raw = pd.to_datetime(["2020-01-13", "2020-03-01", "2021-04-01"])
    assert repair_dates(pd.Series(raw), pd.Series([2020] * 3)).iloc[0] == pd.Timestamp("2021-01-13")


def test_normal_cdf_and_logistic_fit():
    xs = np.linspace(-5, 5, 41)
    exact = np.array([0.5 * (1 + math.erf(x / math.sqrt(2))) for x in xs])
    assert np.abs(ndtr(xs) - exact).max() < 1e-6
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20000), rng.normal(size=20000)])
    y = (rng.random(20000) < sigmoid(X @ np.array([0.3, -0.8]))).astype(float)
    assert fit_logistic(X, y) == pytest.approx([0.3, -0.8], abs=0.06)


def test_spread_and_total_lines_handle_pushes():
    mean = np.array([[4.0], [4.0]])
    win, push = line_probs(mean, 12.0, np.array([4.0, 4.5]))
    assert push[0, 0] > 0.02 and push[1, 0] == 0.0     # only whole-number lines can push
    lose = ndtr((np.array([3.5, 4.5]) - 4.0) / 12.0)
    assert (win[:, 0] + push[:, 0] + lose) == pytest.approx([1.0, 1.0])
    assert win[0, 0] < win[1, 0] + push[0, 0]


def _nba_like(seed=0, n_teams=10, rounds=8):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    off, dfn = rng.normal(0, 3, n_teams), rng.normal(0, 3, n_teams)
    rows, day = [], pd.Timestamp("2015-10-20")
    for _ in range(rounds):
        fx = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fx)
        for i, (h, a) in enumerate(fx):
            if i % (n_teams // 2) == 0:
                day += pd.Timedelta(days=2)
            noise = rng.multivariate_normal([0, 0], 117 * np.array([[1, .33], [.33, 1]]))
            rows.append((2015, day, teams[h], teams[a], round(100 + 1.5 + off[h] - dfn[a] + noise[0]),
                         round(100 - 1.5 + off[a] - dfn[h] + noise[1]), 0.0, 0.0))
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag", "b2b_h", "b2b_a"])
    df["match_id"] = np.arange(len(df))
    return df, off + dfn


def test_points_model_recovers_strength_and_is_causal():
    df, net = _nba_like()
    preds, ll = run_gauss_filter(df, GaussParams(tau=3.0, q_year=0.0, season_jump=0.0))
    late = df.index >= len(df) * 3 // 4
    true_margin = np.array([3.0 + net[int(h[1:])] - net[int(a[1:])] for h, a in zip(df.home, df.away)])
    assert np.corrcoef(preds[late, 0], true_margin[late])[0, 1] > 0.85
    assert np.isfinite(ll).all()
    cut = df.date.iloc[len(df) // 2]
    df2 = df.copy()
    df2.loc[df2.date >= cut, ["hg", "ag"]] = 150
    p2, _ = run_gauss_filter(df2, GaussParams(tau=3.0, q_year=0.0, season_jump=0.0))
    upto = (df.date <= cut).to_numpy()
    assert np.allclose(preds[upto], p2[upto]) and not np.allclose(preds[~upto], p2[~upto])


def _fights(seed=0, n=40, fights=600):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.8, n)
    rows = []
    for i in range(fights):
        a, b = rng.choice(n, 2, replace=False)
        rows.append((pd.Timestamp("2015-01-03") + pd.Timedelta(days=7 * (i // 6)), f"F{a}", f"F{b}",
                     int(rng.random() < sigmoid(skill[a] - skill[b])), 0.0, 0.0))
    df = pd.DataFrame(rows, columns=["date", "home", "away", "red_win", "age_diff", "reach_diff"])
    df["match_id"] = np.arange(len(df))
    return df, skill


def test_fight_ratings_predict_and_are_causal_and_pooled():
    df, skill = _fights()
    preds, ll = run_rating_filter(df, RatingParams(tau=0.8, q_year=0.0))
    late = df.index >= 300
    truth = np.array([skill[int(a[1:])] - skill[int(b[1:])] for a, b in zip(df.home, df.away)])
    assert np.corrcoef(preds[late, 0], truth[late])[0, 1] > 0.7
    assert ll[late].mean() > math.log(0.5)                 # better than a coin flip
    cut = df.date.iloc[300]
    df2 = df.copy()
    df2.loc[df2.date >= cut, "red_win"] = 1
    p2, _ = run_rating_filter(df2, RatingParams(tau=0.8, q_year=0.0))
    upto = (df.date <= cut).to_numpy()
    assert np.allclose(preds[upto], p2[upto])
    # pooling: with a tight prior the early logits stay near zero; without it they swing
    tight, _ = run_rating_filter(df[:60], RatingParams(tau=0.3, q_year=0.0))
    wide, _ = run_rating_filter(df[:60], RatingParams(tau=5.0, q_year=0.0))
    assert np.abs(tight[30:, 0]).mean() < 0.3 * np.abs(wide[30:, 0]).mean()
    # a lay-off widens the uncertainty
    quick, _ = run_rating_filter(df, RatingParams(tau=0.8, q_year=0.0))
    slow, _ = run_rating_filter(df, RatingParams(tau=0.8, q_year=0.5))
    assert slow[late, 1].mean() > quick[late, 1].mean()


def test_hockey_overtime_step_is_coherent():
    g = ScoreGrid(12)
    preds = np.array([[1.0, 0.9, 0.0, 0.0, 0.0], [0.9, 1.0, 0.0, 0.0, 0.0]])
    z = np.zeros((1, 2))
    p = home_win_draws(preds, z, g, 1.3, np.zeros(2))[:, 0]
    assert p[0] + p[1] == pytest.approx(1.0)             # mirror fixtures: someone wins
    p3 = g.p_1x2(g.pmf(np.exp(1.0), np.exp(0.9), 0.0, 1.3))
    assert p[0] == pytest.approx(p3[0] + 0.5 * p3[1])    # coin-flip overtime when the step is switched off
    plain = g.p_1x2(g.pmf(np.exp(1.0), np.exp(0.9)))
    assert p3[1] > plain[1]                              # tie inflation adds regulation ties
    stronger = home_win_draws(preds, z, g, 1.3, np.array([0.0, 1.0]))[:, 0]
    assert stronger[0] > p[0]                            # the better side also wins overtime more often


def _synthetic_market(seed=0, n=6000, model_noise=True):
    rng = np.random.default_rng(seed)
    truth = np.clip(rng.beta(4, 4, n), 0.05, 0.95)
    y = (rng.random(n) >= truth).astype(int)             # 0 = outcome 0 happened
    meta = pd.DataFrame({"match_id": np.arange(n), "season": 2010 + np.arange(n) // 600,
                         "block": np.arange(n) // 12})
    odds = np.column_stack([0.96 / truth, 0.96 / (1 - truth)])
    model = np.clip(rng.beta(4, 4, n), 0.05, 0.95) if model_noise else truth
    draws = np.clip(model[:, None] + rng.normal(0, 0.02, (n, 50)), 0.01, 0.99)
    cfg = {"combine": {"ridge": [1.0, 100.0]}, "bootstrap": {"n": 300, "seed": 1},
           "selection": {"ev_min": [0.0, 0.04], "p_pos_min": [0.5, 0.9], "lcb_quantile": 0.05, "min_bets": 50},
           "champion": {"market_weight": 0.7, "flag_gap": 0.07}}
    stage = {"first_pred_season": 2010, "first_combo_season": 2012, "first_eval_season": 2014, "threshold_lookback": 6}
    return meta, y, truth, odds, draws, model, stage, cfg


def test_harness_gives_a_noise_model_no_weight_and_no_bets():
    meta, y, truth, odds, draws, model, stage, cfg = _synthetic_market()
    r = evaluate_twoway("ml", meta, y, np.zeros(len(y), bool), truth, odds, draws, stage, cfg, p_b_raw=model)
    w = [abs(v["theta"][1]) for v in r["pool_by_season"].values()]
    assert max(w) < 0.1
    assert r["returns"]["D_combo_rule"]["n"] == 0
    d = r["market"]["log_loss_differences"]
    assert d["C_new minus A_market"]["lo"] > 0               # the noise model is worse than the market
    assert d["B_existing_blend minus A_market"]["lo"] > 0    # and so is a 70/30 blend with it
    assert abs(d["D_combo minus A_market"]["mean"]) < 0.003
    # betting the blend's apparent edges loses about the bookmaker's margin or more
    assert r["returns"]["B_blend_edge_ignoring_flag"]["mean"] < 0


def test_harness_scores_but_does_not_bet_when_no_price_exists():
    meta, y, truth, odds, draws, model, stage, cfg = _synthetic_market(1)
    r = evaluate_twoway("win", meta, y, np.zeros(len(y), bool), truth, odds, draws, stage, cfg, bettable=False)
    assert r["returns"] == {} and r["market"]["n_matches"] > 0


def test_harness_ignores_pushes_in_scores_and_returns_stake():
    meta, y, truth, odds, draws, model, stage, cfg = _synthetic_market(2)
    push = np.zeros(len(y), bool)
    push[::10] = True
    r = evaluate_twoway("spread", meta, y, push, truth, odds, draws, stage, cfg, p_b_raw=model)
    full = evaluate_twoway("spread", meta, y, np.zeros(len(y), bool), truth, odds, draws, stage, cfg, p_b_raw=model)
    assert r["market"]["n_matches"] < full["market"]["n_matches"]


def test_corner_neutral_ufc_data_keeps_every_fight_consistent(tmp_path):
    import pandas as pd
    from challenger.data_us import load_ufc
    raw = pd.DataFrame({
        "R_fighter": [f"R{i}" for i in range(400)], "B_fighter": [f"B{i}" for i in range(400)],
        "R_odds": [-150] * 400, "B_odds": [130] * 400, "date": ["2020-01-04"] * 400, "Winner": ["Red"] * 400,
        "R_age": [30] * 400, "B_age": [25] * 400, "R_Reach_cms": [190] * 400, "B_Reach_cms": [180] * 400})
    f = tmp_path / "ufc.csv"
    raw.to_csv(f, index=False)
    a, b = load_ufc(f), load_ufc(f, neutral_corners=True, seed=1)
    assert a.red_win.mean() == 1.0 and 0.4 < b.red_win.mean() < 0.6     # the corner no longer predicts the winner
    # in every fight the winner is still the same person at the same price
    win_a = set(zip(a.home, a.o_red))
    win_b = set(zip(np.where(b.red_win == 1, b.home, b.away), np.where(b.red_win == 1, b.o_red, b.o_blue)))
    assert win_a == win_b
    older = np.where(b.age_diff > 0, b.home, b.away)
    assert all(x.startswith("R") for x in older)
