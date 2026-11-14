import numpy as np
import pandas as pd

from challenger.champion import blend
from challenger.combine import fit_pool, fit_pool_walk_forward
from challenger.evaluate import block_bootstrap_mean, ece, log_loss
from challenger.predictive import rate_draws, standard_draws, summarize
from challenger.scoregrid import ScoreGrid
from challenger.selection import apply_rule, choose_thresholds, ev_summary

G = ScoreGrid(10)


def test_parameter_uncertainty_is_separated_from_monte_carlo_error():
    z = standard_draws(400, 1)
    assert np.allclose(z.mean(axis=0), 0.0)             # antithetic
    eta = np.array([0.4]), np.array([0.1])
    tight = rate_draws(*eta, np.array([0.001]), np.array([0.001]), np.array([0.0]), z)
    wide = rate_draws(*eta, np.array([0.09]), np.array([0.09]), np.array([0.0]), z)
    pt = summarize(G.p_1x2(G.pmf(*tight))[..., 0])
    pw = summarize(G.p_1x2(G.pmf(*wide))[..., 0])
    assert pw["param_sd"][0] > 5 * pt["param_sd"][0]
    assert np.isclose(pw["mc_se"][0], pw["param_sd"][0] / 20.0)
    # more draws shrink Monte Carlo error but not parameter uncertainty
    z2 = standard_draws(6400, 1)
    pw2 = summarize(G.p_1x2(G.pmf(*rate_draws(*eta, np.array([0.09]), np.array([0.09]), np.array([0.0]), z2)))[..., 0])
    assert pw2["mc_se"][0] < 0.3 * pw["mc_se"][0]
    assert abs(pw2["param_sd"][0] / pw["param_sd"][0] - 1) < 0.15


def test_existing_blend_reproduces_the_sheet_example():
    # sheet, Nov 14: "Blend 70% no-vig 26.8% + 30% sim 36% = 29.6%"
    assert round(float(blend(np.array([0.268]), np.array([0.36]), 0.70)[0]), 3) == 0.296


def test_combination_ignores_an_uninformative_model_and_uses_a_good_one():
    rng = np.random.default_rng(0)
    n = 6000
    truth = rng.dirichlet([4, 2.5, 3], n)
    y = np.array([rng.choice(3, p=p) for p in truth])
    market = truth * rng.lognormal(0, 0.10, (n, 3))
    market /= market.sum(axis=1, keepdims=True)
    noise = rng.dirichlet([3, 3, 3], n)
    w = fit_pool(market, noise, y, ridge=10.0).theta
    assert abs(w[0] - 1) < 0.15 and abs(w[1]) < 0.08
    w2 = fit_pool(market, truth, y, ridge=1.0).theta
    assert w2[1] > 0.4


def test_combination_for_a_season_uses_only_earlier_seasons():
    rng = np.random.default_rng(1)
    n = 3000
    pm = rng.dirichlet([3, 3, 3], n)
    pc = rng.dirichlet([3, 3, 3], n)
    y = np.array([rng.choice(3, p=p) for p in pm])
    seasons = np.repeat([1, 2, 3, 4, 5, 6], n // 6)
    a = fit_pool_walk_forward(pm, pc, y, seasons, 4, [1.0, 10.0]).theta
    y2 = y.copy()
    y2[seasons >= 4] = 0                               # rewrite the target season and later
    b = fit_pool_walk_forward(pm, pc, y2, seasons, 4, [1.0, 10.0]).theta
    assert np.allclose(a, b)


def test_expected_return_summary_and_probability_of_positive_edge():
    p = np.array([[0.50, 0.52, 0.54, 0.56], [0.30, 0.31, 0.32, 0.33]])
    s = ev_summary(p, np.array([1.90, 3.00]))
    assert np.allclose(s["ev"], [0.53 * 1.9 - 1, 0.315 * 3 - 1])
    assert np.allclose(s["p_pos"], [0.5, 0.0])
    assert (s["ev_lo"] <= s["ev"]).all() and (s["ev"] <= s["ev_hi"]).all()


def test_no_selection_when_apparent_edges_are_noise():
    """Fair coin-flip markets at 1.95: any 'edge' is noise, so searching the
    threshold grid must come back with no rule."""
    rng = np.random.default_rng(4)
    n = 6000
    c = pd.DataFrame({
        "match_id": np.arange(n), "market": "1x2", "odds": 1.95, "block": rng.integers(0, 150, n),
        "ev": rng.normal(0.0, 0.05, n), "p_pos": rng.uniform(0, 1, n), "won": rng.random(n) < 0.5})
    c["ret"] = np.where(c.won, 0.95, -1.0)
    thr, lcb, tried = choose_thresholds(c, [0.0, 0.02, 0.04, 0.06], [0.5, 0.7, 0.9], min_bets=100)
    assert tried > 0 and thr is None


def test_selection_keeps_one_pick_per_match_and_market():
    c = pd.DataFrame({"match_id": [1, 1, 2], "market": ["1x2"] * 3, "odds": [2.0, 3.5, 2.2],
                      "ev": [0.05, 0.08, 0.01], "p_pos": [0.9, 0.95, 0.6]})
    assert apply_rule(c, 0.02, 0.8).ev.tolist() == [0.08]


def test_block_bootstrap_is_wider_than_naive_when_results_cluster():
    rng = np.random.default_rng(2)
    block_effect = rng.normal(0, 1, 200)
    blocks = np.repeat(np.arange(200), 20)
    v = block_effect[blocks] + rng.normal(0, 0.2, 4000)
    r = block_bootstrap_mean(v, blocks, 1000, 0)
    naive_half = 1.96 * v.std() / np.sqrt(len(v))
    assert (r["hi"] - r["lo"]) / 2 > 2.5 * naive_half
    assert r["lo"] < block_effect.mean() < r["hi"]


def test_scores_reward_calibration():
    rng = np.random.default_rng(3)
    p = rng.dirichlet([2, 2, 2], 20000)
    y = np.array([rng.choice(3, p=q) for q in p])
    sharp = p ** 2 / (p ** 2).sum(axis=1, keepdims=True)   # over-confident version
    assert log_loss(p, y).mean() < log_loss(sharp, y).mean()
    assert ece(p, y) < ece(sharp, y)
