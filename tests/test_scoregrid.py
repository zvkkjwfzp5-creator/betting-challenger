import numpy as np
import pytest

from challenger.predictive import simulate_scores
from challenger.scoregrid import ScoreGrid

G = ScoreGrid(10)


def test_markets_are_coherent_with_one_score_matrix():
    M = G.pmf(np.array([1.6, 0.9]), np.array([1.1, 1.4]), rho=-0.08)
    assert np.allclose(M.sum(axis=(-2, -1)), 1.0)
    assert np.allclose(G.p_1x2(M).sum(axis=-1), 1.0)
    over = G.p_over(M, 2.5)
    under = G.p_event(M, (G.total < 2.5).astype(float))
    assert np.allclose(over + under, 1.0)
    # home -0.5 handicap wins exactly when the home side wins
    win = G.expected_return(M, (G.ah_payoff(-0.5, 2.0, "home") > 0).astype(float))
    assert np.allclose(win, G.p_1x2(M)[..., 0])


def test_joint_probability_is_computed_directly_not_as_a_product():
    M = G.pmf(np.array(1.8), np.array(1.0), 0.0)
    home, over = G.p_1x2(M)[0], G.p_over(M, 2.5)
    joint = G.p_event(M, G.m_home * (G.total > 2.5))
    assert joint <= min(home, over) + 1e-12
    assert joint >= home + over - 1 - 1e-12
    assert abs(joint - home * over) > 0.01          # the legs are not independent
    rng = np.random.default_rng(1)
    h, a = simulate_scores(1.8, 1.0, 200_000, rng, G)
    assert abs(((h > a) & (h + a > 2.5)).mean() - joint) < 0.004


def test_inner_simulation_matches_exact_conditional_probabilities():
    rng = np.random.default_rng(3)
    n = 200_000
    h, a = simulate_scores(1.4, 1.2, n, rng, G, rho=-0.1)
    M = G.pmf(np.array(1.4), np.array(1.2), -0.1)
    exact = G.p_1x2(M)
    sim = np.array([(h > a).mean(), (h == a).mean(), (h < a).mean()])
    se = np.sqrt(exact * (1 - exact) / n)
    assert (np.abs(sim - exact) < 4 * se).all()


@pytest.mark.parametrize("line,diff,expected", [
    (-1.0, 1, 0.0),          # push: stake returned
    (-1.0, 2, 0.9),          # win
    (-0.75, 1, 0.45),        # half win
    (-0.25, 0, -0.5),        # half loss
    (+0.25, 0, 0.45),        # half win on the draw
    (-0.5, 0, -1.0),         # loss
])
def test_asian_handicap_settlement(line, diff, expected):
    pay = G.ah_payoff(line, 1.9, "home")
    assert pay[diff if diff >= 0 else 0, 0 if diff >= 0 else -diff] == pytest.approx(expected)


def test_total_push_and_fair_price_have_zero_expected_return():
    assert G.total_payoff(3.0, 1.95, "over")[2, 1] == 0.0
    M = G.pmf(np.array(1.5), np.array(1.2), 0.0)
    p = G.p_over(M, 2.5)
    assert G.expected_return(M, G.total_payoff(2.5, 1 / p, "over")) == pytest.approx(0.0, abs=1e-12)
    # both sides of a handicap at fair prices: zero expected return each
    for side in ("home", "away"):
        pay1 = G.ah_payoff(-0.25, 2.0, side)
        win = G.expected_return(M, np.clip(pay1, 0, None)) / 1.0     # expected winnings at odds 2.0
        loss = -G.expected_return(M, np.clip(pay1, None, 0))
        fair = 1 + loss / win
        assert G.expected_return(M, G.ah_payoff(-0.25, fair, side)) == pytest.approx(0.0, abs=1e-9)
