import numpy as np
import pytest

from challenger.market import DEVIG, overround


@pytest.mark.parametrize("name", list(DEVIG))
def test_devig_is_a_probability_distribution(name):
    odds = np.array([[1.5, 4.2, 7.0], [2.6, 3.3, 2.75], [1.1, 9.0, 21.0]])
    p = DEVIG[name](odds)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert (p > 0).all()
    # ranking of outcomes is preserved
    assert (np.argsort(p, axis=1) == np.argsort(1 / odds, axis=1)).all()


@pytest.mark.parametrize("name", list(DEVIG))
def test_devig_leaves_fair_prices_unchanged(name):
    p = np.array([[0.5, 0.3, 0.2], [0.62, 0.38, 0.0 + 1e-9]])[:1]
    assert np.allclose(DEVIG[name](1 / p), p, atol=1e-6)


def test_power_and_shin_shade_longshots_more_than_multiplicative():
    odds = np.array([[1.25, 6.0, 11.0]])
    assert overround(odds)[0] > 0
    m = DEVIG["multiplicative"](odds)[0]
    for name in ("power", "shin"):
        p = DEVIG[name](odds)[0]
        assert p[2] < m[2] and p[0] > m[0]
