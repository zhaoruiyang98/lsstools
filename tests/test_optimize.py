import numpy as np
import pytest
from lsstools.optimize import (
    lsq_linear_constr,
    lsq_polynomial_fit,
)


def test_lsq_linear_constr():
    # fmt: off
    A = [[ 3, 2, 1],
         [-4, 0, 3]]
    # fmt: on
    b = [-1, 2]
    L = [1, 1, 1]
    d = [3]
    res = lsq_linear_constr(A, b, L, d)
    assert np.allclose(res.x, [19, -42, 26])


def test_lsq_polynomial_fit():
    c0, c1, c4 = -1, 2, 3
    x = np.linspace(0, 1, 10_000)
    y = c0 + c1 * x + c4 * x**4 + np.random.standard_normal(x.size) / 100
    poly = lsq_polynomial_fit(x, y, deg=[0, 1, 4], roots=[(1, c0 + c1 + c4), (0, c0)])
    assert np.allclose(poly.coef[0], c0, rtol=1e-2, atol=1e-2)
    assert np.allclose(np.sum(poly.coef), c0 + c1 + c4, rtol=1e-2, atol=1e-2)
    assert np.allclose(poly.coef, [c0, c1, 0, 0, c4], rtol=1e-2, atol=1e-2)

    poly = lsq_polynomial_fit(x, y, deg=[0, 1, 4], roots=[(1, c0 + c1 + c4)], deriv_roots=[(0, c1)])
    assert np.allclose(np.sum(poly.coef), c0 + c1 + c4, rtol=1e-2, atol=1e-2)
    assert np.allclose(poly.coef[1], c1, rtol=1e-2, atol=1e-2)
    assert np.allclose(poly.coef, [c0, c1, 0, 0, c4], rtol=1e-2, atol=1e-2)
