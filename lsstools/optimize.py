"""Fitting methods.
"""

from __future__ import annotations
import numpy as np
import scipy.optimize
from numpy import newaxis
from numpy.polynomial import Polynomial
from .static_typing import *


def lsq_linear_constr(
    A: ArrayLikeFloat,
    b: ArrayLikeFloat,
    L: ArrayLikeFloat,
    d: ArrayLikeFloat | None = None,
    full_result: bool = False,
    **kwargs,
) -> scipy.optimize.OptimizeResult:
    R"""Solve a linear least-squares problem with linear constraints.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Design matrix.
    b : array_like, shape (m,)
        Target vector.
    L : array_like, shape (n,) or (p, n)
        Linear constraint.
    d : array_like, shape (p,), optional
        Linear constraint constant, by default zeros.
    full_result : bool, optional
        Return full result if True, by default return truncated result.
    **kwargs : dict, optional
        Extra arguments to :func:`scipy.optimize.lsq_linear`.

    Returns
    -------
    :class:`scipy.optimize.OptimizeResult`

    Warnings
    --------
    Even if :obj:`full_result` is set to :obj:`True`, :attr:`res.unbounded_sol` is unchanged in order
    to save some computational time.

    See Also
    --------
    :func:`scipy.optimize.lsq_linear`

    Notes
    -----
    This function solves the following equation:

    .. math::
        \underset{L\beta=d}{\mathrm{argmin}}||b-A\beta||^{2}
    """
    A = np.asarray(A)
    L = np.atleast_2d(L)
    n_params = A.shape[-1]
    n_nuisance = L.shape[0]
    if d is not None:
        x0 = np.linalg.pinv(L) @ np.atleast_1d(d)
        b = np.asarray(b) - A @ x0
    z = np.zeros((L.shape[0], L.shape[0]))
    # fmt: off
    mat = np.block([[A.T @ A, L.T],
                    [      L,   z]])
    # fmt: on
    vec = np.hstack([A.T @ b, np.zeros(n_nuisance)])
    res: scipy.optimize.OptimizeResult = scipy.optimize.lsq_linear(mat, vec, **kwargs)
    if not full_result:
        res.x = res.x[:n_params]
        res.fun = res.fun[: A.shape[0]]
        res.active_mask = res.active_mask[:n_params]
    if d is not None:
        res.x += np.linalg.pinv(L) @ np.atleast_1d(d)
    return res


def lsq_polynomial_fit(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    deg: ArrayLikeInt,
    roots: Sequence[tuple[float, float]] | None = None,
    full_output: bool = False,
    **kwargs,
) -> Any:
    """Polynomial fit with linear least-squares.

    Parameters
    ----------
    x : array_like, shape (M,)
        :math:`x`-coordinates of the M sample points.
    y : array_like, shape (M,)
        :math:`y`-coordinates of the M sample points.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If *deg* is a single integer all terms up to and
        including the *deg*'th term are included in the fit. A list of integers specifying the
        degrees of the terms to include may be used instead.
    roots : Sequence[tuple[float, float]], optional
        The roots of polynomials, by default None.
    full_output : bool, optional
        Return :class:`scipy.optimize.OptimizeResult` as well, by default False.
    **kwargs : dict, optional
        Extra arguments to :func:`scipy.optimize.lsq_linear`.

    Returns
    -------
    poly : Polynomial
        Polynomial fits.
    result : OptimizeResult (returned only if :obj:`full_output` is True)

    Raises
    ------
    ValueError
        If the provided :obj:`roots` are already able to determine the polynomial.
    """
    if isinstance(deg, Sequence):
        deg = sorted(deg)  # type: ignore
    else:
        deg = list(range(deg + 1))
    roots = roots or []
    if len(roots) > len(deg):
        raise ValueError("polynomial is fully determined by roots")
    basis: NDArrayFloat = np.array(deg)
    A: NDArrayFloat = np.asarray(x)[:, newaxis] ** basis[newaxis, :]
    if not roots:
        res = scipy.optimize.lsq_linear(A, y, **kwargs)
    else:
        L = np.array([_[0] for _ in roots])[:, newaxis] ** basis[newaxis, :]
        d = [_[-1] for _ in roots]
        res = lsq_linear_constr(A, y, L, d, full_result=False, **kwargs)
    coef = res.x
    _result: list = [None] * (deg[-1] + 1)
    idx = 0
    for i in range(deg[-1] + 1):
        if deg[idx] == i:
            _result[i] = coef[idx]
            idx += 1
        else:
            _result[i] = 0.0
    if full_output:
        return Polynomial(_result), res
    return Polynomial(_result)
