"""Mathematical functions.
"""

from __future__ import annotations
import numpy as np
import scipy
from numpy.linalg import LinAlgError
from .static_typing import *


def safe_inv(a: ArrayLikeFloat, pdef: bool = False, cr: float = 10):
    """Safer way to compute the (multiplicative) inverse of a matrix.

    Parameters
    ----------
    a : array_like
        Matrix to be inverted.
    pdef : bool, optional
        Is the input matrix positive definite, by default False. If True, checking the eigenvalues
        using :func:`scipy.linalg.svdvals`.
    cr : float, optional
        Criterion, by default 10.

    Raises
    ------
    LinAlgError
        If the matrix is ill-formed.

    See Also
    --------
    numpy.linalg.inv
    """
    ill_formed = np.log10(np.linalg.cond(a)) > cr
    if pdef:
        ill_formed = ill_formed and np.all(scipy.linalg.svdvals(a) > 0)
    if ill_formed:
        raise LinAlgError("Matrix is ill-formed")
    return np.linalg.inv(a)
