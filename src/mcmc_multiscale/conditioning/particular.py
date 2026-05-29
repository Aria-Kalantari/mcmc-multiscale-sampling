"""Particular solutions for hard conditioning systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import svd


@dataclass(frozen=True)
class LinearInfo:
    """Linear diagnostics for `A theta = c`."""

    rankA: int
    singular_values: np.ndarray
    min_nonzero_singular: float
    cond_effective: float


def svd_min_norm(
    A: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, LinearInfo]:
    """Solve `A theta = c` with the minimum-norm SVD particular solution.

    With `A = U S V.T`, this returns
    `theta_p = V_r @ ((U_r.T @ c) / s_r)`, together with an orthonormal basis
    `Z` for `Null(A)`. The rank tolerance mirrors the MATLAB reference:
    `max(size(A)) * eps(max(s))`.
    """

    A_arr = np.asarray(A, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    if A_arr.ndim != 2:
        raise ValueError("A must be two-dimensional.")
    if c_arr.shape != (A_arr.shape[0],):
        raise ValueError("c must have one entry per row of A.")

    U, s, vh = svd(A_arr, full_matrices=True, check_finite=True)
    if s.size == 0:
        raise ValueError("A has no singular values.")

    tol = max(A_arr.shape) * np.spacing(float(s[0]))
    r = int(np.sum(s > tol))
    if r == 0:
        raise ValueError("Numerical rank of A is zero.")

    V_r = vh[:r, :].T
    theta_p = V_r @ ((U[:, :r].T @ c_arr) / s[:r])
    Z = vh[r:, :].T

    info = LinearInfo(
        rankA=r,
        singular_values=s.astype(np.float64, copy=False),
        min_nonzero_singular=float(s[r - 1]),
        cond_effective=float(s[0] / s[r - 1]),
    )
    return (
        theta_p.astype(np.float64, copy=False),
        Z.astype(np.float64, copy=False),
        info,
    )


def lu_pivot(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for the arbitrary LU/pivot construction.

    # PHASE2/M4: implement the pivot-column particular solution used to
    reproduce the coefficient-norm instability. It is intentionally not used in
    M1, where only the validated SVD minimum-norm route is in scope.
    """

    raise NotImplementedError("lu_pivot is reserved for Phase 2/M4.")
