"""Particular solutions for hard conditioning systems."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, svd


@dataclass(frozen=True)
class LinearInfo:
    """Linear diagnostics for `A theta = c`."""

    rankA: int
    singular_values: np.ndarray
    min_nonzero_singular: float
    cond_effective: float
    pivot_columns: np.ndarray | None = None
    residual: float | None = None
    cond_B: float | None = None


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


def lu_pivot(A: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, LinearInfo]:
    """Return an arbitrary pivot-column particular solution.

    This M4 path intentionally does not stabilize the solution or remove hidden
    null-space content. Pivot columns are selected by QR with column pivoting,
    the square pivot block is solved by LU factorization, and all non-pivot
    coordinates are set to zero.
    """

    A_arr = np.asarray(A, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    if A_arr.ndim != 2:
        raise ValueError("A must be two-dimensional.")
    if c_arr.shape != (A_arr.shape[0],):
        raise ValueError("c must have one entry per row of A.")

    _, s, _ = svd(A_arr, full_matrices=True, check_finite=True)
    if s.size == 0:
        raise ValueError("A has no singular values.")
    tol = max(A_arr.shape) * np.spacing(float(s[0]))
    r = int(np.sum(s > tol))
    n_rows, n_cols = A_arr.shape
    if r != n_rows:
        raise ValueError(
            "lu_pivot requires A to have full row rank for hard constraints."
        )

    _, _, pivots = qr(A_arr, mode="economic", pivoting=True, check_finite=True)
    pivot_columns = np.asarray(pivots[:r], dtype=np.int64)
    B = A_arr[:, pivot_columns]
    lu, lu_piv = lu_factor(B, check_finite=True)
    theta_piv = lu_solve((lu, lu_piv), c_arr, check_finite=True)

    theta_p = np.zeros(n_cols, dtype=np.float64)
    theta_p[pivot_columns] = theta_piv
    residual = np.linalg.norm(A_arr @ theta_p - c_arr) / max(1.0, np.linalg.norm(c_arr))

    info = LinearInfo(
        rankA=r,
        singular_values=s.astype(np.float64, copy=False),
        min_nonzero_singular=float(s[r - 1]),
        cond_effective=float(s[0] / s[r - 1]),
        pivot_columns=pivot_columns,
        residual=float(residual),
        cond_B=float(np.linalg.cond(B)),
    )
    return theta_p, info
