"""Null-space helpers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def null_basis(A: np.ndarray, rtol: float | None = None) -> np.ndarray:
    """Return an orthonormal basis for `Null(A)`.

    The default rank tolerance mirrors the MATLAB static reference:
    `max(size(A)) * eps(max(singular_values))`.
    """

    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.ndim != 2:
        raise ValueError("A must be two-dimensional.")
    _, s, vh = svd(A_arr, full_matrices=True, check_finite=True)
    n_cols = A_arr.shape[1]
    if s.size == 0:
        rank = 0
    else:
        tol = (
            float(rtol) * s[0]
            if rtol is not None
            else max(A_arr.shape) * np.spacing(float(s[0]))
        )
        rank = int(np.sum(s > tol))
    return vh[rank:, :].T[:, : n_cols - rank].astype(np.float64, copy=False)


def project_null(Z: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Project a full-space vector onto `Range(Z) = Null(A)`."""

    Z_arr = np.asarray(Z, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)
    if Z_arr.ndim != 2:
        raise ValueError("Z must be two-dimensional.")
    if eta_arr.shape != (Z_arr.shape[0],):
        raise ValueError("eta must live in the full coefficient space.")
    return Z_arr @ (Z_arr.T @ eta_arr)
