"""Karhunen-Loeve eigenpair utilities."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def top_eigenpairs(C: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the top `k` covariance eigenpairs in descending order.

    Eigenvectors are Euclidean-orthonormal columns of `Phi`; eigenvalues are
    returned in descending order and tiny negative roundoff is clipped to zero.
    Eigenvector signs are mathematically arbitrary.
    """

    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
        raise ValueError("C must be a square matrix.")
    if k <= 0:
        raise ValueError("k must be positive.")

    C_sym = (C_arr + C_arr.T) / 2.0
    n = C_sym.shape[0]
    k_eff = min(k, n)

    if k_eff < n:
        vals, vecs = eigh(
            C_sym,
            subset_by_index=(n - k_eff, n - 1),
            check_finite=True,
        )
    else:
        vals, vecs = eigh(C_sym, check_finite=True)

    order = np.argsort(vals)[::-1]
    lam = np.maximum(vals[order], 0.0).astype(np.float64, copy=False)
    Phi = vecs[:, order].astype(np.float64, copy=False)
    return Phi, lam
