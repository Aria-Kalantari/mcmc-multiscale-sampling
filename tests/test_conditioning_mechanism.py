from __future__ import annotations

import numpy as np

from mcmc_multiscale.conditioning.nullspace import project_null
from mcmc_multiscale.conditioning.particular import lu_pivot, svd_min_norm


def _iterate(
    theta_p: np.ndarray, Z: np.ndarray, beta: float, n_iter: int
) -> np.ndarray:
    theta = np.zeros_like(theta_p)
    alpha = np.sqrt(1.0 - beta**2)
    norms = np.empty(n_iter, dtype=np.float64)
    for idx in range(n_iter):
        theta = theta_p + project_null(Z, alpha * theta)
        norms[idx] = np.linalg.norm(theta)
    return norms


def test_repeated_conditioning_accumulates_lu_hidden_null_but_not_svd() -> None:
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)
    theta_lu, _ = lu_pivot(A, c)
    theta_svd, Z, _ = svd_min_norm(A, c)

    assert np.linalg.norm(Z.T @ theta_lu) > 0.5
    np.testing.assert_allclose(Z.T @ theta_svd, 0.0, atol=1e-12)

    lu_norms = _iterate(theta_lu, Z, beta=0.2, n_iter=80)
    svd_norms = _iterate(theta_svd, Z, beta=0.2, n_iter=80)

    assert lu_norms[-1] > 10.0 * lu_norms[0]
    assert np.max(svd_norms) < 2.0 * svd_norms[0]
    assert lu_norms[-1] > 10.0 * svd_norms[-1]
