from __future__ import annotations

import numpy as np

from mcmc_multiscale.conditioning.particular import lu_pivot, svd_min_norm
from mcmc_multiscale.conditioning.project import stabilize


def test_stabilize_preserves_constraints_and_removes_null_component() -> None:
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)
    theta_lu, _ = lu_pivot(A, c)
    theta_svd, Z, _ = svd_min_norm(A, c)

    theta_stable = stabilize(theta_lu, Z)

    np.testing.assert_allclose(A @ theta_stable, c, atol=1e-12)
    np.testing.assert_allclose(Z.T @ theta_stable, 0.0, atol=1e-12)
    assert np.linalg.norm(theta_stable) <= np.linalg.norm(theta_lu)
    np.testing.assert_allclose(theta_stable, theta_svd, atol=1e-12)
