from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.conditioning.nullspace import null_basis
from mcmc_multiscale.conditioning.particular import lu_pivot, svd_min_norm


def test_lu_pivot_solves_short_wide_system_and_zeros_nonpivots() -> None:
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)

    theta_p, info = lu_pivot(A, c)

    np.testing.assert_allclose(A @ theta_p, c, atol=1e-14)
    assert info.rankA == A.shape[0]
    assert info.pivot_columns is not None
    nonpivots = np.setdiff1d(np.arange(A.shape[1]), info.pivot_columns)
    np.testing.assert_allclose(theta_p[nonpivots], 0.0, atol=0.0)
    assert info.residual is not None
    assert info.residual < 1e-14
    assert info.cond_B is not None
    assert info.cond_B > 1.0


def test_lu_pivot_has_hidden_null_content_unlike_svd_min_norm() -> None:
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)

    theta_lu, _ = lu_pivot(A, c)
    theta_svd, Z, _ = svd_min_norm(A, c)

    assert np.linalg.norm(theta_lu - theta_svd) > 0.5
    assert np.linalg.norm(Z.T @ theta_lu) > 0.5
    np.testing.assert_allclose(Z.T @ theta_svd, 0.0, atol=1e-12)
    np.testing.assert_allclose(A @ theta_lu, c, atol=1e-14)

    Z_from_null = null_basis(A)
    np.testing.assert_allclose(np.abs(Z_from_null.T @ theta_lu), np.abs(Z.T @ theta_lu))


def test_lu_pivot_rejects_rank_deficient_constraints() -> None:
    A = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="full row rank"):
        lu_pivot(A, c)
