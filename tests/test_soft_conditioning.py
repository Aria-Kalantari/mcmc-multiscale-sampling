from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.conditioning.particular import svd_min_norm
from mcmc_multiscale.conditioning.project import affine_project
from mcmc_multiscale.conditioning.soft import (
    soft_min_norm_particular,
    soft_project,
)
from mcmc_multiscale.diagnostics import constraint_residual


def _system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)
    theta = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    return A, c, theta


@pytest.mark.parametrize("rho", [0.0, -1.0, np.nan])
def test_soft_conditioning_rejects_invalid_rho(rho: float) -> None:
    A, c, theta = _system()
    with pytest.raises(ValueError, match="rho"):
        soft_min_norm_particular(A, c, rho)
    with pytest.raises(ValueError, match="rho"):
        soft_project(theta, A, c, rho)


def test_soft_min_norm_residual_decreases_as_rho_increases() -> None:
    A, c, _ = _system()
    theta_small = soft_min_norm_particular(A, c, rho=1.0)
    theta_large = soft_min_norm_particular(A, c, rho=1.0e4)

    assert np.all(np.isfinite(theta_small))
    assert np.all(np.isfinite(theta_large))
    assert constraint_residual(A, theta_large, c) < constraint_residual(
        A, theta_small, c
    )


def test_soft_project_residual_decreases_and_small_rho_stays_closer() -> None:
    A, c, theta = _system()
    projected_small = soft_project(theta, A, c, rho=0.1)
    projected_large = soft_project(theta, A, c, rho=1.0e4)

    assert np.all(np.isfinite(projected_small))
    assert np.all(np.isfinite(projected_large))
    assert constraint_residual(A, projected_large, c) < constraint_residual(
        A, projected_small, c
    )
    assert np.linalg.norm(projected_small - theta) < np.linalg.norm(
        projected_large - theta
    )


def test_large_rho_soft_project_approaches_hard_affine_projection() -> None:
    A, c, theta = _system()
    theta_p, Z, _ = svd_min_norm(A, c)
    hard = affine_project(theta, theta_p, Z)
    soft = soft_project(theta, A, c, rho=1.0e8)

    np.testing.assert_allclose(soft, hard, atol=1e-6, rtol=1e-6)
