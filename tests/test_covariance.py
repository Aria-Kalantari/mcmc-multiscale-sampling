from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.covariance import exp_covariance, sqexp_covariance
from mcmc_multiscale.grid import cell_centered_grid


def test_exp_covariance_symmetry_diagonal_and_positive_spectrum() -> None:
    _, _, _, _, points = cell_centered_grid(5, 4)
    sigma = 1.7
    C = exp_covariance(points, sigma=sigma, ell=0.25)

    np.testing.assert_allclose(C, C.T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.diag(C), sigma**2 + 1e-12, rtol=1e-14, atol=1e-14)

    evals = np.linalg.eigvalsh(C)
    assert np.min(evals) > -1e-10


def test_sqexp_covariance_symmetry_diagonal_and_positive_spectrum() -> None:
    _, _, _, _, points = cell_centered_grid(5, 4)
    sigma = 1.7
    C = sqexp_covariance(points, sigma=sigma, ell=0.25)

    np.testing.assert_allclose(C, C.T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.diag(C), sigma**2 + 1e-12, rtol=1e-14, atol=1e-14)

    evals = np.linalg.eigvalsh(C)
    assert np.min(evals) > -1e-10


def test_sqexp_covariance_two_point_value() -> None:
    # Two points a distance 0.5 apart: C_01 = sigma**2 * exp(-0.5 * d**2 / ell**2).
    points = np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float64)
    sigma, ell = 2.0, 0.5
    C = sqexp_covariance(points, sigma=sigma, ell=ell)

    expected = sigma**2 * np.exp(-0.5 * 0.5**2 / ell**2)
    assert abs(C[0, 1] - expected) < 1e-14
    assert abs(C[0, 1] - 4.0 * np.exp(-0.5)) < 1e-14  # hand-checked literal
    assert abs(C[1, 0] - expected) < 1e-14


def test_sqexp_covariance_validates_inputs() -> None:
    _, _, _, _, points = cell_centered_grid(3, 3)
    with pytest.raises(ValueError, match="ell must be positive"):
        sqexp_covariance(points, sigma=1.0, ell=0.0)
    with pytest.raises(ValueError, match="points must have shape"):
        sqexp_covariance(np.zeros((4, 3)), sigma=1.0, ell=0.2)
