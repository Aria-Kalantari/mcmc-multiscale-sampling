from __future__ import annotations

import numpy as np

from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.grid import cell_centered_grid


def test_exp_covariance_symmetry_diagonal_and_positive_spectrum() -> None:
    _, _, _, _, points = cell_centered_grid(5, 4)
    sigma = 1.7
    C = exp_covariance(points, sigma=sigma, ell=0.25)

    np.testing.assert_allclose(C, C.T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.diag(C), sigma**2 + 1e-12, rtol=1e-14, atol=1e-14)

    evals = np.linalg.eigvalsh(C)
    assert np.min(evals) > -1e-10
