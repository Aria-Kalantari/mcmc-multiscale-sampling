from __future__ import annotations

import numpy as np

from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs


def test_top_eigenpairs_are_descending_and_orthonormal() -> None:
    _, _, _, _, points = cell_centered_grid(6, 5)
    C = exp_covariance(points, sigma=1.0, ell=0.18)

    Phi, lam = top_eigenpairs(C, 8)

    assert Phi.shape == (30, 8)
    assert lam.shape == (8,)
    assert np.all(np.diff(lam) <= 0.0)
    np.testing.assert_allclose(Phi.T @ Phi, np.eye(8), atol=1e-12)


def test_truncated_reconstruction_is_sign_agnostic_projector() -> None:
    _, _, _, _, points = cell_centered_grid(5, 5)
    C = exp_covariance(points, sigma=1.0, ell=0.22)

    Phi, lam = top_eigenpairs(C, 6)
    projector = Phi @ Phi.T
    signed_projector = (Phi * np.array([1, -1, 1, -1, 1, -1])) @ (
        Phi * np.array([1, -1, 1, -1, 1, -1])
    ).T

    np.testing.assert_allclose(projector, signed_projector, atol=1e-12)
    assert np.all(lam >= 0.0)
