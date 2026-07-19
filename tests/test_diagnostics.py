from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.diagnostics import (
    constraint_residual,
    expected_gaussian_norm,
    gelman_rubin,
    integrated_autocorr_time,
    interface_jump,
    mpsrf,
    relative_error,
    running_acceptance,
    theta_norm,
)
from mcmc_multiscale.subdomain import make_subdomain


def test_theta_norm_and_expected_gaussian_norm() -> None:
    assert theta_norm(np.array([3.0, 4.0], dtype=np.float64)) == 5.0
    assert abs(expected_gaussian_norm(1) - np.sqrt(2.0 / np.pi)) < 1e-14


def test_constraint_residual_exact_solution() -> None:
    A = np.array([[1.0, 2.0]], dtype=np.float64)
    theta = np.array([1.0, 2.0], dtype=np.float64)
    c = np.array([5.0], dtype=np.float64)

    assert constraint_residual(A, theta, c) == 0.0


def test_relative_error_and_running_acceptance() -> None:
    x = np.array([2.0, 2.0], dtype=np.float64)
    ref = np.array([1.0, 2.0], dtype=np.float64)
    assert abs(relative_error(x, ref) - (1.0 / np.sqrt(5.0))) < 1e-14

    accepted = np.array([True, False, True, True])
    np.testing.assert_allclose(
        running_acceptance(accepted), [1.0, 0.5, 2.0 / 3.0, 0.75]
    )


def test_integrated_autocorr_time_is_finite_for_simple_chain() -> None:
    x = np.sin(np.linspace(0.0, 4.0 * np.pi, 100, dtype=np.float64))
    tau = integrated_autocorr_time(x, max_lag=20)

    assert np.isfinite(tau)
    assert tau >= 1.0


def test_interface_jump_wrapper() -> None:
    cfg = Config(nx=8, ny=8, n_coarse_x=4, n_coarse_y=4, overlap_cells=1)
    sub = make_subdomain(cfg)
    G = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)

    assert interface_jump(G, sub) == 0.0


def test_mpsrf_reduces_to_gelman_rubin() -> None:
    # For a single parameter, MPSRF must equal the scalar Gelman-Rubin exactly.
    for seed in (0, 1, 7, 42):
        rng = np.random.default_rng(seed)
        m, n = 5, 400
        chains = np.empty((m, n), dtype=np.float64)
        for j in range(m):
            chains[j] = rng.normal(loc=0.3 * j, scale=1.0 + 0.2 * j, size=n)
        assert abs(mpsrf(chains[:, :, None]) - gelman_rubin(chains)) < 1e-12


def test_mpsrf_wellmixed_near_one() -> None:
    rng = np.random.default_rng(3)
    chains = rng.standard_normal((6, 4000, 3))
    assert 0.9 < mpsrf(chains) < 1.05


def test_mpsrf_offset_greater_than_one() -> None:
    rng = np.random.default_rng(5)
    m, n, p = 4, 500, 2
    shifts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    chains = np.empty((m, n, p), dtype=np.float64)
    for j in range(m):
        chains[j] = rng.standard_normal((n, p)) + shifts[j]
    assert mpsrf(chains) > 1.2


def test_mpsrf_invariant_under_common_linear_map() -> None:
    # MPSRF is a generalized-eigenvalue statistic of the pencil (V_hat, W); a
    # common invertible reparameterization x -> A x leaves it unchanged.
    rng = np.random.default_rng(11)
    m, n, p = 4, 600, 3
    scales = np.array([1.0, 0.3, 2.0])
    shifts = rng.standard_normal((m, p))
    chains = np.empty((m, n, p), dtype=np.float64)
    for j in range(m):
        chains[j] = rng.standard_normal((n, p)) * scales + shifts[j]
    A = rng.standard_normal((p, p)) + p * np.eye(p)  # well-conditioned, invertible
    mapped = chains @ A.T
    assert abs(mpsrf(chains) - mpsrf(mapped)) < 1e-8


def test_mpsrf_raises_when_samples_not_exceed_params() -> None:
    with pytest.raises(ValueError, match="n_samples > n_params"):
        mpsrf(np.zeros((4, 3, 5), dtype=np.float64))


def test_mpsrf_validates_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        mpsrf(np.zeros((4, 10), dtype=np.float64))
    with pytest.raises(ValueError, match="at least two chains"):
        mpsrf(np.zeros((1, 10, 2), dtype=np.float64))
