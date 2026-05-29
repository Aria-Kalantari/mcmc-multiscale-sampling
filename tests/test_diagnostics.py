from __future__ import annotations

import numpy as np

from mcmc_multiscale.config import Config
from mcmc_multiscale.diagnostics import (
    constraint_residual,
    expected_gaussian_norm,
    integrated_autocorr_time,
    interface_jump,
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
