from __future__ import annotations

import numpy as np

from mcmc_multiscale.bayes import log_posterior, log_prior, misfit
from mcmc_multiscale.config import Config
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs
from mcmc_multiscale.observations import make_truth


def _small_truth_with_kle() -> tuple[Config, np.ndarray, np.ndarray, object]:
    cfg = Config(nx=8, ny=8, n_global_modes=10, n_obs_x=3, n_obs_y=3, seed=11)
    truth = make_truth(cfg)
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    C = exp_covariance(points, cfg.sigma, cfg.corr_length)
    Phi, lam = top_eigenpairs(C, cfg.n_global_modes)
    return cfg, Phi, lam, truth


def test_noiseless_truth_misfit_is_zero() -> None:
    cfg, Phi, lam, truth = _small_truth_with_kle()

    value = misfit(
        truth.theta_true,
        Phi,
        lam,
        ForwardModel(cfg),
        truth.y_clean,
        truth.sensor_idx,
        cfg.sigma_obs,
        cfg.ny,
        cfg.nx,
    )

    assert value < 1e-20


def test_log_prior_convention_and_ordering() -> None:
    zero = np.zeros(5, dtype=np.float64)
    theta = np.ones(5, dtype=np.float64)

    assert log_prior(zero) == 0.0
    assert log_prior(theta) < log_prior(zero)


def test_log_posterior_is_finite_for_valid_theta() -> None:
    cfg, Phi, lam, truth = _small_truth_with_kle()

    value = log_posterior(
        truth.theta_true,
        Phi,
        lam,
        ForwardModel(cfg),
        truth.y_obs,
        truth.sensor_idx,
        cfg.sigma_obs,
        cfg.ny,
        cfg.nx,
    )

    assert np.isfinite(value)
