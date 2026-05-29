"""Bayesian inverse-problem utilities."""

from __future__ import annotations

import numpy as np

from mcmc_multiscale.field import (
    field_from_theta,
    permeability_from_log_field,
    reshape_field,
)
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.observations import restrict_pressure

GlobalKLE = tuple[np.ndarray, np.ndarray]


def log_prior(theta: np.ndarray) -> float:
    """Return the standard-normal log prior up to an additive constant."""

    theta_arr = np.asarray(theta, dtype=np.float64)
    return float(-0.5 * np.dot(theta_arr, theta_arr))


def log_prior_field(G_vec: np.ndarray, global_kle: GlobalKLE) -> float:
    """Return the global-KLE log prior of a log-field vector.

    Projects `G_vec` onto the orthonormal global KLE basis and evaluates the
    standard-normal coefficient prior, omitting additive constants.
    """

    Phi, lam = global_kle
    G_arr = np.asarray(G_vec, dtype=np.float64)
    Phi_arr = np.asarray(Phi, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    if Phi_arr.ndim != 2:
        raise ValueError("global KLE Phi must be two-dimensional.")
    if G_arr.shape != (Phi_arr.shape[0],):
        raise ValueError("G_vec must have one entry per global KLE row.")
    if lam_arr.shape != (Phi_arr.shape[1],):
        raise ValueError("global KLE lam must have one entry per Phi column.")
    if np.any(lam_arr <= 0.0):
        raise ValueError("global KLE eigenvalues must be positive.")

    theta_global = (Phi_arr.T @ G_arr) / np.sqrt(lam_arr)
    return log_prior(theta_global)


def misfit(
    theta: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    fwd: ForwardModel,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_obs: float,
    ny: int,
    nx: int,
) -> float:
    """Return `0.5 / sigma_obs**2 * ||R(p(theta)) - y_obs||_2**2`."""

    if sigma_obs <= 0.0:
        raise ValueError("sigma_obs must be positive.")
    G_vec = field_from_theta(Phi, lam, theta)
    G = reshape_field(G_vec, ny, nx)
    k = permeability_from_log_field(G)
    p = fwd.solve(k)
    pred = restrict_pressure(p, sensor_idx)
    residual = pred - np.asarray(y_obs, dtype=np.float64)
    return float(0.5 / sigma_obs**2 * np.dot(residual, residual))


def log_likelihood(
    theta: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    fwd: ForwardModel,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_obs: float,
    ny: int,
    nx: int,
) -> float:
    """Return the Gaussian log likelihood up to an additive constant."""

    return -misfit(theta, Phi, lam, fwd, y_obs, sensor_idx, sigma_obs, ny, nx)


def log_posterior(
    theta: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    fwd: ForwardModel,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_obs: float,
    ny: int,
    nx: int,
) -> float:
    """Return log prior plus log likelihood, omitting additive constants."""

    return log_prior(theta) + log_likelihood(
        theta, Phi, lam, fwd, y_obs, sensor_idx, sigma_obs, ny, nx
    )
