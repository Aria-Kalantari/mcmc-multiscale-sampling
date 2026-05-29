"""Small diagnostic helpers for sampler experiments."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from mcmc_multiscale.subdomain import Subdomain, interface_jump_rms


def theta_norm(theta: np.ndarray) -> float:
    """Return the Euclidean norm of a coefficient vector."""

    return float(np.linalg.norm(np.asarray(theta, dtype=np.float64)))


def expected_gaussian_norm(dim: int) -> float:
    """Return `E ||X||` for `X ~ N(0, I_dim)`."""

    if dim <= 0:
        raise ValueError("dim must be positive.")
    return float(np.sqrt(2.0) * np.exp(gammaln((dim + 1.0) / 2.0) - gammaln(dim / 2.0)))


def constraint_residual(A: np.ndarray, theta: np.ndarray, c: np.ndarray) -> float:
    """Return relative hard-constraint residual for `A theta = c`."""

    A_arr = np.asarray(A, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    return float(
        np.linalg.norm(A_arr @ theta_arr - c_arr) / max(1.0, np.linalg.norm(c_arr))
    )


def relative_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    """Return `||x - x_ref|| / max(1, ||x_ref||)`."""

    x_arr = np.asarray(x, dtype=np.float64)
    ref_arr = np.asarray(x_ref, dtype=np.float64)
    if x_arr.shape != ref_arr.shape:
        raise ValueError("x and x_ref must have matching shapes.")
    return float(np.linalg.norm(x_arr - ref_arr) / max(1.0, np.linalg.norm(ref_arr)))


def running_acceptance(accepted: np.ndarray) -> np.ndarray:
    """Return cumulative acceptance rates for a Boolean trace."""

    accepted_arr = np.asarray(accepted, dtype=bool)
    if accepted_arr.ndim != 1:
        raise ValueError("accepted must be one-dimensional.")
    if accepted_arr.size == 0:
        return np.empty(0, dtype=np.float64)
    cumulative = np.cumsum(accepted_arr, dtype=np.float64)
    counts = np.arange(1, accepted_arr.size + 1, dtype=np.float64)
    return cumulative / counts


def integrated_autocorr_time(x: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate integrated autocorrelation time for a scalar chain."""

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    n = x_arr.size
    if n < 2:
        return 1.0
    centered = x_arr - np.mean(x_arr)
    var = float(np.dot(centered, centered) / n)
    if var <= 0.0:
        return 1.0
    lag_max = n - 1 if max_lag is None else min(max_lag, n - 1)
    tau = 1.0
    for lag in range(1, lag_max + 1):
        acf = float(np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * var))
        if acf <= 0.0:
            break
        tau += 2.0 * acf
    return float(tau)


def interface_jump(G: np.ndarray, sub: Subdomain) -> float:
    """Return the M1 RMS interface jump diagnostic."""

    return interface_jump_rms(np.asarray(G, dtype=np.float64), sub)
