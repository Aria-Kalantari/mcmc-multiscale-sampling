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
    """Estimate scalar-chain IAT with Geyer initial-monotone truncation.

    Positive adjacent autocorrelation sums are retained until the first
    non-positive pair. Retained pair sums are monotonized downward before
    evaluating `tau = 1 + 2 * sum(rho_k)`.
    """

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite values.")
    n = x_arr.size
    if n < 2:
        return 1.0
    centered = x_arr - np.mean(x_arr)
    var = float(np.dot(centered, centered) / n)
    if var <= 0.0:
        return 1.0
    lag_max = n - 1 if max_lag is None else min(max_lag, n - 1)
    if lag_max < 1:
        return 1.0

    rho = np.empty(lag_max, dtype=np.float64)
    for lag in range(1, lag_max + 1):
        rho[lag - 1] = np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * var)

    pair_sums: list[float] = []
    for first in range(0, lag_max, 2):
        pair_sum = float(rho[first])
        if first + 1 < lag_max:
            pair_sum += float(rho[first + 1])
        if pair_sum <= 0.0:
            break
        if pair_sums:
            pair_sum = min(pair_sum, pair_sums[-1])
        pair_sums.append(pair_sum)
    return float(max(1.0, 1.0 + 2.0 * sum(pair_sums)))


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """Return scalar-chain effective sample size `N / tau`."""

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if x_arr.size == 0:
        return 0.0
    return float(x_arr.size / integrated_autocorr_time(x_arr, max_lag=max_lag))


def gelman_rubin(chains: np.ndarray) -> float:
    """Return split-free Gelman-Rubin `R_hat` for scalar chains.

    Input has shape `(n_chains, n_samples)`. The estimator uses between-chain
    variance `B`, within-chain variance `W`, and
    `V_hat = ((n - 1) / n) * W + B / n`.
    """

    chains_arr = np.asarray(chains, dtype=np.float64)
    if chains_arr.ndim != 2:
        raise ValueError("chains must have shape (n_chains, n_samples).")
    n_chains, n_samples = chains_arr.shape
    if n_chains < 2:
        raise ValueError("gelman_rubin requires at least two chains.")
    if n_samples < 2:
        raise ValueError("gelman_rubin requires at least two samples per chain.")
    if not np.all(np.isfinite(chains_arr)):
        raise ValueError("chains must contain only finite values.")

    chain_means = np.mean(chains_arr, axis=1)
    within = float(np.mean(np.var(chains_arr, axis=1, ddof=1)))
    between = float(n_samples * np.var(chain_means, ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else np.inf
    variance_hat = ((n_samples - 1.0) / n_samples) * within + between / n_samples
    return float(np.sqrt(max(0.0, variance_hat / within)))


def posterior_summary(field_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and sample standard deviation for field samples."""

    samples = np.asarray(field_samples, dtype=np.float64)
    if samples.ndim < 2:
        raise ValueError("field_samples must have shape (n_samples, ...).")
    if samples.shape[0] < 1:
        raise ValueError("field_samples must contain at least one sample.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("field_samples must contain only finite values.")
    std = (
        np.std(samples, axis=0, ddof=1)
        if samples.shape[0] > 1
        else np.zeros(samples.shape[1:], dtype=np.float64)
    )
    return np.mean(samples, axis=0), std


def credible_interval_coverage(
    field_samples: np.ndarray, G_true: np.ndarray, level: float = 0.9
) -> float:
    """Return central credible-interval coverage of a reference field."""

    samples = np.asarray(field_samples, dtype=np.float64)
    truth = np.asarray(G_true, dtype=np.float64)
    if samples.ndim < 2:
        raise ValueError("field_samples must have shape (n_samples, ...).")
    if samples.shape[0] < 1:
        raise ValueError("field_samples must contain at least one sample.")
    if samples.shape[1:] != truth.shape:
        raise ValueError("G_true shape must match one field sample.")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
        raise ValueError("field samples and truth must contain only finite values.")
    if level <= 0.0 or level >= 1.0:
        raise ValueError("level must satisfy 0 < level < 1.")

    tail = 0.5 * (1.0 - level)
    lower, upper = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def interface_jump(G: np.ndarray, sub: Subdomain) -> float:
    """Return the M1 RMS interface jump diagnostic."""

    return interface_jump_rms(np.asarray(G, dtype=np.float64), sub)
