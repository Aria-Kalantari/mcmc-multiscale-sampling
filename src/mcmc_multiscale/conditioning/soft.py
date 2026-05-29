"""Soft / regularized conditioning maps."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve


def _validate_inputs(
    A: np.ndarray, c: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, float]:
    A_arr = np.asarray(A, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    rho_float = float(rho)
    if A_arr.ndim != 2:
        raise ValueError("A must be two-dimensional.")
    if c_arr.shape != (A_arr.shape[0],):
        raise ValueError("c must have one entry per row of A.")
    if not np.isfinite(rho_float) or rho_float <= 0.0:
        raise ValueError("rho must be positive.")
    return A_arr, c_arr, rho_float


def soft_min_norm_particular(A: np.ndarray, c: np.ndarray, rho: float) -> np.ndarray:
    """Return the soft minimum-norm particular solution.

    Solves `min_theta 0.5 ||theta||^2 + 0.5 rho ||A theta - c||^2`
    through the small SPD dual system
    `theta = rho A.T solve(I + rho A A.T, c)`.
    """

    A_arr, c_arr, rho_float = _validate_inputs(A, c, rho)
    M = np.eye(A_arr.shape[0], dtype=np.float64) + rho_float * (A_arr @ A_arr.T)
    y = solve(M, c_arr, assume_a="pos", check_finite=True)
    return (rho_float * (A_arr.T @ y)).astype(np.float64, copy=False)


def soft_project(
    theta: np.ndarray, A: np.ndarray, c: np.ndarray, rho: float
) -> np.ndarray:
    """Return the proximal soft-conditioning projection of `theta`.

    Solves `min_x 0.5 ||x - theta||^2 + 0.5 rho ||A x - c||^2`
    through `x = theta - rho A.T solve(I + rho A A.T, A theta - c)`.
    """

    A_arr, c_arr, rho_float = _validate_inputs(A, c, rho)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.shape != (A_arr.shape[1],):
        raise ValueError("theta must have one entry per column of A.")
    residual = A_arr @ theta_arr - c_arr
    M = np.eye(A_arr.shape[0], dtype=np.float64) + rho_float * (A_arr @ A_arr.T)
    y = solve(M, residual, assume_a="pos", check_finite=True)
    return (theta_arr - rho_float * (A_arr.T @ y)).astype(np.float64, copy=False)
