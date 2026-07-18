"""Covariance matrices for Gaussian log-permeability fields."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def exp_covariance(points: np.ndarray, sigma: float, ell: float) -> np.ndarray:
    """Return the exponential covariance matrix.

    The covariance is

    `C_ij = sigma**2 * exp(-||x_i - x_j||_2 / ell)`.

    The result is explicitly symmetrized and receives the MATLAB-reference
    jitter `1e-12 * I`.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2).")
    if ell <= 0:
        raise ValueError("ell must be positive.")

    dist = cdist(pts, pts, metric="euclidean")
    C = np.asarray(sigma, dtype=np.float64) ** 2 * np.exp(-dist / ell)
    C = (C + C.T) / 2.0
    C += 1e-12 * np.eye(C.shape[0], dtype=np.float64)
    return C.astype(np.float64, copy=False)


def sqexp_covariance(points: np.ndarray, sigma: float, ell: float) -> np.ndarray:
    """Return the squared-exponential (Gaussian) covariance matrix.

    The covariance is

    `C_ij = sigma**2 * exp(-||x_i - x_j||_2**2 / (2 * ell**2))`

    with `ell` the standard length scale. Its spectrum collapses far faster than
    the exponential kernel's, so far fewer KLE modes capture the same energy. The
    result is explicitly symmetrized and receives the MATLAB-reference jitter
    `1e-12 * I`.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2).")
    if ell <= 0:
        raise ValueError("ell must be positive.")

    sqdist = cdist(pts, pts, metric="sqeuclidean")
    C = np.asarray(sigma, dtype=np.float64) ** 2 * np.exp(-0.5 * sqdist / ell**2)
    C = (C + C.T) / 2.0
    C += 1e-12 * np.eye(C.shape[0], dtype=np.float64)
    return C.astype(np.float64, copy=False)
