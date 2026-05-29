"""Field reconstruction helpers."""

from __future__ import annotations

import numpy as np


def field_from_theta(Phi: np.ndarray, lam: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the truncated KLE log-field vector.

    Computes `G_vec = Phi @ (sqrt(lam) * theta)` in float64.
    """

    Phi_arr = np.asarray(Phi, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if Phi_arr.ndim != 2:
        raise ValueError("Phi must be two-dimensional.")
    if lam_arr.shape != (Phi_arr.shape[1],):
        raise ValueError("lam must have one entry per column of Phi.")
    if theta_arr.shape != (Phi_arr.shape[1],):
        raise ValueError("theta must have one entry per column of Phi.")
    return Phi_arr @ (np.sqrt(lam_arr) * theta_arr)


def reshape_field(G_vec: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Reshape a MATLAB-ordered field vector to a `(ny, nx)` array."""

    vec = np.asarray(G_vec, dtype=np.float64)
    if vec.shape != (ny * nx,):
        raise ValueError("G_vec length must equal ny * nx.")
    return np.reshape(vec, (ny, nx), order="F")


def permeability_from_log_field(G: np.ndarray) -> np.ndarray:
    """Return permeability `k = exp(G)` from a log-permeability field."""

    return np.exp(np.asarray(G, dtype=np.float64))
