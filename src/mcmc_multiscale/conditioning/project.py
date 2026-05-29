"""Affine projection helpers for hard conditioning."""

from __future__ import annotations

import numpy as np


def affine_project(theta: np.ndarray, theta_p: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Project `theta` onto `theta_p + Null(A)`."""

    theta_arr = np.asarray(theta, dtype=np.float64)
    theta_p_arr = np.asarray(theta_p, dtype=np.float64)
    Z_arr = np.asarray(Z, dtype=np.float64)
    if theta_arr.shape != theta_p_arr.shape:
        raise ValueError("theta and theta_p must have the same shape.")
    if Z_arr.shape[0] != theta_arr.shape[0]:
        raise ValueError("Z must have one row per coefficient.")
    return theta_p_arr + Z_arr @ (Z_arr.T @ (theta_arr - theta_p_arr))


def stabilize(theta_p_any: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Remove any hidden null-space component from a particular solution."""

    theta_p_arr = np.asarray(theta_p_any, dtype=np.float64)
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.shape[0] != theta_p_arr.shape[0]:
        raise ValueError("Z must have one row per coefficient.")
    return theta_p_arr - Z_arr @ (Z_arr.T @ theta_p_arr)
