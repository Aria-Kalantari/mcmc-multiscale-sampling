"""Synthetic pressure observations for the Bayesian inverse problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc_multiscale.config import Config
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.field import (
    field_from_theta,
    permeability_from_log_field,
    reshape_field,
)
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs


@dataclass(frozen=True)
class TruthData:
    theta_true: np.ndarray
    G_true: np.ndarray
    k_true: np.ndarray
    p_true: np.ndarray
    sensor_idx: np.ndarray
    y_obs: np.ndarray
    y_clean: np.ndarray


def _regular_axis_indices(n: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("sensor count must be positive.")
    if count > n:
        raise ValueError("sensor count cannot exceed the grid size.")
    positions = (np.arange(count, dtype=np.float64) + 0.5) * n / count
    return np.floor(positions).astype(np.int64)


def regular_sensor_indices(nx: int, ny: int, n_x: int, n_y: int) -> np.ndarray:
    """Return regular pressure-sensor indices in Fortran flattening order.

    # CONFIRM: M2 defaults to a regular pressure-sensor grid; the final
    observation layout/count is still a modeling choice.
    """

    cols = _regular_axis_indices(nx, n_x)
    rows = _regular_axis_indices(ny, n_y)
    Cols, Rows = np.meshgrid(cols, rows)
    sensor_idx = (Rows.ravel(order="F") + Cols.ravel(order="F") * ny).astype(np.int64)
    if np.unique(sensor_idx).size != sensor_idx.size:
        raise ValueError("regular sensor selection produced duplicate cells.")
    return sensor_idx


def restrict_pressure(p_2d: np.ndarray, sensor_idx: np.ndarray) -> np.ndarray:
    """Restrict a `(ny, nx)` pressure field to flattened sensor indices."""

    p = np.asarray(p_2d, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("p_2d must be two-dimensional.")
    idx = np.asarray(sensor_idx, dtype=np.int64)
    flat = p.ravel(order="F")
    if np.any(idx < 0) or np.any(idx >= flat.size):
        raise ValueError("sensor_idx contains out-of-bounds entries.")
    return flat[idx].astype(np.float64, copy=False)


def make_truth(cfg: Config, rng: np.random.Generator | None = None) -> TruthData:
    """Generate deterministic synthetic truth and noisy pressure observations."""

    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    C = exp_covariance(points, cfg.sigma, cfg.corr_length)
    Phi, lam = top_eigenpairs(C, cfg.n_global_modes)

    theta_true = rng.standard_normal(Phi.shape[1], dtype=np.float64)
    G_vec = field_from_theta(Phi, lam, theta_true)
    G_true = reshape_field(G_vec, cfg.ny, cfg.nx)
    k_true = permeability_from_log_field(G_true)
    p_true = ForwardModel(cfg).solve(k_true)

    sensor_idx = regular_sensor_indices(cfg.nx, cfg.ny, cfg.n_obs_x, cfg.n_obs_y)
    y_clean = restrict_pressure(p_true, sensor_idx)
    noise = cfg.sigma_obs * rng.standard_normal(y_clean.shape, dtype=np.float64)
    y_obs = y_clean + noise

    return TruthData(
        theta_true=theta_true.astype(np.float64, copy=False),
        G_true=G_true,
        k_true=k_true,
        p_true=p_true,
        sensor_idx=sensor_idx,
        y_obs=y_obs.astype(np.float64, copy=False),
        y_clean=y_clean,
    )
