"""Configuration for the M1/M2 research core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Single source of M1/M2 numerical parameters.

    `target_row` and `target_col` intentionally follow the MATLAB convention:
    they are one-based coarse-subdomain identifiers, not zero-based Python
    indices.
    """

    nx: int = 48
    ny: int = 48
    n_coarse_x: int = 4
    n_coarse_y: int = 4
    target_row: int = 2
    target_col: int = 2
    overlap_cells: int = 2
    sigma: float = 1.0
    corr_length: float = 0.18
    n_global_modes: int = 90
    Nc: int = 30
    Mb_list: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    n_samples: int = 100
    seed: int = 7
    sigma_obs: float = 0.01
    n_obs_x: int = 8
    n_obs_y: int = 8
    left_pressure: float = 1.0
    right_pressure: float = 0.0
    beta: float = 0.2
    acceptance: str = "posterior"
    n_chains: int = 4
