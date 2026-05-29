"""Overlapping subdomain construction and interface diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc_multiscale.config import Config


@dataclass(frozen=True)
class Subdomain:
    """Index data for one coarse core and its overlapping local region."""

    core_cols: np.ndarray
    core_rows: np.ndarray
    hat_cols: np.ndarray
    hat_rows: np.ndarray
    local_global_idx: np.ndarray
    core_local_idx: np.ndarray
    buffer_local_idx: np.ndarray

    @property
    def core_global_idx(self) -> np.ndarray:
        """Global vector indices of core cells in MATLAB/Fortran order."""

        return self.local_global_idx[self.core_local_idx]


def make_subdomain(cfg: Config) -> Subdomain:
    """Mirror MATLAB `makeSubdomain` for the configured target.

    `Config.target_row` and `Config.target_col` are one-based. New helper
    functions below use zero-based coarse coordinates internally.
    """

    return make_subdomain_at(cfg, cfg.target_row - 1, cfg.target_col - 1)


def make_subdomain_at(cfg: Config, target_row: int, target_col: int) -> Subdomain:
    """Build a subdomain at zero-based coarse coordinates."""

    if cfg.nx % cfg.n_coarse_x != 0 or cfg.ny % cfg.n_coarse_y != 0:
        raise ValueError("nx and ny must be divisible by the coarse grid.")

    block_x = cfg.nx // cfg.n_coarse_x
    block_y = cfg.ny // cfg.n_coarse_y

    if not (0 <= target_col < cfg.n_coarse_x):
        raise ValueError("target_col must be zero-based and within n_coarse_x.")
    if not (0 <= target_row < cfg.n_coarse_y):
        raise ValueError("target_row must be zero-based and within n_coarse_y.")

    col0 = target_col * block_x
    row0 = target_row * block_y
    core_cols = np.arange(col0, col0 + block_x, dtype=np.int64)
    core_rows = np.arange(row0, row0 + block_y, dtype=np.int64)

    ov = cfg.overlap_cells
    hat_cols = np.arange(
        max(0, core_cols[0] - ov), min(cfg.nx, core_cols[-1] + ov + 1), dtype=np.int64
    )
    hat_rows = np.arange(
        max(0, core_rows[0] - ov), min(cfg.ny, core_rows[-1] + ov + 1), dtype=np.int64
    )

    HatCols, HatRows = np.meshgrid(hat_cols, hat_rows)
    local_global_idx = (
        HatRows.ravel(order="F") + HatCols.ravel(order="F") * cfg.ny
    ).astype(np.int64)

    is_core = (
        (HatCols.ravel(order="F") >= core_cols[0])
        & (HatCols.ravel(order="F") <= core_cols[-1])
        & (HatRows.ravel(order="F") >= core_rows[0])
        & (HatRows.ravel(order="F") <= core_rows[-1])
    )

    return Subdomain(
        core_cols=core_cols,
        core_rows=core_rows,
        hat_cols=hat_cols,
        hat_rows=hat_rows,
        local_global_idx=local_global_idx,
        core_local_idx=np.flatnonzero(is_core).astype(np.int64),
        buffer_local_idx=np.flatnonzero(~is_core).astype(np.int64),
    )


def iter_coarse_subdomains(cfg: Config) -> list[tuple[int, int]]:
    """Return all zero-based coarse coordinates in sorted row-major order."""

    return [
        (row, col) for row in range(cfg.n_coarse_y) for col in range(cfg.n_coarse_x)
    ]


def checkerboard_color(row: int, col: int) -> int:
    """Return the 2-color checkerboard color for zero-based coordinates."""

    return int((row + col) % 2)


def red_black_order(cfg: Config) -> dict[int, list[tuple[int, int]]]:
    """Return zero-based coarse coordinates grouped by checkerboard color."""

    order: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for row, col in iter_coarse_subdomains(cfg):
        order[checkerboard_color(row, col)].append((row, col))
    return order


def interface_jump_rms(G: np.ndarray, sub: Subdomain) -> float:
    """Return the RMS jump across the core boundary.

    This ports MATLAB `interfaceJumpRMS`: for each side of the core, compare
    the inside boundary cells with their immediate outside neighbors.
    """

    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.ndim != 2:
        raise ValueError("G must be a two-dimensional field array.")

    rows = sub.core_rows
    cols = sub.core_cols
    diffs: list[np.ndarray] = []

    if cols[0] > 0:
        diffs.append(
            G_arr[np.ix_(rows, [cols[0]])].ravel()
            - G_arr[np.ix_(rows, [cols[0] - 1])].ravel()
        )
    if cols[-1] < G_arr.shape[1] - 1:
        diffs.append(
            G_arr[np.ix_(rows, [cols[-1]])].ravel()
            - G_arr[np.ix_(rows, [cols[-1] + 1])].ravel()
        )
    if rows[0] > 0:
        diffs.append(
            G_arr[np.ix_([rows[0]], cols)].ravel()
            - G_arr[np.ix_([rows[0] - 1], cols)].ravel()
        )
    if rows[-1] < G_arr.shape[0] - 1:
        diffs.append(
            G_arr[np.ix_([rows[-1]], cols)].ravel()
            - G_arr[np.ix_([rows[-1] + 1], cols)].ravel()
        )

    if not diffs:
        return 0.0
    all_diffs = np.concatenate(diffs)
    return float(np.sqrt(np.mean(all_diffs**2)))
