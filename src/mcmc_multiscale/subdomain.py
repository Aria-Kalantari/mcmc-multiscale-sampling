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
    """Mirror MATLAB `makeSubdomain` using zero-based Python indices."""

    if cfg.nx % cfg.n_coarse_x != 0 or cfg.ny % cfg.n_coarse_y != 0:
        raise ValueError("nx and ny must be divisible by the coarse grid.")

    block_x = cfg.nx // cfg.n_coarse_x
    block_y = cfg.ny // cfg.n_coarse_y

    if not (1 <= cfg.target_col <= cfg.n_coarse_x):
        raise ValueError("target_col is one-based and must be within n_coarse_x.")
    if not (1 <= cfg.target_row <= cfg.n_coarse_y):
        raise ValueError("target_row is one-based and must be within n_coarse_y.")

    col0 = (cfg.target_col - 1) * block_x
    row0 = (cfg.target_row - 1) * block_y
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
