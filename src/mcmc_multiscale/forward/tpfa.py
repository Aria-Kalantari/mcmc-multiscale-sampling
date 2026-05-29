"""Cell-centered TPFA Darcy pressure solver."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve

from mcmc_multiscale.config import Config
from mcmc_multiscale.field import reshape_field


def _as_source_array(
    source: np.ndarray | None,
    ny: int,
    nx: int,
) -> np.ndarray:
    if source is None:
        return np.zeros((ny, nx), dtype=np.float64)
    source_arr = np.asarray(source, dtype=np.float64)
    if source_arr.shape != (ny, nx):
        raise ValueError("source must have shape (ny, nx).")
    return source_arr


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


def _add_entry(
    rows: list[int],
    cols: list[int],
    data: list[float],
    row: int,
    col: int,
    value: float,
) -> None:
    rows.append(row)
    cols.append(col)
    data.append(float(value))


def _assemble_system(
    k_field_2d: np.ndarray,
    cfg: Config,
    source: np.ndarray | None = None,
) -> tuple[csc_matrix, np.ndarray]:
    """Assemble the sparse TPFA system for `-div(k grad p) = f`.

    Unknowns are flattened in the repository's MATLAB-compatible Fortran order:
    zero-based `(row, col)` maps to `row + col * ny`.

    # CONFIRM: The default boundary convention is Dirichlet pressure on the
    left/right boundaries and homogeneous no-flow Neumann on top/bottom.
    """

    k = np.asarray(k_field_2d, dtype=np.float64)
    if k.shape != (cfg.ny, cfg.nx):
        raise ValueError("k_field_2d must have shape (cfg.ny, cfg.nx).")
    if not np.all(np.isfinite(k)):
        raise ValueError("k_field_2d must be finite.")
    if np.any(k <= 0.0):
        raise ValueError("k_field_2d must be strictly positive.")

    f = _as_source_array(source, cfg.ny, cfg.nx)
    dx = np.float64(1.0 / cfg.nx)
    dy = np.float64(1.0 / cfg.ny)
    volume = dx * dy

    rhs = (f * volume).ravel(order="F").astype(np.float64, copy=True)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def idx(row: int, col: int) -> int:
        return row + col * cfg.ny

    def add_symmetric_face(
        row_a: int, col_a: int, row_b: int, col_b: int, T: float
    ) -> None:
        ia = idx(row_a, col_a)
        ib = idx(row_b, col_b)
        _add_entry(rows, cols, data, ia, ia, T)
        _add_entry(rows, cols, data, ib, ib, T)
        _add_entry(rows, cols, data, ia, ib, -T)
        _add_entry(rows, cols, data, ib, ia, -T)

    for col in range(cfg.nx - 1):
        k_face = _harmonic_mean(k[:, col], k[:, col + 1])
        transmissibility = k_face * dy / dx
        for row in range(cfg.ny):
            add_symmetric_face(row, col, row, col + 1, transmissibility[row])

    for row in range(cfg.ny - 1):
        k_face = _harmonic_mean(k[row, :], k[row + 1, :])
        transmissibility = k_face * dx / dy
        for col in range(cfg.nx):
            add_symmetric_face(row, col, row + 1, col, transmissibility[col])

    left_T = 2.0 * k[:, 0] * dy / dx
    right_T = 2.0 * k[:, -1] * dy / dx

    # CONFIRM: Boundary pressures default to p=1 on x=0 and p=0 on x=1.
    for row in range(cfg.ny):
        left_idx = idx(row, 0)
        right_idx = idx(row, cfg.nx - 1)
        _add_entry(rows, cols, data, left_idx, left_idx, left_T[row])
        _add_entry(rows, cols, data, right_idx, right_idx, right_T[row])
        rhs[left_idx] += left_T[row] * cfg.left_pressure
        rhs[right_idx] += right_T[row] * cfg.right_pressure

    n = cfg.nx * cfg.ny
    matrix = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsc()
    return matrix, rhs


class ForwardModel:
    """Sparse TPFA pressure solver for the default Darcy forward problem."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def solve(self, k_field_2d: np.ndarray) -> np.ndarray:
        """Solve the source-free TPFA system and return pressure `(ny, nx)`.

        The public M2 solve path uses `f = 0`. Tests use `_assemble_system`
        directly for manufactured-source verification.
        """

        matrix, rhs = _assemble_system(k_field_2d, self.cfg, source=None)
        p_vec = spsolve(matrix, rhs).astype(np.float64, copy=False)
        return reshape_field(p_vec, self.cfg.ny, self.cfg.nx)
