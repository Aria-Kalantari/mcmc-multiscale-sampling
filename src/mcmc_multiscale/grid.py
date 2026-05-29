"""Grid helpers with MATLAB-compatible vector ordering."""

from __future__ import annotations

import numpy as np


def cell_centered_grid(
    nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a uniform cell-centered grid on `[0, 1]^2`.

    The two-dimensional arrays have shape `(ny, nx)`, matching MATLAB
    `meshgrid(xVec, yVec)`. The returned `points` array mirrors MATLAB
    `[X(:), Y(:)]`: cells are flattened in Fortran/column-major order, so the
    global vector index for zero-based `(row, col)` is `row + col * ny`.
    """

    x_vec = ((np.arange(nx, dtype=np.float64) + 0.5) / nx).astype(np.float64)
    y_vec = ((np.arange(ny, dtype=np.float64) + 0.5) / ny).astype(np.float64)
    X, Y = np.meshgrid(x_vec, y_vec)
    points = np.column_stack((X.ravel(order="F"), Y.ravel(order="F"))).astype(
        np.float64, copy=False
    )
    return x_vec, y_vec, X.astype(np.float64), Y.astype(np.float64), points
