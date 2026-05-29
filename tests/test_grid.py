from __future__ import annotations

import numpy as np

from mcmc_multiscale.field import reshape_field
from mcmc_multiscale.grid import cell_centered_grid


def test_cell_centers_and_shapes() -> None:
    x_vec, y_vec, X, Y, points = cell_centered_grid(4, 3)

    np.testing.assert_allclose(x_vec, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(y_vec, [1 / 6, 0.5, 5 / 6])
    assert X.shape == (3, 4)
    assert Y.shape == (3, 4)
    assert points.shape == (12, 2)


def test_fortran_flattening_matches_matlab_x_colon() -> None:
    _, _, _, _, points = cell_centered_grid(3, 2)

    expected = np.array(
        [
            [1 / 6, 1 / 4],
            [1 / 6, 3 / 4],
            [1 / 2, 1 / 4],
            [1 / 2, 3 / 4],
            [5 / 6, 1 / 4],
            [5 / 6, 3 / 4],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(points, expected)


def test_reshape_field_uses_fortran_order() -> None:
    G = reshape_field(np.arange(6, dtype=np.float64), ny=2, nx=3)

    np.testing.assert_array_equal(G, np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]))
