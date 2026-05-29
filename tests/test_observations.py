from __future__ import annotations

import numpy as np

from mcmc_multiscale.config import Config
from mcmc_multiscale.observations import (
    make_truth,
    regular_sensor_indices,
    restrict_pressure,
)


def test_restrict_pressure_uses_fortran_flattening() -> None:
    p = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], dtype=np.float64)
    sensor_idx = np.array([0, 3, 5], dtype=np.int64)

    values = restrict_pressure(p, sensor_idx)

    np.testing.assert_array_equal(values, np.array([0.0, 3.0, 5.0]))


def test_regular_sensor_indices_are_valid_unique_flat_indices() -> None:
    sensor_idx = regular_sensor_indices(nx=9, ny=7, n_x=3, n_y=4)

    assert sensor_idx.shape == (12,)
    assert np.unique(sensor_idx).size == sensor_idx.size
    assert np.min(sensor_idx) >= 0
    assert np.max(sensor_idx) < 9 * 7


def test_make_truth_is_deterministic_for_fixed_seed() -> None:
    cfg = Config(nx=8, ny=8, n_global_modes=10, n_obs_x=3, n_obs_y=3, seed=123)

    truth_a = make_truth(cfg)
    truth_b = make_truth(cfg)

    np.testing.assert_allclose(truth_a.theta_true, truth_b.theta_true)
    np.testing.assert_allclose(truth_a.G_true, truth_b.G_true)
    np.testing.assert_allclose(truth_a.k_true, truth_b.k_true)
    np.testing.assert_allclose(truth_a.p_true, truth_b.p_true)
    np.testing.assert_array_equal(truth_a.sensor_idx, truth_b.sensor_idx)
    np.testing.assert_allclose(truth_a.y_clean, truth_b.y_clean)
    np.testing.assert_allclose(truth_a.y_obs, truth_b.y_obs)
