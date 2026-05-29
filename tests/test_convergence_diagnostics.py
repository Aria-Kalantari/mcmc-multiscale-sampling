from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from experiments.exp08_convergence_diagnostics import _resolved_lengths
from mcmc_multiscale.config import Config
from mcmc_multiscale.diagnostics import (
    credible_interval_coverage,
    effective_sample_size,
    gelman_rubin,
    integrated_autocorr_time,
    posterior_summary,
    sampling_efficiency,
)


def _ar1(phi: float, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_samples, dtype=np.float64)
    x = np.empty(n_samples, dtype=np.float64)
    x[0] = noise[0] / np.sqrt(1.0 - phi**2)
    for idx in range(1, n_samples):
        x[idx] = phi * x[idx - 1] + noise[idx]
    return x


def test_config_defaults_to_four_chains() -> None:
    assert Config().n_chains == 4


def test_iat_and_ess_match_ar1_theory() -> None:
    phi = 0.7
    x = _ar1(phi=phi, n_samples=80_000, seed=12)
    expected_tau = (1.0 + phi) / (1.0 - phi)
    tau = integrated_autocorr_time(x)
    ess = effective_sample_size(x)

    assert tau == pytest.approx(expected_tau, rel=0.15)
    assert ess == pytest.approx(x.size / expected_tau, rel=0.15)
    assert ess == pytest.approx(x.size / tau)


def test_gelman_rubin_distinguishes_mixed_and_disjoint_chains() -> None:
    rng = np.random.default_rng(13)
    mixed = rng.standard_normal((4, 4_000))
    disjoint = mixed + np.asarray([[-4.0], [-2.0], [2.0], [4.0]])

    assert gelman_rubin(mixed) == pytest.approx(1.0, abs=0.01)
    assert gelman_rubin(disjoint) > 2.0


def test_sampling_efficiency_normalizes_ess_by_compute_cost() -> None:
    per_thousand_solves, per_second = sampling_efficiency(
        total_ess=25.0, total_forward_solves=5_000, wall_seconds=10.0
    )

    assert per_thousand_solves == pytest.approx(5.0)
    assert per_second == pytest.approx(2.5)


def test_sampling_efficiency_rejects_invalid_costs() -> None:
    with pytest.raises(ValueError, match="forward"):
        sampling_efficiency(1.0, 0, 1.0)
    with pytest.raises(ValueError, match="wall"):
        sampling_efficiency(1.0, 1, 0.0)


def test_exp08_long_profile_is_opt_in_and_overridable() -> None:
    responsive = _resolved_lengths(
        Namespace(
            long=False,
            single_iter=None,
            red_black_sweeps=None,
            global_iter=None,
        )
    )
    long = _resolved_lengths(
        Namespace(
            long=True,
            single_iter=None,
            red_black_sweeps=None,
            global_iter=None,
        )
    )
    overridden = _resolved_lengths(
        Namespace(
            long=True,
            single_iter=11,
            red_black_sweeps=12,
            global_iter=13,
        )
    )

    assert (
        responsive.single_iter,
        responsive.red_black_sweeps,
        responsive.global_iter,
    ) == (
        200,
        10,
        500,
    )
    assert (long.single_iter, long.red_black_sweeps, long.global_iter) == (
        2_000,
        100,
        5_000,
    )
    assert (
        overridden.single_iter,
        overridden.red_black_sweeps,
        overridden.global_iter,
    ) == (11, 12, 13)


def test_posterior_summary_returns_per_cell_mean_and_std() -> None:
    samples = np.asarray(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[3.0, 5.0], [7.0, 9.0]],
        ],
        dtype=np.float64,
    )

    mean, std = posterior_summary(samples)

    np.testing.assert_allclose(mean, [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(std, np.sqrt(2.0))


def test_credible_interval_coverage_matches_gaussian_level() -> None:
    rng = np.random.default_rng(14)
    n_cells = 2_000
    truth = rng.standard_normal(n_cells, dtype=np.float64)
    samples = rng.standard_normal((2_000, n_cells), dtype=np.float64)

    coverage = credible_interval_coverage(samples, truth, level=0.9)

    assert coverage == pytest.approx(0.9, abs=0.03)


def test_credible_interval_coverage_validates_inputs() -> None:
    samples = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="level"):
        credible_interval_coverage(samples, np.zeros(3), level=1.0)
    with pytest.raises(ValueError, match="shape"):
        credible_interval_coverage(samples, np.zeros(2))


def test_seeded_diagnostics_are_deterministic() -> None:
    x_a = _ar1(phi=0.5, n_samples=1_000, seed=15)
    x_b = _ar1(phi=0.5, n_samples=1_000, seed=15)

    np.testing.assert_array_equal(x_a, x_b)
    assert integrated_autocorr_time(x_a) == integrated_autocorr_time(x_b)
    assert effective_sample_size(x_a) == effective_sample_size(x_b)
