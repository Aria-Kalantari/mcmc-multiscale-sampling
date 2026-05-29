from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.sampler import red_black_conditioned_sampler
from mcmc_multiscale.subdomain import (
    checkerboard_color,
    iter_coarse_subdomains,
    make_subdomain,
    make_subdomain_at,
    red_black_order,
)


def _small_cfg() -> Config:
    return Config(
        nx=12,
        ny=12,
        n_coarse_x=4,
        n_coarse_y=4,
        overlap_cells=1,
        n_global_modes=8,
        Nc=4,
        n_obs_x=3,
        n_obs_y=3,
        sigma_obs=1.0e6,
        beta=0.1,
        seed=13,
    )


def test_make_subdomain_at_preserves_default_target() -> None:
    cfg = Config()
    default_sub = make_subdomain(cfg)
    explicit_sub = make_subdomain_at(cfg, cfg.target_row - 1, cfg.target_col - 1)

    np.testing.assert_array_equal(default_sub.core_rows, explicit_sub.core_rows)
    np.testing.assert_array_equal(default_sub.core_cols, explicit_sub.core_cols)
    np.testing.assert_array_equal(
        default_sub.local_global_idx, explicit_sub.local_global_idx
    )
    np.testing.assert_array_equal(
        default_sub.core_local_idx, explicit_sub.core_local_idx
    )
    np.testing.assert_array_equal(
        default_sub.buffer_local_idx, explicit_sub.buffer_local_idx
    )


def test_arbitrary_subdomain_at_has_valid_indices() -> None:
    cfg = Config()
    sub = make_subdomain_at(cfg, 0, 3)

    assert sub.core_local_idx.size == 144
    assert sub.buffer_local_idx.size > 0
    assert np.all(sub.local_global_idx >= 0)
    assert np.all(sub.local_global_idx < cfg.nx * cfg.ny)
    assert np.all(sub.core_global_idx >= 0)
    assert np.all(sub.core_global_idx < cfg.nx * cfg.ny)


def test_checkerboard_order_covers_all_subdomains_once() -> None:
    cfg = _small_cfg()
    all_subdomains = iter_coarse_subdomains(cfg)
    order = red_black_order(cfg)
    flattened = order[0] + order[1]

    assert sorted(flattened) == all_subdomains
    assert len(flattened) == len(set(flattened))
    assert len(order[0]) == 8
    assert len(order[1]) == 8


def test_edge_adjacent_subdomains_have_opposite_colors() -> None:
    cfg = _small_cfg()
    for row, col in iter_coarse_subdomains(cfg):
        color = checkerboard_color(row, col)
        if row + 1 < cfg.n_coarse_y:
            assert checkerboard_color(row + 1, col) != color
        if col + 1 < cfg.n_coarse_x:
            assert checkerboard_color(row, col + 1) != color


def test_diagonal_neighbors_may_share_color() -> None:
    assert checkerboard_color(0, 0) == checkerboard_color(1, 1)


def test_red_black_sampler_runs_one_sweep_and_visits_every_subdomain() -> None:
    cfg = _small_cfg()
    states = list(
        red_black_conditioned_sampler(
            cfg,
            n_sweeps=1,
            Mb=2,
            theta_p_method="svd",
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
        )
    )

    visited = {(state.subdomain_row, state.subdomain_col) for state in states}
    assert len(states) == cfg.n_coarse_x * cfg.n_coarse_y
    assert {state.color for state in states} == {0, 1}
    assert visited == set(iter_coarse_subdomains(cfg))
    assert max(state.constraint_residual_candidate for state in states) < 1e-10
    assert all(np.isfinite(state.theta_norm_candidate) for state in states)
    assert all(np.isfinite(state.theta_norm_accepted) for state in states)
    assert all(np.isfinite(state.log_likelihood_accepted) for state in states)
    assert all(np.all(np.isfinite(state.G_candidate)) for state in states)
    assert all(np.all(np.isfinite(state.G_accepted)) for state in states)


def test_red_black_sampler_is_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    kwargs = dict(n_sweeps=1, Mb=2, theta_p_method="svd", beta=cfg.beta)

    states_a = list(
        red_black_conditioned_sampler(
            cfg, rng=np.random.default_rng(cfg.seed), **kwargs
        )
    )
    states_b = list(
        red_black_conditioned_sampler(
            cfg, rng=np.random.default_rng(cfg.seed), **kwargs
        )
    )

    for state_a, state_b in zip(states_a, states_b):
        assert state_a.sweep == state_b.sweep
        assert state_a.color == state_b.color
        assert state_a.subdomain_row == state_b.subdomain_row
        assert state_a.subdomain_col == state_b.subdomain_col
        assert state_a.accepted == state_b.accepted
        np.testing.assert_allclose(state_a.G_candidate, state_b.G_candidate)
        np.testing.assert_allclose(state_a.G_accepted, state_b.G_accepted)
        assert state_a.theta_norm_candidate == pytest.approx(
            state_b.theta_norm_candidate
        )


def test_red_black_sampler_rejects_unsupported_modes() -> None:
    cfg = _small_cfg()
    with pytest.raises(ValueError, match="n_sweeps"):
        list(
            red_black_conditioned_sampler(
                cfg,
                n_sweeps=0,
                Mb=2,
                theta_p_method="svd",
                rng=np.random.default_rng(cfg.seed),
            )
        )
    with pytest.raises(ValueError, match="conditioning_mode"):
        list(
            red_black_conditioned_sampler(
                cfg,
                n_sweeps=1,
                Mb=2,
                theta_p_method="svd",
                conditioning_mode="bad",
                rng=np.random.default_rng(cfg.seed),
            )
        )
    with pytest.raises(ValueError, match="rho"):
        list(
            red_black_conditioned_sampler(
                cfg,
                n_sweeps=1,
                Mb=2,
                theta_p_method="soft",
                conditioning_mode="soft",
                rng=np.random.default_rng(cfg.seed),
            )
        )


def test_red_black_svd_default_has_bounded_finite_norms() -> None:
    cfg = _small_cfg()
    states = list(
        red_black_conditioned_sampler(
            cfg,
            n_sweeps=2,
            Mb=2,
            theta_p_method="svd",
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
        )
    )

    max_ratio = max(
        state.theta_norm_candidate / state.expected_norm for state in states
    )
    assert max_ratio < 3.0
