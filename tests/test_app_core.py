from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from experiments.exp05_stability_fixes import run_comparison
from mcmc_multiscale.app_core import (
    MethodRunConfig,
    default_m5_methods,
    run_method,
    run_red_black,
    summarize_states,
)
from mcmc_multiscale.config import Config


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


def test_summarize_states_matches_generated_run() -> None:
    cfg = _small_cfg()
    states, _ = run_method(
        cfg=cfg,
        method=MethodRunConfig("hard_data_svd", "svd"),
        n_iter=4,
        Mb=2,
        beta=cfg.beta,
        seed=cfg.seed,
    )

    summary = summarize_states(
        states,
        label="hard_data_svd",
        Mb=2,
        beta=cfg.beta,
        seed=cfg.seed,
    )

    accepted = np.asarray([state.accepted for state in states], dtype=bool)
    candidate_norms = np.asarray(
        [state.theta_norm_candidate for state in states], dtype=np.float64
    )
    accepted_norms = np.asarray(
        [state.theta_norm_accepted for state in states], dtype=np.float64
    )

    assert summary.acceptance_rate == pytest.approx(float(np.mean(accepted)))
    assert summary.accepted_count == int(np.count_nonzero(accepted))
    assert summary.max_candidate_theta_norm == pytest.approx(np.max(candidate_norms))
    assert summary.final_candidate_theta_norm == pytest.approx(candidate_norms[-1])
    assert summary.max_accepted_theta_norm == pytest.approx(np.max(accepted_norms))
    assert summary.final_accepted_theta_norm == pytest.approx(accepted_norms[-1])


def test_run_method_is_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    method = MethodRunConfig("hard_data_lu_stabilized", "lu_stabilized")

    states_a, summary_a = run_method(cfg, method, 3, 2, cfg.beta, cfg.seed)
    states_b, summary_b = run_method(cfg, method, 3, 2, cfg.beta, cfg.seed)

    assert summary_a == summary_b
    for state_a, state_b in zip(states_a, states_b):
        np.testing.assert_allclose(state_a.G_candidate, state_b.G_candidate)
        np.testing.assert_allclose(
            state_a.theta_local_candidate,
            state_b.theta_local_candidate,
        )
        assert state_a.accepted == state_b.accepted


def test_default_m5_methods_returns_expected_labels() -> None:
    labels = [method.label for method in default_m5_methods([1.0e1, 1.0e3])]

    assert labels == [
        "hard_data_lu",
        "hard_data_svd",
        "hard_data_lu_stabilized",
        "hard_zero_svd",
        "soft_data_rho_1e+01",
        "soft_data_rho_1e+03",
    ]


def test_app_core_does_not_import_streamlit() -> None:
    sys.modules.pop("mcmc_multiscale.app_core", None)
    sys.modules.pop("streamlit", None)

    importlib.import_module("mcmc_multiscale.app_core")

    assert "streamlit" not in sys.modules


def test_exp05_uses_shared_helper_path_on_small_config() -> None:
    rows = run_comparison(
        cfg=_small_cfg(),
        n_iter=1,
        Mb=2,
        beta=0.1,
        seed=13,
        rho_values=[1.0e1],
    )

    assert [row.label for row in rows] == [
        "hard_data_lu",
        "hard_data_svd",
        "hard_data_lu_stabilized",
        "hard_zero_svd",
        "soft_data_rho_1e+01",
    ]
    assert all(row.n_iter == 1 for row in rows)


def test_run_red_black_collects_deterministic_dashboard_summary() -> None:
    cfg = _small_cfg()
    states_a, summary_a = run_red_black(
        cfg=cfg,
        n_sweeps=1,
        Mb=2,
        theta_p_method="svd",
        rng=np.random.default_rng(cfg.seed),
        beta=cfg.beta,
    )
    states_b, summary_b = run_red_black(
        cfg=cfg,
        n_sweeps=1,
        Mb=2,
        theta_p_method="svd",
        rng=np.random.default_rng(cfg.seed),
        beta=cfg.beta,
    )

    expected_updates = cfg.n_coarse_x * cfg.n_coarse_y
    assert len(states_a) == expected_updates
    assert summary_a.total_updates == expected_updates
    assert set(summary_a.colors.tolist()) == {0, 1}
    assert summary_a.both_colors_updated
    assert summary_a.all_subdomains_updated
    assert summary_a.n_color0_subdomains == expected_updates // 2
    assert summary_a.n_color1_subdomains == expected_updates // 2
    assert np.all(np.isfinite(summary_a.candidate_theta_norms))
    assert np.all(np.isfinite(summary_a.accepted_theta_norms))
    assert np.all(np.isfinite(summary_a.constraint_residuals))
    assert np.all(np.isfinite(summary_a.final_G_accepted))
    assert np.all(np.isfinite(summary_a.final_G_candidate))

    assert summary_a.acceptance_rate == pytest.approx(summary_b.acceptance_rate)
    np.testing.assert_allclose(
        summary_a.candidate_theta_norms,
        summary_b.candidate_theta_norms,
    )
    np.testing.assert_allclose(summary_a.final_G_accepted, summary_b.final_G_accepted)
    for state_a, state_b in zip(states_a, states_b):
        assert state_a.color == state_b.color
        assert state_a.subdomain_row == state_b.subdomain_row
        assert state_a.subdomain_col == state_b.subdomain_col
        assert state_a.accepted == state_b.accepted
