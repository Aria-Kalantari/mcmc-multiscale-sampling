from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.sampler import ConditionedSamplerState, conditioned_sampler


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


def test_conditioned_sampler_yields_requested_finite_lu_states() -> None:
    cfg = _small_cfg()
    states = list(
        conditioned_sampler(
            cfg,
            n_iter=3,
            Mb=2,
            theta_p_method="lu",
            rng=np.random.default_rng(cfg.seed),
            beta=0.1,
        )
    )

    assert len(states) == 3
    assert all(isinstance(state, ConditionedSamplerState) for state in states)
    for state in states:
        assert state.G_candidate.shape == (cfg.ny, cfg.nx)
        assert state.G_accepted.shape == (cfg.ny, cfg.nx)
        assert state.pressure_candidate.shape == (cfg.ny, cfg.nx)
        assert state.pressure_accepted.shape == (cfg.ny, cfg.nx)
        assert np.all(np.isfinite(state.G_candidate))
        assert np.all(np.isfinite(state.G_accepted))
        assert np.all(np.isfinite(state.k_candidate))
        assert np.all(np.isfinite(state.k_accepted))
        assert np.all(np.isfinite(state.pressure_candidate))
        assert np.all(np.isfinite(state.pressure_accepted))
        assert np.isfinite(state.log_likelihood_candidate)
        assert np.isfinite(state.log_likelihood_accepted)
        assert state.constraint_residual_candidate < 1e-10
        assert state.cond_A > 0.0
        assert state.cond_B is not None
        assert state.hidden_null_norm > 1e-12


def test_conditioned_sampler_is_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    states_a = list(
        conditioned_sampler(cfg, 2, 2, "lu", np.random.default_rng(cfg.seed), beta=0.1)
    )
    states_b = list(
        conditioned_sampler(cfg, 2, 2, "lu", np.random.default_rng(cfg.seed), beta=0.1)
    )

    for a, b in zip(states_a, states_b):
        np.testing.assert_allclose(a.theta_local_candidate, b.theta_local_candidate)
        np.testing.assert_allclose(a.theta_local_accepted, b.theta_local_accepted)
        np.testing.assert_allclose(a.G_candidate, b.G_candidate)
        assert a.accepted == b.accepted


def test_conditioned_sampler_carries_previous_accepted_theta_forward() -> None:
    cfg = _small_cfg()
    beta = 0.05
    states = list(
        conditioned_sampler(cfg, 2, 2, "lu", np.random.default_rng(cfg.seed), beta=beta)
    )

    assert states[0].accepted
    alpha = np.sqrt(1.0 - beta**2)
    carried_difference = np.linalg.norm(
        states[1].theta_proposed - alpha * states[0].theta_local_accepted
    )
    assert carried_difference < 0.5


def test_conditioned_sampler_unsupported_update_scheme_raises() -> None:
    cfg = _small_cfg()
    with pytest.raises(NotImplementedError, match="single"):
        list(
            conditioned_sampler(
                cfg,
                n_iter=1,
                Mb=2,
                theta_p_method="lu",
                rng=np.random.default_rng(cfg.seed),
                update_scheme="red_black",
            )
        )


def test_conditioned_sampler_svd_control_path_runs() -> None:
    cfg = _small_cfg()
    states = list(
        conditioned_sampler(
            cfg,
            n_iter=2,
            Mb=2,
            theta_p_method="svd",
            rng=np.random.default_rng(cfg.seed),
            beta=0.1,
        )
    )

    assert len(states) == 2
    assert all(state.cond_B is None for state in states)
    assert max(state.hidden_null_norm for state in states) < 1e-10
    assert max(state.constraint_residual_candidate for state in states) < 1e-10
