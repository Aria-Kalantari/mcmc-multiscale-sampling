from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.sampler import conditioned_sampler


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


def _candidate_max(cfg: Config, **kwargs: object) -> float:
    states = list(
        conditioned_sampler(
            cfg,
            n_iter=30,
            Mb=2,
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
            **kwargs,
        )
    )
    return max(state.theta_norm_candidate for state in states)


def test_sampler_supports_lu_stabilized_zero_rhs_and_soft_modes() -> None:
    cfg = _small_cfg()

    lu_stable = list(
        conditioned_sampler(
            cfg,
            n_iter=2,
            Mb=2,
            theta_p_method="lu_stabilized",
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
        )
    )
    zero = list(
        conditioned_sampler(
            cfg,
            n_iter=2,
            Mb=2,
            theta_p_method="svd",
            rhs_mode="zero",
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
        )
    )
    soft = list(
        conditioned_sampler(
            cfg,
            n_iter=2,
            Mb=2,
            theta_p_method="soft",
            conditioning_mode="soft",
            rho=100.0,
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
        )
    )

    assert max(state.hidden_null_norm for state in lu_stable) < 1e-10
    assert max(state.hidden_null_norm for state in zero) == 0.0
    assert all(state.cond_B is None for state in zero)
    assert all(np.isnan(state.hidden_null_norm) for state in soft)
    assert all(state.cond_B is None for state in soft)
    assert all(np.isfinite(state.constraint_residual_candidate) for state in soft)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"theta_p_method": "bad"}, "theta_p_method"),
        ({"theta_p_method": "lu", "rhs_mode": "bad"}, "rhs_mode"),
        ({"theta_p_method": "lu", "conditioning_mode": "bad"}, "conditioning_mode"),
        ({"theta_p_method": "svd", "conditioning_mode": "soft"}, "rho"),
        ({"theta_p_method": "svd", "conditioning_mode": "soft", "rho": -1.0}, "rho"),
    ],
)
def test_sampler_rejects_unsupported_m5_modes(
    kwargs: dict[str, object], match: str
) -> None:
    cfg = _small_cfg()
    with pytest.raises((ValueError, NotImplementedError), match=match):
        list(
            conditioned_sampler(
                cfg,
                n_iter=1,
                Mb=2,
                rng=np.random.default_rng(cfg.seed),
                beta=cfg.beta,
                **kwargs,
            )
        )


def test_m5_sampler_modes_are_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    kwargs = dict(theta_p_method="lu_stabilized")

    states_a = list(
        conditioned_sampler(
            cfg, 3, 2, rng=np.random.default_rng(cfg.seed), beta=cfg.beta, **kwargs
        )
    )
    states_b = list(
        conditioned_sampler(
            cfg, 3, 2, rng=np.random.default_rng(cfg.seed), beta=cfg.beta, **kwargs
        )
    )

    for a, b in zip(states_a, states_b):
        np.testing.assert_allclose(a.theta_local_candidate, b.theta_local_candidate)
        np.testing.assert_allclose(a.G_candidate, b.G_candidate)
        assert a.accepted == b.accepted


def test_sampler_level_stability_fix_regression() -> None:
    cfg = _small_cfg()
    lu_max = _candidate_max(cfg, theta_p_method="lu")
    svd_max = _candidate_max(cfg, theta_p_method="svd")
    stabilized_max = _candidate_max(cfg, theta_p_method="lu_stabilized")
    soft_max = _candidate_max(
        cfg, theta_p_method="soft", conditioning_mode="soft", rho=100.0
    )

    assert lu_max / svd_max > 1.3
    assert stabilized_max / svd_max < 1.5
    assert soft_max / lu_max < 0.9
