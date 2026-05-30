from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from mcmc_multiscale.bayes import log_prior_field
from mcmc_multiscale.config import Config
from mcmc_multiscale.diagnostics import relative_error
from mcmc_multiscale.observations import make_truth
from mcmc_multiscale.sampler import (
    RedBlackSamplerState,
    _acceptance_log_ratio,
    _null_space_log_prior_ratio,
    _null_pcn_proposal_correction,
    conditioned_sampler,
    red_black_conditioned_sampler,
)


def _small_cfg() -> Config:
    return Config(
        nx=12,
        ny=12,
        n_coarse_x=2,
        n_coarse_y=2,
        overlap_cells=1,
        n_global_modes=16,
        Nc=4,
        n_obs_x=4,
        n_obs_y=4,
        sigma_obs=0.01,
        beta=0.2,
        seed=7,
    )


def _posterior_mean_relative_k_error(
    states: list[RedBlackSamplerState], truth_k: np.ndarray
) -> float:
    burn = len(states) // 3
    k_mean = np.mean(
        np.stack([np.exp(state.G_accepted) for state in states[burn:]]), axis=0
    )
    return relative_error(k_mean, truth_k)


@pytest.fixture(scope="module")
def recovery_runs() -> (
    tuple[Config, list[RedBlackSamplerState], list[RedBlackSamplerState]]
):
    cfg = _small_cfg()
    kwargs = dict(
        cfg=cfg,
        n_sweeps=100,
        Mb=2,
        theta_p_method="svd",
        beta=cfg.beta,
    )
    posterior = list(
        red_black_conditioned_sampler(
            rng=np.random.default_rng(cfg.seed), acceptance="posterior", **kwargs
        )
    )
    likelihood_only = list(
        red_black_conditioned_sampler(
            rng=np.random.default_rng(cfg.seed),
            acceptance="likelihood_only",
            **kwargs,
        )
    )
    return cfg, posterior, likelihood_only


def test_log_prior_field_round_trips_global_kle_coefficients() -> None:
    rng = np.random.default_rng(12)
    Phi, _ = np.linalg.qr(rng.standard_normal((7, 4)))
    lam = np.asarray([3.0, 1.5, 0.7, 0.2], dtype=np.float64)
    theta = np.asarray([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
    G_vec = Phi @ (np.sqrt(lam) * theta)

    assert log_prior_field(G_vec, (Phi, lam)) == pytest.approx(
        -0.5 * np.dot(theta, theta)
    )


def test_config_defaults_to_posterior_acceptance() -> None:
    assert Config().acceptance == "posterior"
    assert Config().prior_mode == "global_field"


def test_null_space_log_prior_ratio_matches_known_case() -> None:
    theta_current = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    theta_candidate = np.asarray([2.0, 0.5, 4.0], dtype=np.float64)
    Z = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    assert _null_space_log_prior_ratio(
        theta_current, theta_candidate, Z
    ) == pytest.approx(-0.5 * ((2.0**2 + 0.5**2) - (1.0**2 + 2.0**2)))


def test_null_space_prior_mode_is_non_degenerate_despite_pcn_q_ratio() -> None:
    theta_current = np.asarray([0.5, -1.2], dtype=np.float64)
    theta_candidate = np.asarray([1.1, 0.3], dtype=np.float64)
    Z = np.eye(2, dtype=np.float64)
    proposal_correction = _null_pcn_proposal_correction(
        theta_current, theta_candidate, Z, proposal="pcn", conditioning_mode="hard"
    )
    likelihood_only = _acceptance_log_ratio(-1.5, -2.0, "likelihood_only")
    null_space = _acceptance_log_ratio(
        log_like_candidate=-1.5,
        log_like_current=-2.0,
        acceptance="posterior",
        proposal_correction=proposal_correction,
        prior_mode="null_space",
        theta_current=theta_current,
        theta_candidate=theta_candidate,
        Z=Z,
    )

    assert null_space == pytest.approx(
        likelihood_only + _null_space_log_prior_ratio(theta_current, theta_candidate, Z)
    )
    assert null_space != pytest.approx(likelihood_only)


def test_hard_null_pcn_ratio_reduces_to_likelihood_only_for_fixed_constraints() -> None:
    theta_current = np.asarray([0.5, -1.2], dtype=np.float64)
    theta_candidate = np.asarray([1.1, 0.3], dtype=np.float64)
    Z = np.eye(2, dtype=np.float64)
    proposal_correction = _null_pcn_proposal_correction(
        theta_current, theta_candidate, Z, proposal="pcn", conditioning_mode="hard"
    )

    posterior = _acceptance_log_ratio(
        log_like_candidate=-1.5,
        log_like_current=-2.0,
        acceptance="posterior",
        G_candidate_vec=theta_candidate,
        G_current_vec=theta_current,
        global_kle=(Z, np.ones(2, dtype=np.float64)),
        proposal_correction=proposal_correction,
    )
    likelihood_only = _acceptance_log_ratio(-1.5, -2.0, "likelihood_only")

    assert posterior == pytest.approx(likelihood_only)


def test_posterior_null_pcn_transition_leaves_standard_normal_invariant() -> None:
    rng = np.random.default_rng(33)
    n_samples = 10_000
    dim = 3
    beta = 0.35
    alpha = np.sqrt(1.0 - beta**2)
    Phi = np.eye(dim, dtype=np.float64)
    lam = np.ones(dim, dtype=np.float64)
    current = rng.standard_normal((n_samples, dim))
    candidate = alpha * current + beta * rng.standard_normal((n_samples, dim))

    log_alpha = np.asarray(
        [
            _acceptance_log_ratio(
                0.0,
                0.0,
                "posterior",
                G_candidate_vec=theta_candidate,
                G_current_vec=theta_current,
                global_kle=(Phi, lam),
                proposal_correction=_null_pcn_proposal_correction(
                    theta_current,
                    theta_candidate,
                    Phi,
                    proposal="pcn",
                    conditioning_mode="hard",
                ),
            )
            for theta_current, theta_candidate in zip(current, candidate)
        ],
        dtype=np.float64,
    )

    assert np.max(np.abs(log_alpha)) < 1e-12
    assert np.max(np.abs(np.mean(candidate, axis=0))) < 0.04
    assert np.max(np.abs(np.cov(candidate, rowvar=False) - np.eye(dim))) < 0.05


def test_likelihood_only_default_reproduces_explicit_mode() -> None:
    cfg = _small_cfg()
    kwargs = dict(
        cfg=cfg,
        n_iter=3,
        Mb=2,
        theta_p_method="svd",
        beta=cfg.beta,
    )
    implicit = list(conditioned_sampler(rng=np.random.default_rng(cfg.seed), **kwargs))
    explicit = list(
        conditioned_sampler(
            rng=np.random.default_rng(cfg.seed),
            acceptance="likelihood_only",
            **kwargs,
        )
    )

    for state_implicit, state_explicit in zip(implicit, explicit):
        assert state_implicit.accepted == state_explicit.accepted
        np.testing.assert_allclose(
            state_implicit.theta_local_candidate, state_explicit.theta_local_candidate
        )
        np.testing.assert_allclose(state_implicit.G_accepted, state_explicit.G_accepted)


def test_posterior_red_black_sampler_is_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    kwargs = dict(
        cfg=cfg,
        n_sweeps=2,
        Mb=2,
        theta_p_method="svd",
        beta=cfg.beta,
        acceptance="posterior",
    )
    states_a = list(
        red_black_conditioned_sampler(rng=np.random.default_rng(cfg.seed), **kwargs)
    )
    states_b = list(
        red_black_conditioned_sampler(rng=np.random.default_rng(cfg.seed), **kwargs)
    )

    for state_a, state_b in zip(states_a, states_b):
        assert state_a.accepted == state_b.accepted
        np.testing.assert_allclose(state_a.G_accepted, state_b.G_accepted)
        assert state_a.theta_norm_candidate == pytest.approx(
            state_b.theta_norm_candidate
        )


def test_global_field_prior_mode_is_the_posterior_default() -> None:
    cfg = _small_cfg()
    kwargs = dict(
        cfg=cfg,
        n_sweeps=2,
        Mb=2,
        theta_p_method="svd",
        beta=cfg.beta,
        acceptance="posterior",
    )
    implicit = list(
        red_black_conditioned_sampler(rng=np.random.default_rng(cfg.seed), **kwargs)
    )
    explicit = list(
        red_black_conditioned_sampler(
            rng=np.random.default_rng(cfg.seed), prior_mode="global_field", **kwargs
        )
    )

    for state_implicit, state_explicit in zip(implicit, explicit):
        assert state_implicit.accepted == state_explicit.accepted
        np.testing.assert_allclose(state_implicit.G_accepted, state_explicit.G_accepted)


def test_null_space_prior_red_black_sampler_is_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    kwargs = dict(
        cfg=cfg,
        n_sweeps=2,
        Mb=2,
        theta_p_method="svd",
        beta=cfg.beta,
        acceptance="posterior",
        prior_mode="null_space",
    )
    states_a = list(
        red_black_conditioned_sampler(rng=np.random.default_rng(cfg.seed), **kwargs)
    )
    states_b = list(
        red_black_conditioned_sampler(rng=np.random.default_rng(cfg.seed), **kwargs)
    )

    for state_a, state_b in zip(states_a, states_b):
        assert state_a.accepted == state_b.accepted
        np.testing.assert_allclose(state_a.G_accepted, state_b.G_accepted)
        assert state_a.null_space_norm_accepted == pytest.approx(
            state_b.null_space_norm_accepted
        )


def test_null_space_prior_mode_changes_reduced_sampler_acceptances() -> None:
    cfg = _small_cfg()
    kwargs = dict(
        cfg=cfg,
        n_sweeps=30,
        Mb=2,
        theta_p_method="svd",
        beta=cfg.beta,
        acceptance="posterior",
    )
    global_field = list(
        red_black_conditioned_sampler(
            rng=np.random.default_rng(cfg.seed), prior_mode="global_field", **kwargs
        )
    )
    null_space = list(
        red_black_conditioned_sampler(
            rng=np.random.default_rng(cfg.seed), prior_mode="null_space", **kwargs
        )
    )

    assert any(
        state_global.accepted != state_null.accepted
        for state_global, state_null in zip(global_field, null_space)
    )


def test_reduced_posterior_recovery_is_bounded_and_improves_likelihood_only(
    recovery_runs: tuple[
        Config, list[RedBlackSamplerState], list[RedBlackSamplerState]
    ],
) -> None:
    cfg, posterior, likelihood_only = recovery_runs
    truth = make_truth(cfg, np.random.default_rng(cfg.seed))
    posterior_error = _posterior_mean_relative_k_error(posterior, truth.k_true)
    likelihood_error = _posterior_mean_relative_k_error(likelihood_only, truth.k_true)
    max_norm_ratio = max(
        state.theta_norm_candidate / state.expected_norm for state in posterior
    )

    # This fast route-(a) baseline improves only modestly. Stronger recovery is
    # left to the constrained-manifold formulation in SPEC section 3.8(b).
    assert posterior_error < 0.70
    assert posterior_error < likelihood_error
    assert max_norm_ratio < 1.6


def test_invalid_acceptance_mode_raises() -> None:
    cfg = _small_cfg()
    states: Iterator[RedBlackSamplerState] = red_black_conditioned_sampler(
        cfg=cfg,
        n_sweeps=1,
        Mb=2,
        theta_p_method="svd",
        rng=np.random.default_rng(cfg.seed),
        acceptance="bad",
    )
    with pytest.raises(ValueError, match="acceptance"):
        list(states)


def test_invalid_prior_mode_raises_for_posterior_acceptance() -> None:
    cfg = _small_cfg()
    states = red_black_conditioned_sampler(
        cfg=cfg,
        n_sweeps=1,
        Mb=2,
        theta_p_method="svd",
        rng=np.random.default_rng(cfg.seed),
        acceptance="posterior",
        prior_mode="bad",
    )
    with pytest.raises(ValueError, match="prior_mode"):
        list(states)
