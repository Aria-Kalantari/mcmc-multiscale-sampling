from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.mcmc import MCMCState, collect_chain, metropolis_hastings
from mcmc_multiscale.proposals import (
    ProposalResult,
    make_pcn_proposal,
    make_random_walk_proposal,
)


def _standard_normal_log_prior(theta: np.ndarray) -> float:
    theta_arr = np.asarray(theta, dtype=np.float64)
    return float(-0.5 * np.dot(theta_arr, theta_arr))


def _gaussian_log_density(
    theta: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray
) -> float:
    diff = np.asarray(theta, dtype=np.float64) - mean
    return float(-0.5 * diff @ cov_inv @ diff)


def _run_chain(
    log_density_fn,
    proposal_fn,
    theta0: np.ndarray,
    n_iter: int,
    seed: int,
    log_prior_fn=None,
) -> tuple[np.ndarray, np.ndarray, list[MCMCState]]:
    states = list(
        metropolis_hastings(
            log_density_fn=log_density_fn,
            proposal_fn=proposal_fn,
            theta0=theta0,
            n_iter=n_iter,
            rng=np.random.default_rng(seed),
            log_prior_fn=log_prior_fn,
        )
    )
    chain, accepted = collect_chain(states)
    return chain, accepted, states


def test_random_walk_samples_1d_standard_normal() -> None:
    chain, accepted, _ = _run_chain(
        _standard_normal_log_prior,
        make_random_walk_proposal(beta=0.9),
        np.array([0.0], dtype=np.float64),
        n_iter=12_000,
        seed=10,
    )
    samples = chain[2_000:, 0]

    assert abs(float(np.mean(samples))) < 0.08
    assert abs(float(np.var(samples)) - 1.0) < 0.10
    assert 0.45 < float(np.mean(accepted)) < 0.90


def test_random_walk_samples_2d_gaussian_target() -> None:
    mean = np.array([1.0, -0.5], dtype=np.float64)
    cov = np.array([[1.0, 0.3], [0.3, 0.6]], dtype=np.float64)
    cov_inv = np.linalg.inv(cov)

    chain, accepted, _ = _run_chain(
        lambda theta: _gaussian_log_density(theta, mean, cov_inv),
        make_random_walk_proposal(beta=0.8),
        np.zeros(2, dtype=np.float64),
        n_iter=20_000,
        seed=20,
    )
    samples = chain[4_000:, :]

    np.testing.assert_allclose(np.mean(samples, axis=0), mean, atol=0.12)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), cov, atol=0.14)
    assert 0.25 < float(np.mean(accepted)) < 0.85


def test_pcn_samples_standard_normal_through_engine() -> None:
    chain, accepted, states = _run_chain(
        _standard_normal_log_prior,
        make_pcn_proposal(beta=1.0),
        np.array([0.0], dtype=np.float64),
        n_iter=5_000,
        seed=30,
        log_prior_fn=_standard_normal_log_prior,
    )
    samples = chain[500:, 0]

    assert np.all(accepted)
    assert all(state.acceptance_probability == 1.0 for state in states)
    assert abs(float(np.mean(samples))) < 0.06
    assert abs(float(np.var(samples)) - 1.0) < 0.08


def test_pcn_without_log_prior_raises_value_error() -> None:
    sampler = metropolis_hastings(
        log_density_fn=_standard_normal_log_prior,
        proposal_fn=make_pcn_proposal(beta=0.5),
        theta0=np.zeros(1, dtype=np.float64),
        n_iter=1,
        rng=np.random.default_rng(40),
    )

    with pytest.raises(ValueError, match="prior-preserving proposals require"):
        list(sampler)


def test_pcn_with_full_posterior_does_not_double_count_prior() -> None:
    posterior_mean = np.array([0.8], dtype=np.float64)
    posterior_var = 0.36
    posterior_cov_inv = np.array([[1.0 / posterior_var]], dtype=np.float64)

    def log_target(theta: np.ndarray) -> float:
        return _gaussian_log_density(theta, posterior_mean, posterior_cov_inv)

    chain, accepted, _ = _run_chain(
        log_target,
        make_pcn_proposal(beta=0.7),
        np.array([0.0], dtype=np.float64),
        n_iter=30_000,
        seed=50,
        log_prior_fn=_standard_normal_log_prior,
    )
    samples = chain[5_000:, 0]

    assert abs(float(np.mean(samples)) - posterior_mean[0]) < 0.07
    assert abs(float(np.var(samples)) - posterior_var) < 0.05
    assert 0.25 < float(np.mean(accepted)) < 0.95


def test_random_walk_acceptance_decreases_as_beta_increases() -> None:
    _, accepted_small, _ = _run_chain(
        _standard_normal_log_prior,
        make_random_walk_proposal(beta=0.2),
        np.zeros(1, dtype=np.float64),
        n_iter=4_000,
        seed=60,
    )
    _, accepted_large, _ = _run_chain(
        _standard_normal_log_prior,
        make_random_walk_proposal(beta=3.0),
        np.zeros(1, dtype=np.float64),
        n_iter=4_000,
        seed=60,
    )

    small_rate = float(np.mean(accepted_small))
    large_rate = float(np.mean(accepted_large))
    assert small_rate > large_rate + 0.30


def test_same_seed_reproduces_same_chain() -> None:
    args = (
        _standard_normal_log_prior,
        make_random_walk_proposal(beta=0.7),
        np.zeros(2, dtype=np.float64),
        40,
        70,
    )
    chain_a, accepted_a, _ = _run_chain(*args)
    chain_b, accepted_b, _ = _run_chain(*args)

    np.testing.assert_allclose(chain_a, chain_b)
    np.testing.assert_array_equal(accepted_a, accepted_b)


def test_callback_is_called_once_per_fully_consumed_iteration() -> None:
    seen: list[int] = []

    def callback(state: MCMCState) -> None:
        seen.append(state.iteration)

    states = list(
        metropolis_hastings(
            log_density_fn=_standard_normal_log_prior,
            proposal_fn=make_random_walk_proposal(beta=0.4),
            theta0=np.zeros(1, dtype=np.float64),
            n_iter=7,
            rng=np.random.default_rng(80),
            callback=callback,
        )
    )

    assert len(states) == 7
    assert seen == list(range(1, 8))


def test_proposals_to_impossible_region_are_rejected() -> None:
    def bounded_target(theta: np.ndarray) -> float:
        if abs(float(theta[0])) > 1.0:
            return -np.inf
        return 0.0

    def impossible_proposal(
        theta: np.ndarray, rng: np.random.Generator
    ) -> ProposalResult:
        del theta, rng
        return ProposalResult(
            theta=np.array([2.0], dtype=np.float64),
            log_q_forward=0.0,
            log_q_reverse=0.0,
            prior_preserving=False,
        )

    states = list(
        metropolis_hastings(
            log_density_fn=bounded_target,
            proposal_fn=impossible_proposal,
            theta0=np.zeros(1, dtype=np.float64),
            n_iter=3,
            rng=np.random.default_rng(90),
        )
    )

    assert not any(state.accepted for state in states)
    assert all(state.acceptance_probability == 0.0 for state in states)
    for state in states:
        np.testing.assert_allclose(state.theta, np.zeros(1, dtype=np.float64))


def test_stationarity_sanity_for_one_step_random_walk_mh() -> None:
    rng = np.random.default_rng(100)
    initial = rng.standard_normal((5_000, 2), dtype=np.float64)
    after = np.empty_like(initial)

    for idx, theta0 in enumerate(initial):
        state = next(
            metropolis_hastings(
                log_density_fn=_standard_normal_log_prior,
                proposal_fn=make_random_walk_proposal(beta=0.8),
                theta0=theta0,
                n_iter=1,
                rng=rng,
            )
        )
        after[idx, :] = state.theta

    np.testing.assert_allclose(
        np.mean(after, axis=0), np.mean(initial, axis=0), atol=0.05
    )
    np.testing.assert_allclose(
        np.cov(after, rowvar=False), np.cov(initial, rowvar=False), atol=0.06
    )


def test_collect_chain_empty_and_nonempty() -> None:
    empty_chain, empty_accepted = collect_chain([])
    assert empty_chain.shape == (0, 0)
    assert empty_accepted.shape == (0,)

    states = list(
        metropolis_hastings(
            log_density_fn=_standard_normal_log_prior,
            proposal_fn=make_random_walk_proposal(beta=0.4),
            theta0=np.zeros(1, dtype=np.float64),
            n_iter=2,
            rng=np.random.default_rng(110),
        )
    )
    chain, accepted = collect_chain(states)
    assert chain.shape == (2, 1)
    assert accepted.shape == (2,)
