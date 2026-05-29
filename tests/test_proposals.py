from __future__ import annotations

import inspect

import numpy as np
import pytest

from mcmc_multiscale.proposals import (
    ProposalResult,
    make_pcn_proposal,
    make_random_walk_proposal,
    pcn,
    random_walk,
)


def test_pcn_preserves_shape_and_metadata() -> None:
    theta = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    result = pcn(theta, beta=0.3, rng=np.random.default_rng(1))

    assert result.theta.shape == theta.shape
    assert result.theta.dtype == np.float64
    assert result.log_q_forward == 0.0
    assert result.log_q_reverse == 0.0
    assert result.prior_preserving


def test_pcn_is_deterministic_with_fixed_seed() -> None:
    theta = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    result_a = pcn(theta, beta=0.7, rng=np.random.default_rng(123))
    result_b = pcn(theta, beta=0.7, rng=np.random.default_rng(123))

    np.testing.assert_allclose(result_a.theta, result_b.theta)


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.1, np.inf])
def test_pcn_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="pCN beta"):
        pcn(np.zeros(2, dtype=np.float64), beta=beta, rng=np.random.default_rng(1))


def test_random_walk_preserves_shape_and_metadata() -> None:
    theta = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    result = random_walk(theta, beta=0.3, rng=np.random.default_rng(1))

    assert result.theta.shape == theta.shape
    assert result.theta.dtype == np.float64
    assert result.log_q_forward == 0.0
    assert result.log_q_reverse == 0.0
    assert not result.prior_preserving


def test_random_walk_is_deterministic_with_fixed_seed() -> None:
    theta = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    result_a = random_walk(theta, beta=0.7, rng=np.random.default_rng(123))
    result_b = random_walk(theta, beta=0.7, rng=np.random.default_rng(123))

    np.testing.assert_allclose(result_a.theta, result_b.theta)


@pytest.mark.parametrize("beta", [0.0, -0.1, np.nan])
def test_random_walk_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="random-walk beta"):
        random_walk(
            np.zeros(2, dtype=np.float64), beta=beta, rng=np.random.default_rng(1)
        )


def test_proposal_factories_return_two_argument_engine_callables() -> None:
    theta = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(5)

    for factory in (make_pcn_proposal, make_random_walk_proposal):
        proposal = factory(0.5)
        assert len(inspect.signature(proposal).parameters) == 2
        result = proposal(theta, rng)
        assert isinstance(result, ProposalResult)
        assert result.theta.shape == theta.shape
