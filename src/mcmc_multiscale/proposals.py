"""Proposal kernels for Metropolis-Hastings samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProposalResult:
    """A proposed state plus MH proposal-density metadata."""

    theta: np.ndarray
    log_q_forward: float
    log_q_reverse: float
    prior_preserving: bool


def _theta_array(theta: np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1:
        raise ValueError("theta must be a one-dimensional array.")
    return theta_arr


def pcn(theta: np.ndarray, beta: float, rng: np.random.Generator) -> ProposalResult:
    """Return a preconditioned Crank-Nicolson proposal.

    The pCN kernel is reversible with respect to the standard Gaussian prior.
    It therefore returns `prior_preserving=True`; the MH engine is responsible
    for subtracting the prior difference when the supplied log density is a full
    posterior.
    """

    if not np.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("pCN beta must satisfy 0 < beta <= 1.")
    theta_arr = _theta_array(theta)
    xi = rng.standard_normal(theta_arr.shape, dtype=np.float64)
    theta_prop = np.sqrt(1.0 - beta**2) * theta_arr + beta * xi
    return ProposalResult(
        theta=theta_prop.astype(np.float64, copy=False),
        log_q_forward=0.0,
        log_q_reverse=0.0,
        prior_preserving=True,
    )


def random_walk(
    theta: np.ndarray, beta: float, rng: np.random.Generator
) -> ProposalResult:
    """Return a symmetric Gaussian random-walk proposal."""

    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("random-walk beta must be positive.")
    theta_arr = _theta_array(theta)
    xi = rng.standard_normal(theta_arr.shape, dtype=np.float64)
    return ProposalResult(
        theta=(theta_arr + beta * xi).astype(np.float64, copy=False),
        log_q_forward=0.0,
        log_q_reverse=0.0,
        prior_preserving=False,
    )


def make_pcn_proposal(
    beta: float,
) -> Callable[[np.ndarray, np.random.Generator], ProposalResult]:
    """Return a two-argument pCN proposal callable for the MH engine."""

    def proposal(theta: np.ndarray, rng: np.random.Generator) -> ProposalResult:
        return pcn(theta, beta, rng)

    return proposal


def make_random_walk_proposal(
    beta: float,
) -> Callable[[np.ndarray, np.random.Generator], ProposalResult]:
    """Return a two-argument random-walk proposal callable for the MH engine."""

    def proposal(theta: np.ndarray, rng: np.random.Generator) -> ProposalResult:
        return random_walk(theta, beta, rng)

    return proposal
