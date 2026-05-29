"""Generic Metropolis-Hastings engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from mcmc_multiscale.proposals import ProposalResult


@dataclass(frozen=True)
class MCMCState:
    """One yielded Metropolis-Hastings state."""

    iteration: int
    theta: np.ndarray
    log_density: float
    accepted: bool
    acceptance_probability: float


def _as_theta(theta: np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1:
        raise ValueError("theta0 must be a one-dimensional array.")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta0 must contain only finite values.")
    return theta_arr.copy()


def _eval_log_density(
    fn: Callable[[np.ndarray], float], theta: np.ndarray, label: str
) -> float:
    value = float(fn(theta))
    if np.isnan(value):
        raise ValueError(f"{label} log density evaluated to nan.")
    return value


def _acceptance_probability(log_alpha: float) -> float:
    if np.isnan(log_alpha):
        raise ValueError("MH log acceptance ratio evaluated to nan.")
    if log_alpha >= 0.0:
        return 1.0
    if np.isneginf(log_alpha):
        return 0.0
    return float(np.exp(log_alpha))


def metropolis_hastings(
    log_density_fn: Callable[[np.ndarray], float],
    proposal_fn: Callable[[np.ndarray, np.random.Generator], ProposalResult],
    theta0: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    log_prior_fn: Callable[[np.ndarray], float] | None = None,
    callback: Callable[[MCMCState], None] | None = None,
) -> Iterator[MCMCState]:
    """Yield states from a generic Metropolis-Hastings chain.

    If a proposal returns `prior_preserving=True`, the supplied `log_density_fn`
    is interpreted as a full posterior and `log_prior_fn` is required. The
    acceptance ratio then subtracts the prior difference, yielding the pCN
    likelihood-ratio correction and avoiding prior double-counting.
    """

    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    theta_current = _as_theta(theta0)
    log_density_current = _eval_log_density(log_density_fn, theta_current, "initial")

    for iteration in range(1, n_iter + 1):
        proposal = proposal_fn(theta_current.copy(), rng)
        theta_prop = _as_theta(proposal.theta)
        if theta_prop.shape != theta_current.shape:
            raise ValueError("proposal returned theta with incompatible shape.")

        log_density_prop = _eval_log_density(log_density_fn, theta_prop, "proposed")

        if proposal.prior_preserving and log_prior_fn is None:
            raise ValueError(
                "prior-preserving proposals require log_prior_fn to avoid "
                "double-counting the prior."
            )

        if np.isneginf(log_density_prop):
            log_alpha = -np.inf
        else:
            log_alpha = (
                log_density_prop
                - log_density_current
                + float(proposal.log_q_reverse)
                - float(proposal.log_q_forward)
            )
            if proposal.prior_preserving:
                if log_prior_fn is None:
                    raise ValueError("log_prior_fn is required for pCN proposals.")
                log_prior_current = _eval_log_density(
                    log_prior_fn, theta_current, "current prior"
                )
                log_prior_prop = _eval_log_density(
                    log_prior_fn, theta_prop, "proposed prior"
                )
                log_alpha -= log_prior_prop - log_prior_current

        accept_prob = _acceptance_probability(log_alpha)
        accepted = bool(np.log(rng.uniform()) < min(0.0, log_alpha))

        if accepted:
            theta_current = theta_prop.copy()
            log_density_current = log_density_prop

        state = MCMCState(
            iteration=iteration,
            theta=theta_current.copy(),
            log_density=float(log_density_current),
            accepted=accepted,
            acceptance_probability=accept_prob,
        )
        yield state
        if callback is not None:
            callback(state)


def collect_chain(states: list[MCMCState]) -> tuple[np.ndarray, np.ndarray]:
    """Collect a list of states into `(theta_chain, accepted)` arrays."""

    if not states:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=bool),
        )
    chain = np.stack([state.theta for state in states]).astype(np.float64, copy=False)
    accepted = np.asarray([state.accepted for state in states], dtype=bool)
    return chain, accepted
