"""M3 Gaussian-target sanity checks for the generic MCMC engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.mcmc import collect_chain, metropolis_hastings  # noqa: E402
from mcmc_multiscale.proposals import (  # noqa: E402
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
) -> tuple[np.ndarray, np.ndarray]:
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
    return collect_chain(states)


def _random_walk_gaussian() -> tuple[np.ndarray, np.ndarray, float]:
    target_mean = np.array([1.0, -0.5], dtype=np.float64)
    target_cov = np.array([[1.0, 0.3], [0.3, 0.6]], dtype=np.float64)
    cov_inv = np.linalg.inv(target_cov)
    chain, accepted = _run_chain(
        lambda theta: _gaussian_log_density(theta, target_mean, cov_inv),
        make_random_walk_proposal(beta=0.8),
        np.zeros(2, dtype=np.float64),
        n_iter=20_000,
        seed=300,
    )
    samples = chain[4_000:, :]
    return (
        np.mean(samples, axis=0),
        np.cov(samples, rowvar=False),
        float(np.mean(accepted)),
    )


def _pcn_standard_normal() -> tuple[np.ndarray, np.ndarray, float]:
    chain, accepted = _run_chain(
        _standard_normal_log_prior,
        make_pcn_proposal(beta=1.0),
        np.zeros(2, dtype=np.float64),
        n_iter=6_000,
        seed=301,
        log_prior_fn=_standard_normal_log_prior,
    )
    samples = chain[500:, :]
    return (
        np.mean(samples, axis=0),
        np.cov(samples, rowvar=False),
        float(np.mean(accepted)),
    )


def _pcn_posterior_sanity() -> tuple[float, float, float]:
    posterior_mean = np.array([0.8], dtype=np.float64)
    posterior_var = 0.36
    posterior_cov_inv = np.array([[1.0 / posterior_var]], dtype=np.float64)

    def log_target(theta: np.ndarray) -> float:
        return _gaussian_log_density(theta, posterior_mean, posterior_cov_inv)

    chain, accepted = _run_chain(
        log_target,
        make_pcn_proposal(beta=0.7),
        np.zeros(1, dtype=np.float64),
        n_iter=30_000,
        seed=302,
        log_prior_fn=_standard_normal_log_prior,
    )
    samples = chain[5_000:, 0]
    return float(np.mean(samples)), float(np.var(samples)), float(np.mean(accepted))


def _acceptance_response() -> tuple[float, float]:
    _, accepted_small = _run_chain(
        _standard_normal_log_prior,
        make_random_walk_proposal(beta=0.2),
        np.zeros(1, dtype=np.float64),
        n_iter=4_000,
        seed=303,
    )
    _, accepted_large = _run_chain(
        _standard_normal_log_prior,
        make_random_walk_proposal(beta=3.0),
        np.zeros(1, dtype=np.float64),
        n_iter=4_000,
        seed=303,
    )
    return float(np.mean(accepted_small)), float(np.mean(accepted_large))


def _stationarity_sanity() -> tuple[float, float]:
    rng = np.random.default_rng(304)
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

    mean_shift = float(
        np.max(np.abs(np.mean(after, axis=0) - np.mean(initial, axis=0)))
    )
    cov_shift = float(
        np.max(np.abs(np.cov(after, rowvar=False) - np.cov(initial, rowvar=False)))
    )
    return mean_shift, cov_shift


def main() -> None:
    target_mean = np.array([1.0, -0.5], dtype=np.float64)
    target_cov = np.array([[1.0, 0.3], [0.3, 0.6]], dtype=np.float64)
    rw_mean, rw_cov, rw_accept = _random_walk_gaussian()
    pcn_mean, pcn_cov, pcn_accept = _pcn_standard_normal()
    posterior_mean, posterior_var, posterior_accept = _pcn_posterior_sanity()
    small_accept, large_accept = _acceptance_response()
    stationarity_mean_shift, stationarity_cov_shift = _stationarity_sanity()

    print("M3 MCMC GAUSSIAN SANITY")
    print(f"Target mean: {target_mean}")
    print(f"Random-walk empirical mean: {rw_mean}")
    print(f"Target covariance:\n{target_cov}")
    print(f"Random-walk empirical covariance:\n{rw_cov}")
    print(f"Random-walk acceptance rate: {rw_accept:.4f}")
    print()
    print("pCN standard-normal target mean: [0. 0.]")
    print(f"pCN standard-normal empirical mean: {pcn_mean}")
    print("pCN standard-normal target covariance:\n[[1. 0.]\n [0. 1.]]")
    print(f"pCN standard-normal empirical covariance:\n{pcn_cov}")
    print(f"pCN standard-normal acceptance rate: {pcn_accept:.4f}")
    print()
    print("pCN full-posterior correction sanity:")
    print("  target posterior mean: 0.8000")
    print(f"  empirical posterior mean: {posterior_mean:.4f}")
    print("  target posterior variance: 0.3600")
    print(f"  empirical posterior variance: {posterior_var:.4f}")
    print(f"  pCN posterior acceptance rate: {posterior_accept:.4f}")
    print()
    print("Random-walk beta response:")
    print(f"  beta=0.2 acceptance rate: {small_accept:.4f}")
    print(f"  beta=3.0 acceptance rate: {large_accept:.4f}")
    print()
    print("One-step stationarity sanity from standard-normal initial states:")
    print(f"  max mean shift: {stationarity_mean_shift:.4e}")
    print(f"  max covariance shift: {stationarity_cov_shift:.4e}")


if __name__ == "__main__":
    main()
