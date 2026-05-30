"""Correctness gates for the precision-based block-Gibbs update.

See ``docs/conditioned_posterior_derivation.md`` (sections 4.1-4.5 and 7). The
two decisive gates are no-data invariance of ``N(0, C_tau)`` and linear-Gaussian
exactness; the rest pin down the Woodbury precision, pCN reversibility, the
reduction cases, and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.conditioning.gaussian_block import (
    build_precision,
    conditional_mean,
    make_block_conditioner,
    propose_block,
)
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs
from mcmc_multiscale.sampler import block_gibbs_sampler
from mcmc_multiscale.subdomain import iter_coarse_subdomains, make_subdomain_at


def _orthonormal(rng: np.random.Generator, n: int, n_modes: int) -> np.ndarray:
    """Return an ``(n, n_modes)`` matrix with Euclidean-orthonormal columns."""

    q, _ = np.linalg.qr(rng.standard_normal((n, n_modes)))
    return np.ascontiguousarray(q, dtype=np.float64)


def _dense_c_tau(Phi: np.ndarray, lam: np.ndarray, tau2: float) -> np.ndarray:
    return Phi @ np.diag(lam) @ Phi.T + tau2 * np.eye(Phi.shape[0])


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    quad = diff @ np.linalg.solve(cov, diff)
    return float(-0.5 * (x.size * np.log(2.0 * np.pi) + logdet + quad))


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


def _global_kle(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    covariance = exp_covariance(points, cfg.sigma, cfg.corr_length)
    return top_eigenpairs(covariance, cfg.n_global_modes)


# --- Woodbury precision ------------------------------------------------------


def test_woodbury_precision_matches_dense_inverse_and_submatrix() -> None:
    rng = np.random.default_rng(0)
    n, n_modes = 10, 4
    Phi = _orthonormal(rng, n, n_modes)
    lam = np.array([3.0, 1.6, 0.7, 0.2], dtype=np.float64)
    tau2 = 1e-2
    precision = build_precision(Phi, lam, tau2)
    q_dense = np.linalg.inv(_dense_c_tau(Phi, lam, tau2))

    np.testing.assert_allclose(precision.dense(), q_dense, atol=1e-9)
    g = rng.standard_normal(n)
    np.testing.assert_allclose(precision.matvec(g), q_dense @ g, atol=1e-9)
    # Matvec also acts column-wise on a stack.
    stack = rng.standard_normal((n, 3))
    np.testing.assert_allclose(precision.matvec(stack), q_dense @ stack, atol=1e-9)

    idx = np.array([1, 3, 5, 7], dtype=np.int64)
    np.testing.assert_allclose(
        precision.submatrix(idx), q_dense[np.ix_(idx, idx)], atol=1e-9
    )
    block = make_block_conditioner(precision, idx)
    np.testing.assert_allclose(
        block.chol_lower @ block.chol_lower.T, q_dense[np.ix_(idx, idx)], atol=1e-9
    )


def test_build_precision_rejects_nonpositive_nugget() -> None:
    Phi = np.eye(3, dtype=np.float64)
    lam = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError, match="tau2"):
        build_precision(Phi, lam, 0.0)


# --- conditional mean (regression form) --------------------------------------


def test_conditional_mean_matches_covariance_weighted_regression() -> None:
    rng = np.random.default_rng(5)
    n, n_modes = 9, 3
    Phi = _orthonormal(rng, n, n_modes)
    lam = np.array([2.5, 1.0, 0.3], dtype=np.float64)
    tau2 = 5e-2
    precision = build_precision(Phi, lam, tau2)
    idx = np.array([0, 2, 4, 6], dtype=np.int64)
    rest = np.array([i for i in range(n) if i not in idx], dtype=np.int64)
    block = make_block_conditioner(precision, idx)
    g = rng.standard_normal(n)

    q_dense = precision.dense()
    q_ss = q_dense[np.ix_(idx, idx)]
    q_sr = q_dense[np.ix_(idx, rest)]
    expected = -np.linalg.solve(q_ss, q_sr @ g[rest])  # -Q_SS^{-1} Q_SR g_R

    np.testing.assert_allclose(
        conditional_mean(precision, block, g), expected, atol=1e-9
    )


# --- pCN-around-conditional reversibility ------------------------------------


def test_pcn_around_conditional_satisfies_detailed_balance() -> None:
    rng = np.random.default_rng(1)
    n, n_modes = 9, 3
    Phi = _orthonormal(rng, n, n_modes)
    lam = np.array([2.5, 1.0, 0.3], dtype=np.float64)
    tau2 = 5e-2
    precision = build_precision(Phi, lam, tau2)
    idx = np.array([0, 2, 4, 6], dtype=np.int64)
    block = make_block_conditioner(precision, idx)
    g = rng.standard_normal(n)
    mean = conditional_mean(precision, block, g)
    cov = np.linalg.inv(precision.submatrix(idx))
    beta = 0.4
    alpha = np.sqrt(1.0 - beta**2)

    for _ in range(50):
        x = rng.standard_normal(idx.size)
        y = rng.standard_normal(idx.size)
        forward = _mvn_logpdf(x, mean, cov) + _mvn_logpdf(
            y, mean + alpha * (x - mean), beta**2 * cov
        )
        reverse = _mvn_logpdf(y, mean, cov) + _mvn_logpdf(
            x, mean + alpha * (y - mean), beta**2 * cov
        )
        assert forward == pytest.approx(reverse, abs=1e-9)


def test_propose_block_realises_the_analytic_transition_kernel() -> None:
    rng = np.random.default_rng(11)
    n, n_modes = 9, 3
    Phi = _orthonormal(rng, n, n_modes)
    lam = np.array([2.5, 1.0, 0.3], dtype=np.float64)
    tau2 = 5e-2
    precision = build_precision(Phi, lam, tau2)
    idx = np.array([0, 2, 4, 6], dtype=np.int64)
    block = make_block_conditioner(precision, idx)
    g = rng.standard_normal(n)
    mean = conditional_mean(precision, block, g)
    cov = np.linalg.inv(precision.submatrix(idx))
    beta = 0.4
    alpha = np.sqrt(1.0 - beta**2)

    x = rng.standard_normal(idx.size)
    g_fixed = g.copy()
    g_fixed[idx] = x  # m_S depends only on the (unchanged) exterior
    draws = np.array(
        [propose_block(precision, block, g_fixed, beta, rng) for _ in range(20000)]
    )
    np.testing.assert_allclose(draws.mean(axis=0), mean + alpha * (x - mean), atol=0.03)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), beta**2 * cov, atol=0.03)


def test_propose_block_rejects_invalid_beta() -> None:
    rng = np.random.default_rng(0)
    Phi = _orthonormal(rng, 6, 2)
    precision = build_precision(Phi, np.array([1.0, 0.5]), 1e-2)
    block = make_block_conditioner(precision, np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="beta"):
        propose_block(precision, block, np.zeros(6), 0.0, rng)


# --- gate 1: no-data invariance ---------------------------------------------


def test_no_data_sweep_leaves_prior_invariant() -> None:
    cfg = Config(
        nx=8,
        ny=8,
        n_coarse_x=2,
        n_coarse_y=2,
        overlap_cells=1,
        n_global_modes=12,
    )
    Phi, lam = _global_kle(cfg)
    tau2 = 1e-2
    precision = build_precision(Phi, lam, tau2)
    n = cfg.nx * cfg.ny
    conditioners = [
        make_block_conditioner(
            precision, make_subdomain_at(cfg, row, col).core_global_idx
        )
        for row, col in iter_coarse_subdomains(cfg)
    ]

    # The cores must partition every cell for the sweep to be a valid Gibbs scan.
    all_idx = np.concatenate([cond.global_idx for cond in conditioners])
    np.testing.assert_array_equal(np.sort(all_idx), np.arange(n))

    c_tau = _dense_c_tau(Phi, lam, tau2)
    chol_c = np.linalg.cholesky(c_tau)
    rng = np.random.default_rng(2)
    n_samples = 4000
    samples = (chol_c @ rng.standard_normal((n, n_samples))).T.copy()

    beta = 1.0  # independent conditional draws: exact systematic-scan Gibbs
    for s in range(n_samples):
        g = samples[s]
        for cond in conditioners:
            g[cond.global_idx] = propose_block(precision, cond, g, beta, rng)

    assert np.max(np.abs(samples.mean(axis=0))) < 0.1
    assert np.max(np.abs(np.cov(samples, rowvar=False) - c_tau)) < 0.15


# --- gate 2: linear-Gaussian exactness --------------------------------------


def test_linear_gaussian_sweep_recovers_analytic_posterior() -> None:
    rng = np.random.default_rng(3)
    n, n_modes = 6, 3
    Phi = _orthonormal(rng, n, n_modes)
    lam = np.array([2.0, 1.0, 0.4], dtype=np.float64)
    tau2 = 0.05
    precision = build_precision(Phi, lam, tau2)
    q_dense = precision.dense()
    blocks = [
        make_block_conditioner(precision, np.array([0, 1, 2], dtype=np.int64)),
        make_block_conditioner(precision, np.array([3, 4, 5], dtype=np.int64)),
    ]

    n_obs = 4
    sigma = 0.5
    obs_operator = 0.7 * rng.standard_normal((n_obs, n))
    y_obs = rng.standard_normal(n_obs)
    sigma_post = np.linalg.inv(q_dense + obs_operator.T @ obs_operator / sigma**2)
    mu_post = sigma_post @ (obs_operator.T @ y_obs / sigma**2)

    def misfit(g: np.ndarray) -> float:
        residual = obs_operator @ g - y_obs
        return 0.5 * float(residual @ residual) / sigma**2

    beta = 0.6
    g = np.zeros(n)
    current_misfit = misfit(g)
    n_sweeps, burn, thin = 50_000, 10_000, 4
    collected: list[np.ndarray] = []
    for sweep in range(n_sweeps):
        for block in blocks:
            candidate = g.copy()
            candidate[block.global_idx] = propose_block(precision, block, g, beta, rng)
            candidate_misfit = misfit(candidate)
            log_alpha = current_misfit - candidate_misfit  # likelihood-only
            if np.log(rng.uniform()) < min(0.0, log_alpha):
                g = candidate
                current_misfit = candidate_misfit
        if sweep >= burn and (sweep - burn) % thin == 0:
            collected.append(g.copy())

    samples = np.asarray(collected)
    np.testing.assert_allclose(samples.mean(axis=0), mu_post, atol=0.04)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), sigma_post, atol=0.05)


# --- reduction cases ---------------------------------------------------------


def test_whole_domain_block_reduces_to_global_pcn() -> None:
    rng = np.random.default_rng(4)
    n, n_modes = 8, 3
    Phi = _orthonormal(rng, n, n_modes)
    lam = np.array([2.0, 0.9, 0.3], dtype=np.float64)
    tau2 = 0.02
    precision = build_precision(Phi, lam, tau2)
    block = make_block_conditioner(precision, np.arange(n, dtype=np.int64))
    g = rng.standard_normal(n)

    # S = all cells, R = empty: m_S = 0 and Sigma_S = C_tau, i.e. global pCN.
    np.testing.assert_allclose(
        conditional_mean(precision, block, g), np.zeros(n), atol=1e-9
    )
    c_tau = _dense_c_tau(Phi, lam, tau2)
    np.testing.assert_allclose(
        block.chol_lower @ block.chol_lower.T, np.linalg.inv(c_tau), atol=1e-9
    )


def test_block_gibbs_acceptance_is_exactly_likelihood_only() -> None:
    cfg = _small_cfg()
    states = list(
        block_gibbs_sampler(cfg, n_sweeps=8, rng=np.random.default_rng(cfg.seed))
    )
    for previous, current in zip(states[:-1], states[1:]):
        log_alpha = current.log_likelihood_candidate - previous.log_likelihood_accepted
        expected = float(np.exp(min(0.0, log_alpha)))
        assert current.acceptance_probability == pytest.approx(expected)


# --- determinism & wiring ----------------------------------------------------


def test_block_gibbs_is_deterministic_for_fixed_seed() -> None:
    cfg = _small_cfg()
    run_a = list(
        block_gibbs_sampler(cfg, n_sweeps=3, rng=np.random.default_rng(cfg.seed))
    )
    run_b = list(
        block_gibbs_sampler(cfg, n_sweeps=3, rng=np.random.default_rng(cfg.seed))
    )

    assert len(run_a) == len(run_b) == 3 * cfg.n_coarse_x * cfg.n_coarse_y
    for state_a, state_b in zip(run_a, run_b):
        assert state_a.accepted == state_b.accepted
        np.testing.assert_allclose(state_a.G_accepted, state_b.G_accepted)
        assert state_a.theta_norm_accepted == pytest.approx(state_b.theta_norm_accepted)


def test_block_gibbs_keeps_the_field_bounded_near_the_prior_scale() -> None:
    cfg = _small_cfg()
    states = list(
        block_gibbs_sampler(cfg, n_sweeps=60, rng=np.random.default_rng(cfg.seed))
    )
    max_ratio = max(state.theta_norm_accepted / state.expected_norm for state in states)
    assert max_ratio < 1.5


def test_config_exposes_block_nugget_default() -> None:
    assert Config().block_nugget == pytest.approx(1e-4)
