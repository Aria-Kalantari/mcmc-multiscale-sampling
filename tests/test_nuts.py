"""Correctness gates for the NUTS reference sampler (src/mcmc_multiscale/nuts.py).

These encode the project's standard verification discipline for a new sampler:

  * gradient of U matches a finite-difference gradient (and the cheap
    single-adjoint-solve gradient equals the validated full-Jacobian gradient);
  * no-data invariance: with U(theta) = 0.5||theta||^2 the chain samples the
    prior N(0, I) -- recovered to MC error;
  * linear-Gaussian exactness: on a linear forward map NUTS reproduces the
    analytic mu_post, Sigma_post to MC error (the decisive trust gate);
  * determinism: a fixed seed gives a bit-for-bit identical chain.
"""

from __future__ import annotations

import numpy as np

from mcmc_multiscale.config import Config
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs
from mcmc_multiscale.nuts import make_potential, nuts_sample
from mcmc_multiscale.observations import make_truth


# --------------------------------------------------------------------------- #
def _fd_grad(u_fn, theta: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of a scalar potential."""
    g = np.empty(theta.size, dtype=np.float64)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        g[i] = (u_fn(tp) - u_fn(tm)) / (2.0 * h)
    return g


def _sample_nuts(value_and_grad, theta0, n_iter, n_warmup, seed):
    """Run a NUTS chain and return ``(theta_chain, divergence_count)``."""
    rng = np.random.default_rng(seed)
    chain, diverged = [], 0
    for st in nuts_sample(value_and_grad, theta0, n_iter, rng, n_warmup=n_warmup):
        chain.append(st.theta.copy())
        diverged += int(st.diverged)
    return np.asarray(chain), diverged


# --------------------------------------------------------------------------- #
def test_grad_matches_fd_and_single_matches_full() -> None:
    """grad U == finite difference, and single-solve == full-Jacobian gradient."""
    cfg = Config(nx=16, ny=16, n_global_modes=12, n_obs_x=4, n_obs_y=4)
    truth = make_truth(cfg)
    _, _, _, _, pts = cell_centered_grid(cfg.nx, cfg.ny)
    Phi, lam = top_eigenpairs(
        exp_covariance(pts, cfg.sigma, cfg.corr_length), cfg.n_global_modes
    )
    u_full, _, vg_full = make_potential(
        cfg, Phi, lam, truth.sensor_idx, truth.y_obs, gradient="full_jacobian"
    )
    _, _, vg_single = make_potential(
        cfg, Phi, lam, truth.sensor_idx, truth.y_obs, gradient="single_solve"
    )
    theta = truth.theta_true

    g_fd = _fd_grad(u_full, theta)
    _, g_full = vg_full(theta)
    _, g_single = vg_single(theta)

    assert np.linalg.norm(g_full - g_fd) / np.linalg.norm(g_fd) < 1e-5
    # single-solve is algebraically identical to the full-Jacobian route
    assert np.linalg.norm(g_single - g_full) / np.linalg.norm(g_full) < 1e-8
    assert np.linalg.norm(g_single - g_fd) / np.linalg.norm(g_fd) < 1e-5


def test_no_data_recovers_standard_normal() -> None:
    """No likelihood: U(theta) = 0.5||theta||^2 -> the chain samples N(0, I)."""
    N = 8

    def value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        return 0.5 * float(theta @ theta), theta

    chain, diverged = _sample_nuts(
        value_and_grad, np.zeros(N), n_iter=4000, n_warmup=500, seed=1
    )
    assert diverged == 0
    assert np.max(np.abs(chain.mean(axis=0))) < 0.1
    cov = np.cov(chain, rowvar=False)
    assert np.max(np.abs(cov - np.eye(N))) < 0.15


def test_linear_gaussian_exactness() -> None:
    """Linear forward map: NUTS reproduces the analytic Gaussian posterior."""
    N, n_obs = 6, 10
    rng = np.random.default_rng(3)
    L = rng.standard_normal((n_obs, N))
    sig = 0.5
    theta_star = rng.standard_normal(N)
    y = L @ theta_star + sig * rng.standard_normal(n_obs)

    # analytic Gaussian posterior (identity prior precision)
    Sigma = np.linalg.inv(np.eye(N) + (L.T @ L) / sig**2)
    m = Sigma @ (L.T @ y) / sig**2

    def value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        r = L @ theta - y
        u = 0.5 * float(theta @ theta) + 0.5 * float(r @ r) / sig**2
        grad = theta + (L.T @ r) / sig**2
        return u, grad

    chain, diverged = _sample_nuts(
        value_and_grad, m.copy(), n_iter=8000, n_warmup=1000, seed=5
    )
    assert diverged == 0
    assert np.max(np.abs(chain.mean(axis=0) - m)) < 0.05
    assert np.max(np.abs(np.cov(chain, rowvar=False) - Sigma)) < 0.05


def test_determinism_for_fixed_seed() -> None:
    """A fixed seed gives a bit-for-bit identical chain (and step size)."""
    N, n_obs = 6, 10
    rng = np.random.default_rng(3)
    L = rng.standard_normal((n_obs, N))
    sig = 0.5
    theta_star = rng.standard_normal(N)
    y = L @ theta_star + sig * rng.standard_normal(n_obs)

    def value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        r = L @ theta - y
        u = 0.5 * float(theta @ theta) + 0.5 * float(r @ r) / sig**2
        return u, theta + (L.T @ r) / sig**2

    def run() -> tuple[np.ndarray, float]:
        gen = np.random.default_rng(11)
        states = list(nuts_sample(value_and_grad, np.zeros(N), 300, gen, n_warmup=200))
        chain = np.stack([s.theta for s in states])
        return chain, states[-1].step_size

    chain1, eps1 = run()
    chain2, eps2 = run()
    np.testing.assert_allclose(chain1, chain2, rtol=0.0, atol=0.0)
    assert eps1 == eps2


if __name__ == "__main__":
    test_grad_matches_fd_and_single_matches_full()
    print("grad == FD, single == full : PASS")
    test_no_data_recovers_standard_normal()
    print("no-data => N(0, I)         : PASS")
    test_linear_gaussian_exactness()
    print("linear-Gaussian exactness  : PASS")
    test_determinism_for_fixed_seed()
    print("determinism                : PASS")
    print("ALL NUTS GATES PASS")
