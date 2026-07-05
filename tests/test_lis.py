"""Correctness gates for the LIS accelerator (src/mcmc_multiscale/lis.py).

These encode the project's standard verification discipline for a new sampler:

  * adjoint Jacobian == finite-difference Jacobian (the cheap-gradient route is
    the same operator, only cheaper);
  * no-data invariance: with no likelihood the LIS kernel leaves the prior
    N(0, I) invariant (and with rank 0 it *is* global pCN);
  * linear-Gaussian exactness: on a linear forward map the Laplace reference is
    exact, so every proposal is accepted (alpha == 1) and the chain draws i.i.d.
    from the analytic Gaussian posterior -- recovered to MC error.
"""

from __future__ import annotations

import numpy as np

from mcmc_multiscale.config import Config
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs
from mcmc_multiscale.mcmc import metropolis_hastings
from mcmc_multiscale.observations import make_truth
from mcmc_multiscale.proposals import make_pcn_proposal
from mcmc_multiscale import lis


# --------------------------------------------------------------------------- #
def test_adjoint_matches_finite_difference() -> None:
    cfg = Config(nx=20, ny=20, n_global_modes=24, n_obs_x=5, n_obs_y=5)
    truth = make_truth(cfg)
    _, _, _, _, pts = cell_centered_grid(cfg.nx, cfg.ny)
    Phi, lam = top_eigenpairs(
        exp_covariance(pts, cfg.sigma, cfg.corr_length), cfg.n_global_modes
    )
    Ja = lis.darcy_jacobian_adjoint(cfg, Phi, lam, truth.theta_true, truth.sensor_idx)
    Jf = lis.darcy_jacobian_fd(cfg, Phi, lam, truth.theta_true, truth.sensor_idx)
    assert np.linalg.norm(Ja - Jf) / np.linalg.norm(Jf) < 1e-5


# --------------------------------------------------------------------------- #
def _sample(proposal, log_post, theta0, n_iter, seed, log_prior=None):
    rng = np.random.default_rng(seed)
    chain, acc = [], 0
    for st in metropolis_hastings(
        log_post, proposal, theta0, n_iter, rng, log_prior_fn=log_prior
    ):
        chain.append(st.theta.copy())
        acc += st.accepted
    return np.asarray(chain), acc / n_iter


def test_no_data_invariance_leaves_prior() -> None:
    """No likelihood: the LIS kernel must leave the prior N(0, I) invariant."""
    N = 12
    rng = np.random.default_rng(0)
    # arbitrary informed basis, prior-scale reference (post_var=1, mean=0) -> reference == prior
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    sub = lis.InformedSubspace(
        U=Q[:, :5],
        mean=np.zeros(N),
        post_var=np.ones(5),
        eig=np.zeros(5),
        source="test",
    )
    proposal = lis.make_lis_proposal(sub, beta_informed=0.6, beta_complement=0.6)
    log_post = lambda th: -0.5 * float(th @ th)  # prior only  # noqa: E731
    chain, acc = _sample(proposal, log_post, np.zeros(N), 40_000, seed=1)
    burn = 5_000
    X = chain[burn:]
    assert abs(acc - 1.0) < 1e-9  # reference == target -> always accept
    assert np.max(np.abs(X.mean(0))) < 0.1
    cov = np.cov(X, rowvar=False)
    assert np.max(np.abs(cov - np.eye(N))) < 0.15


def test_rank0_is_global_pcn() -> None:
    """Rank-0 LIS proposal IS the global pCN map, and its MH correction equals
    pCN's prior-difference correction (reference == prior)."""
    N = 8
    theta = np.random.default_rng(2).standard_normal(N)
    beta = 0.3
    sub0 = lis.InformedSubspace(
        U=np.zeros((N, 0)),
        mean=np.zeros(N),
        post_var=np.zeros(0),
        eig=np.zeros(0),
        source="t",
    )
    p_lis = lis.make_lis_proposal(sub0, 0.5, beta)(theta, np.random.default_rng(7))
    p_pcn = make_pcn_proposal(beta)(theta, np.random.default_rng(7))
    assert np.allclose(p_lis.theta, p_pcn.theta, atol=1e-12)  # same proposal map
    # reference == prior => (log_q_reverse - log_q_forward) == prior(theta) - prior(theta')
    prior = lambda th: -0.5 * float(th @ th)  # noqa: E731
    assert (
        abs(
            (p_lis.log_q_reverse - p_lis.log_q_forward)
            - (prior(theta) - prior(p_lis.theta))
        )
        < 1e-12
    )


def test_linear_gaussian_exactness() -> None:
    """Linear forward map: Laplace reference is exact -> alpha==1, exact posterior."""
    N, n_obs = 16, 10
    rng = np.random.default_rng(3)
    L = rng.standard_normal((n_obs, N))
    sig = 0.5
    theta_star = rng.standard_normal(N)
    y = L @ theta_star + sig * rng.standard_normal(n_obs)

    # analytic Gaussian posterior
    H = (L.T @ L) / sig**2
    Sigma = np.linalg.inv(np.eye(N) + H)
    m = Sigma @ (L.T @ y) / sig**2

    sub = lis.informed_subspace_from_hessian(
        H, mean=m, mu_threshold=1e-9
    )  # exact Laplace

    def log_post(th: np.ndarray) -> float:
        return -0.5 * th @ th - 0.5 * np.sum((L @ th - y) ** 2) / sig**2

    proposal = lis.make_lis_proposal(sub, beta_informed=1.0, beta_complement=0.5)
    chain, acc = _sample(proposal, log_post, m.copy(), 60_000, seed=5)
    X = chain[2_000:]
    assert acc > 0.999  # exact reference -> (essentially) always accept
    assert np.max(np.abs(X.mean(0) - m)) < 0.05
    assert np.max(np.abs(np.cov(X, rowvar=False) - Sigma)) < 0.05


if __name__ == "__main__":
    test_adjoint_matches_finite_difference()
    print("adjoint == FD            : PASS")
    test_no_data_invariance_leaves_prior()
    print("no-data invariance       : PASS")
    test_rank0_is_global_pcn()
    print("rank-0 == global pCN     : PASS")
    test_linear_gaussian_exactness()
    print("linear-Gaussian exactness: PASS")
    print("ALL LIS GATES PASS")
