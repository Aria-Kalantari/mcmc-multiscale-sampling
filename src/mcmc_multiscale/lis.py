"""Likelihood-Informed Subspace (LIS) acceleration for the Darcy inverse problem.

Motivation (see docs/lis_method.md). Global pCN is dimension-robust but mixes
slowly in the **likelihood-informed directions**: where the sparse pressure data
constrains the field, the posterior is far narrower than the prior, so a
prior-scale pCN step overshoots and is rejected there. For this problem the
informed subspace is low-dimensional (the smoothing forward map + few sensors
constrain only a handful of large-scale modes), so the chain's autocorrelation
time is set by a small number of modes. This module fixes exactly those modes.

Method: split the (whitened, prior N(0,I)) KLE coordinate space  R^N  into an
**informed** subspace  U  (r columns) and its orthogonal complement.

* Informed block: propose by pCN **about the Laplace approximation**
  N(m_r, diag(s_i^2)) restricted to U  (s_i = posterior std in mode i). With
  beta_informed = 1 this is an independence sampler (near-iid in those modes);
  with beta_informed < 1 it is robust pCN-about-Laplace.
* Complement block: ordinary pCN about the prior N(0, I) (posterior approx prior
  there), step beta_complement.

Because U and its complement are orthogonal subspaces of an *isotropic* Gaussian
prior, the two blocks are prior-independent (no Schur complement, unlike a dense
spatial-block precision). The product kernel is reversible w.r.t. the Gaussian
reference  q* = N(m_r, diag s^2)_U (x) N(0,I)_perp; the Metropolis correction
targets the **true** posterior exactly, at one forward solve per step:

    log alpha = [log pi(theta') - log q*(theta')] - [log pi(theta) - log q*(theta)]

The informed subspace is built two ways (the experiment compares them):
  * `informed_subspace_from_samples`  -- gradient-free, from a pilot pCN sample
    covariance (production path);
  * `informed_subspace_from_hessian` / `build_informed_subspace_adjoint` -- from
    the prior-preconditioned Gauss-Newton Hessian via a cheap Darcy adjoint at
    the MAP (one-time validation that the gradient-free subspace is right).

Correctness reductions (verified in tests/test_lis.py):
  * no data / empty subspace (r=0)  ->  exactly global pCN (likelihood-ratio
    acceptance);
  * linear-Gaussian forward + exact Laplace scales  ->  acceptance == 1, draws
    are i.i.d. from the analytic Gaussian posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse.linalg import splu

from mcmc_multiscale.config import Config
from mcmc_multiscale.field import field_from_theta, reshape_field
from mcmc_multiscale.forward.tpfa import _assemble_system
from mcmc_multiscale.observations import restrict_pressure
from mcmc_multiscale.proposals import ProposalResult


# --------------------------------------------------------------------------- #
# Darcy adjoint: prior-preconditioned Gauss-Newton Hessian of the misfit
# --------------------------------------------------------------------------- #
def _perm_2d(
    Phi: np.ndarray, lam: np.ndarray, theta: np.ndarray, cfg: Config
) -> np.ndarray:
    return np.exp(reshape_field(field_from_theta(Phi, lam, theta), cfg.ny, cfg.nx))


def _resid_dir_deriv(
    k2d: np.ndarray, p2d: np.ndarray, v2d: np.ndarray, cfg: Config
) -> np.ndarray:
    """Directional derivative  d/ds [ T(k+s v) p - b(k+s v) ]|_{s=0}  (p fixed).

    Matches `tpfa._assemble_system` exactly (harmonic-mean faces, Dirichlet
    left/right, Fortran flattening). Returned as a flat (n,) vector.
    """
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    g = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
    # x-faces (col, col+1)
    ka, kb = k2d[:, :-1], k2d[:, 1:]
    va, vb = v2d[:, :-1], v2d[:, 1:]
    dT = (2.0 * (kb**2 * va + ka**2 * vb) / (ka + kb) ** 2) * (dy / dx)
    pa, pb = p2d[:, :-1], p2d[:, 1:]
    g[:, :-1] += dT * (pa - pb)
    g[:, 1:] += dT * (pb - pa)
    # y-faces (row, row+1)
    ka, kb = k2d[:-1, :], k2d[1:, :]
    va, vb = v2d[:-1, :], v2d[1:, :]
    dT = (2.0 * (kb**2 * va + ka**2 * vb) / (ka + kb) ** 2) * (dx / dy)
    pa, pb = p2d[:-1, :], p2d[1:, :]
    g[:-1, :] += dT * (pa - pb)
    g[1:, :] += dT * (pb - pa)
    # Dirichlet boundaries (half-cell transmissibility 2*k*dy/dx)
    g[:, 0] += 2.0 * dy / dx * v2d[:, 0] * (p2d[:, 0] - cfg.left_pressure)
    g[:, -1] += 2.0 * dy / dx * v2d[:, -1] * (p2d[:, -1] - cfg.right_pressure)
    return g.ravel(order="F")


def darcy_jacobian_adjoint(
    cfg: Config,
    Phi: np.ndarray,
    lam: np.ndarray,
    theta: np.ndarray,
    sensor_idx: np.ndarray,
) -> np.ndarray:
    """Jacobian  J = d R(p(theta)) / d theta  in R^{N_obs x N}, via the adjoint.

    Cost = 1 forward + N_obs adjoint solves of the *same* factorized operator
    (a single `splu`), i.e. O(N_obs) solves -- the cheap-gradient route. The
    result is identical (to round-off) to `darcy_jacobian_fd`; see tests.
    """
    k2d = _perm_2d(Phi, lam, theta, cfg)
    A, b = _assemble_system(k2d, cfg)
    lu = splu(A.tocsc())
    p = np.asarray(lu.solve(b), dtype=np.float64)
    p2d = np.reshape(p, (cfg.ny, cfg.nx), order="F")
    n = cfg.nx * cfg.ny
    n_obs = sensor_idx.size
    N = Phi.shape[1]
    R = np.zeros((n, n_obs))
    R[np.asarray(sensor_idx, dtype=np.int64), np.arange(n_obs)] = 1.0
    Lam = lu.solve(R)  # (n, N_obs); T is symmetric so adjoint uses same factor
    sqrt_lam = np.sqrt(lam)
    Gmat = np.empty((n, N), dtype=np.float64)
    for m in range(N):
        v2d = k2d * reshape_field(sqrt_lam[m] * Phi[:, m], cfg.ny, cfg.nx)
        Gmat[:, m] = _resid_dir_deriv(k2d, p2d, v2d, cfg)
    return -(Lam.T @ Gmat)


def darcy_jacobian_fd(
    cfg: Config,
    Phi: np.ndarray,
    lam: np.ndarray,
    theta: np.ndarray,
    sensor_idx: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference Jacobian (verification only; 2N forward solves)."""
    N = Phi.shape[1]

    def observe(th: np.ndarray) -> np.ndarray:
        A, b = _assemble_system(_perm_2d(Phi, lam, th, cfg), cfg)
        p = np.reshape(
            np.asarray(splu(A.tocsc()).solve(b)), (cfg.ny, cfg.nx), order="F"
        )
        return restrict_pressure(p, sensor_idx)

    J = np.empty((sensor_idx.size, N), dtype=np.float64)
    for m in range(N):
        tp, tm = theta.copy(), theta.copy()
        tp[m] += eps
        tm[m] -= eps
        J[:, m] = (observe(tp) - observe(tm)) / (2.0 * eps)
    return J


def gn_hessian(J: np.ndarray, sigma_obs: float) -> np.ndarray:
    """Prior-preconditioned Gauss-Newton Hessian  H = J^T J / sigma_obs^2.

    theta is already whitened (prior N(0,I)), so H lives in prior-preconditioned
    coordinates and the Laplace posterior covariance is  (I + H)^{-1}.
    """
    return (J.T @ J) / float(sigma_obs) ** 2


def _observe(
    cfg: Config,
    Phi: np.ndarray,
    lam: np.ndarray,
    theta: np.ndarray,
    sensor_idx: np.ndarray,
) -> np.ndarray:
    """One forward solve, restricted to sensors (Fortran flatten, matches bayes)."""
    A, b = _assemble_system(_perm_2d(Phi, lam, theta, cfg), cfg)
    p = np.reshape(np.asarray(splu(A.tocsc()).solve(b)), (cfg.ny, cfg.nx), order="F")
    return restrict_pressure(p, sensor_idx)


def gauss_newton_map(
    cfg: Config,
    Phi: np.ndarray,
    lam: np.ndarray,
    sensor_idx: np.ndarray,
    y_obs: np.ndarray,
    theta0: np.ndarray | None = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> tuple[np.ndarray, int]:
    """Levenberg-Marquardt MAP of  F(theta) = 0.5||theta||^2 + misfit(theta).

    Damped Gauss-Newton with backtracking on the LM parameter from theta=0 (the
    well-conditioned k=1 state); a plain GN step diverges from a dispersed start
    because k = exp(G) is strongly nonlinear in the tails. Returns (theta_map,
    forward_solves_used) so the one-time setup cost is accounted in the budget.
    """
    N = Phi.shape[1]
    theta = np.zeros(N) if theta0 is None else np.asarray(theta0, np.float64).copy()
    y = np.asarray(y_obs, np.float64)
    s2 = float(cfg.sigma_obs) ** 2
    eye = np.eye(N)
    n_solve = 0

    def objective(t: np.ndarray) -> float:
        nonlocal n_solve
        r = _observe(cfg, Phi, lam, t, sensor_idx) - y
        n_solve += 1
        return float(0.5 * t @ t + 0.5 * r @ r / s2)

    f = objective(theta)
    lm = 1.0
    for _ in range(max_iter):
        J = darcy_jacobian_adjoint(cfg, Phi, lam, theta, sensor_idx)
        n_solve += 1 + sensor_idx.size
        r = _observe(cfg, Phi, lam, theta, sensor_idx) - y
        grad = theta + J.T @ r / s2
        H = (J.T @ J) / s2
        improved = False
        delta = np.zeros(N)
        for _ls in range(30):
            delta = -np.linalg.solve(eye + H + lm * eye, grad)
            f_new = objective(theta + delta)
            if f_new < f - 1e-12:
                theta = theta + delta
                f = f_new
                lm = max(lm / 3.0, 1e-8)
                improved = True
                break
            lm *= 5.0
        if not improved or np.linalg.norm(delta) < tol:
            break
    return theta, n_solve


def build_informed_subspace_adjoint(
    cfg: Config,
    Phi: np.ndarray,
    lam: np.ndarray,
    theta_ref: np.ndarray,
    sensor_idx: np.ndarray,
    rank: int | None = None,
    mu_threshold: float = 1.0,
) -> tuple["InformedSubspace", int]:
    """Adjoint GN-Hessian informed subspace at `theta_ref`. Returns (subspace, solves)."""
    J = darcy_jacobian_adjoint(cfg, Phi, lam, theta_ref, sensor_idx)
    n_solve = 1 + sensor_idx.size
    H = gn_hessian(J, cfg.sigma_obs)
    sub = informed_subspace_from_hessian(
        H, theta_ref, rank=rank, mu_threshold=mu_threshold
    )
    return sub, n_solve


# --------------------------------------------------------------------------- #
# Informed subspace
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InformedSubspace:
    """Orthonormal informed basis U (N x r), reference mean, posterior scales."""

    U: np.ndarray  # (N, r) orthonormal columns
    mean: np.ndarray  # (N,) reference mean (MAP / pilot mean)
    post_var: np.ndarray  # (r,) posterior variance per informed mode
    eig: np.ndarray  # (r,) GN-Hessian eigenvalue (hessian) or sample variance (pilot)
    source: str

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])


def informed_subspace_from_hessian(
    H: np.ndarray,
    mean: np.ndarray,
    rank: int | None = None,
    mu_threshold: float = 1.0,
) -> InformedSubspace:
    """Top eigenvectors of the GN-Hessian H; posterior var = 1 / (1 + mu)."""
    mu_all, V = np.linalg.eigh(H)  # ascending
    order = np.argsort(mu_all)[::-1]
    mu_all = mu_all[order]
    V = V[:, order]
    if rank is None:
        rank = int(np.sum(mu_all > mu_threshold))
    rank = max(0, min(rank, V.shape[1]))
    U = V[:, :rank]
    mu = np.clip(mu_all[:rank], 0.0, None)
    return InformedSubspace(
        U=U,
        mean=np.asarray(mean, np.float64),
        post_var=1.0 / (1.0 + mu),
        eig=mu,
        source="hessian",
    )


def informed_subspace_from_samples(
    samples: np.ndarray,
    rank: int,
    var_floor: float = 1e-4,
) -> InformedSubspace:
    """Gradient-free informed subspace from pilot samples (shape (n_samples, N)).

    The most informed directions are those of *smallest* posterior variance
    (the chain is most constrained there). Eigen-decompose the sample covariance
    and take the `rank` smallest-variance directions; the reference mean is the
    sample mean and the per-mode posterior variances are the sample eigenvalues.
    """
    X = np.asarray(samples, dtype=np.float64)
    mean = X.mean(axis=0)
    Cov = np.cov(X, rowvar=False)
    var_all, V = np.linalg.eigh(Cov)  # ascending -> smallest variance first
    rank = max(0, min(rank, V.shape[1]))
    U = V[:, :rank]
    post_var = np.clip(var_all[:rank], var_floor, None)
    return InformedSubspace(
        U=U,
        mean=mean,
        post_var=post_var,
        eig=var_all[:rank],
        source="pilot",
    )


def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two orthonormal column spans."""
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Operator-weighted pCN proposal (Laplace reference in the informed subspace)
# --------------------------------------------------------------------------- #
def make_lis_proposal(
    subspace: InformedSubspace,
    beta_informed: float,
    beta_complement: float,
) -> Callable[[np.ndarray, np.random.Generator], ProposalResult]:
    """Return a two-argument LIS proposal for the generic MH engine.

    Informed block: pCN about the Laplace approximation N(m_r, diag(s^2)) on U
    (beta_informed = 1 -> independence sampler). Complement block: pCN about the
    prior on U^perp (step beta_complement). The proposal carries the Gaussian-
    reference log-densities so the engine forms the exact posterior-targeting
    acceptance (prior_preserving=False). With rank 0 it is exactly global pCN.
    """
    if not 0.0 < beta_informed <= 1.0:
        raise ValueError("beta_informed must satisfy 0 < beta_informed <= 1.")
    if not 0.0 < beta_complement <= 1.0:
        raise ValueError("beta_complement must satisfy 0 < beta_complement <= 1.")

    U = np.asarray(subspace.U, dtype=np.float64)
    r = U.shape[1]
    mr = U.T @ subspace.mean  # (r,) informed-coord reference mean
    s = np.sqrt(np.clip(subspace.post_var, 1e-12, None))  # (r,) posterior std
    inv_s2 = 1.0 / s**2
    a_inf = float(np.sqrt(1.0 - beta_informed**2))
    a_comp = float(np.sqrt(1.0 - beta_complement**2))

    def log_qstar(theta: np.ndarray) -> float:
        a = U.T @ theta
        w = theta - U @ a
        return float(-0.5 * (np.sum((a - mr) ** 2 * inv_s2) + w @ w))

    def proposal(theta: np.ndarray, rng: np.random.Generator) -> ProposalResult:
        theta = np.asarray(theta, dtype=np.float64)
        a = U.T @ theta
        w = theta - U @ a
        # informed block: pCN about N(m_r, diag s^2)
        a_new = mr + a_inf * (a - mr) + beta_informed * s * rng.standard_normal(r)
        # complement block: pCN about prior on U^perp
        xi = rng.standard_normal(theta.shape[0])
        xi_perp = xi - U @ (U.T @ xi)
        w_new = a_comp * w + beta_complement * xi_perp
        theta_new = U @ a_new + w_new
        return ProposalResult(
            theta=theta_new.astype(np.float64, copy=False),
            log_q_forward=log_qstar(theta_new),
            log_q_reverse=log_qstar(theta),
            prior_preserving=False,
        )

    return proposal


_LIS_MODULE_COMPLETE = True
