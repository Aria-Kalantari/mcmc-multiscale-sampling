"""Hand-rolled NUTS gold-standard reference sampler (SPEC 3.10 / M13).

Samples the *exact* posterior over the whitened global KLE coefficients
``theta ~ N(0, I_N)`` (``G = Phi sqrt(lam) theta``) via Hamiltonian dynamics on
the potential

    U(theta) = 0.5 ||theta||^2 + Phi(theta),
    grad U(theta) = theta + J^T r / sigma_obs^2,

with misfit ``Phi(theta) = 0.5 / sigma_obs^2 ||R(p(theta)) - y||^2``,
``r = R(p(theta)) - y`` and ``J = dR(p(theta))/dtheta`` the adjoint Jacobian.
The role is a *trusted converged reference* (target R-hat <~ 1.01), not an
accelerator claim.

The sampler is the No-U-Turn Sampler (Hoffman & Gelman 2014) with multinomial
(log-weight) trajectory sampling (Betancourt 2017), a leapfrog integrator, a
recursive no-U-turn tree, and dual-averaging step-size adaptation. An optional
diagonal mass matrix is supported; the default is the identity metric, which is
natural because ``theta`` is already whitened (the prior contributes exactly
``I`` to the precision).

Everything numerical is reused from the validated code:
  * ``U`` uses :func:`mcmc_multiscale.bayes.misfit`;
  * the full-Jacobian gradient reuses :func:`lis.darcy_jacobian_adjoint`
    (validated against :func:`lis.darcy_jacobian_fd`);
  * the cheap single-adjoint-solve gradient reuses ``lis._resid_dir_deriv`` and
    keeps the ``-(Lam^T Gmat)`` sign convention (``grad = theta - gm / s2``).

Pure, seeded (explicit :class:`numpy.random.Generator`), float64, no globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy.sparse.linalg import splu

from mcmc_multiscale import lis
from mcmc_multiscale.bayes import misfit
from mcmc_multiscale.config import Config
from mcmc_multiscale.field import reshape_field
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.forward.tpfa import _assemble_system
from mcmc_multiscale.observations import restrict_pressure

ValueAndGrad = Callable[[np.ndarray], "tuple[float, np.ndarray]"]


# --------------------------------------------------------------------------- #
# Potential and gradient closures
# --------------------------------------------------------------------------- #
def make_potential(
    cfg: Config,
    Phi: np.ndarray,
    lam: np.ndarray,
    sensor_idx: np.ndarray,
    y_obs: np.ndarray,
    fwd: ForwardModel | None = None,
    *,
    gradient: str = "full_jacobian",
) -> tuple[
    Callable[[np.ndarray], float],
    Callable[[np.ndarray], np.ndarray],
    ValueAndGrad,
]:
    """Build ``(U, grad_U, value_and_grad)`` for the Darcy posterior.

    ``gradient`` selects the misfit-gradient backend:

    * ``"full_jacobian"`` (default, correctness anchor) forms the validated
      ``J = darcy_jacobian_adjoint`` (1 + N_obs solves) and returns
      ``grad = theta + J^T r / sigma_obs^2``.
    * ``"single_solve"`` (2 solves) forms ``J^T r = -Gmat^T (T^{-1}(R r))``
      directly, reusing the same ``splu`` factor for the forward and adjoint
      solves. Algebraically identical to the full route (gated at 1e-8).

    Returns three callables sharing the same bound problem data. ``U`` costs one
    forward solve; ``value_and_grad`` is the workhorse used by the integrator.
    """
    if gradient not in ("full_jacobian", "single_solve"):
        raise ValueError("gradient must be 'full_jacobian' or 'single_solve'.")

    Phi = np.asarray(Phi, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    y = np.asarray(y_obs, dtype=np.float64)
    sidx = np.asarray(sensor_idx, dtype=np.int64)
    fwd = ForwardModel(cfg) if fwd is None else fwd
    s2 = float(cfg.sigma_obs) ** 2
    sqrt_lam = np.sqrt(lam)
    n = cfg.nx * cfg.ny
    N = Phi.shape[1]

    def U_fn(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        return float(
            0.5 * theta @ theta
            + misfit(theta, Phi, lam, fwd, y, sidx, cfg.sigma_obs, cfg.ny, cfg.nx)
        )

    def _scatter(r: np.ndarray) -> np.ndarray:
        """Inject a sensor residual into the Fortran-flattened cell vector (R r)."""
        z = np.zeros(n, dtype=np.float64)
        z[sidx] = r
        return z

    def _diverge(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return np.inf, np.zeros_like(theta)

    def value_and_grad_full(theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        # Guard: a leapfrog overshoot can push k = exp(G) to overflow or make the
        # TPFA factor singular; signal a divergence (U = inf) instead of raising,
        # so the integrator shrinks the step and the tree terminates cleanly.
        if not np.all(np.isfinite(lis._perm_2d(Phi, lam, theta, cfg))):
            return _diverge(theta)
        try:
            jac = lis.darcy_jacobian_adjoint(cfg, Phi, lam, theta, sidx)
            r = lis._observe(cfg, Phi, lam, theta, sidx) - y
        except (RuntimeError, np.linalg.LinAlgError):
            return _diverge(theta)
        u = 0.5 * float(theta @ theta) + 0.5 * float(r @ r) / s2
        if not np.isfinite(u):
            return _diverge(theta)
        grad = theta + (jac.T @ r) / s2
        return float(u), grad

    def value_and_grad_single(theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        k2d = lis._perm_2d(Phi, lam, theta, cfg)
        if not np.all(np.isfinite(k2d)):  # divergence guard (see full path)
            return _diverge(theta)
        try:
            matrix, rhs = _assemble_system(k2d, cfg)
            lu = splu(matrix.tocsc())
            p = np.asarray(lu.solve(rhs), dtype=np.float64)
            p2d = np.reshape(p, (cfg.ny, cfg.nx), order="F")
            r = restrict_pressure(p2d, sidx) - y
            # adjoint solve (T symmetric -> same factor): w = T^{-1}(R r)
            w = np.asarray(lu.solve(_scatter(r)), dtype=np.float64)
        except (RuntimeError, np.linalg.LinAlgError):
            return _diverge(theta)
        u = 0.5 * float(theta @ theta) + 0.5 * float(r @ r) / s2
        if not np.isfinite(u):
            return _diverge(theta)
        gm = np.empty(N, dtype=np.float64)
        for m in range(N):
            v2d = k2d * reshape_field(sqrt_lam[m] * Phi[:, m], cfg.ny, cfg.nx)
            gm[m] = w @ lis._resid_dir_deriv(k2d, p2d, v2d, cfg)
        grad = theta - gm / s2  # keep -(Lam^T Gmat) sign of darcy_jacobian_adjoint
        return float(u), grad

    value_and_grad = (
        value_and_grad_single if gradient == "single_solve" else value_and_grad_full
    )

    def grad_fn(theta: np.ndarray) -> np.ndarray:
        return value_and_grad(theta)[1]

    return U_fn, grad_fn, value_and_grad


# --------------------------------------------------------------------------- #
# Hamiltonian primitives
# --------------------------------------------------------------------------- #
def _kinetic(r: np.ndarray, inv_mass: np.ndarray) -> float:
    """Kinetic energy ``0.5 r^T M^{-1} r`` for a diagonal metric ``M``."""
    return 0.5 * float(r @ (inv_mass * r))


def leapfrog(
    theta: np.ndarray,
    r: np.ndarray,
    grad: np.ndarray,
    eps: float,
    inv_mass: np.ndarray,
    value_and_grad: ValueAndGrad,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One leapfrog step; returns ``(theta', r', grad', U')``.

    ``r_half = r - (eps/2) grad``; ``theta' = theta + eps M^{-1} r_half``;
    ``r' = r_half - (eps/2) grad'``. The new gradient ``grad'`` is returned so it
    is never recomputed downstream (each gradient is the whole solve cost).
    Direction is folded into a signed ``eps``.
    """
    r_half = r - 0.5 * eps * grad
    theta_new = theta + eps * (inv_mass * r_half)
    u_new, grad_new = value_and_grad(theta_new)
    r_new = r_half - 0.5 * eps * grad_new
    return theta_new, r_new, grad_new, u_new


def _no_u_turn(
    theta_minus: np.ndarray,
    theta_plus: np.ndarray,
    r_minus: np.ndarray,
    r_plus: np.ndarray,
    inv_mass: np.ndarray,
) -> bool:
    """True while the trajectory has not made a U-turn (velocity form, both ends).

    Under the identity metric this reduces to ``dot(theta_plus - theta_minus, r)``
    at both endpoints; the ``inv_mass`` factor is the generalized-velocity form
    required whenever a non-trivial diagonal mass matrix is used.
    """
    dtheta = theta_plus - theta_minus
    return bool(
        dtheta @ (inv_mass * r_minus) >= 0.0 and dtheta @ (inv_mass * r_plus) >= 0.0
    )


# --------------------------------------------------------------------------- #
# Recursive no-U-turn tree (multinomial / log-weight)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Tree:
    """A built subtree: both endpoints (+ their gradients), a sampled candidate,
    its log weight, the summed Metropolis accept statistic, and validity flags."""

    theta_minus: np.ndarray
    r_minus: np.ndarray
    grad_minus: np.ndarray
    theta_plus: np.ndarray
    r_plus: np.ndarray
    grad_plus: np.ndarray
    theta_cand: np.ndarray
    grad_cand: np.ndarray
    u_cand: float
    logw: float
    n_alpha: int
    sum_alpha: float
    n_leapfrog: int
    valid: bool
    diverged: bool


def _build_tree(
    theta: np.ndarray,
    r: np.ndarray,
    grad: np.ndarray,
    v: int,
    depth: int,
    eps: float,
    h0: float,
    inv_mass: np.ndarray,
    value_and_grad: ValueAndGrad,
    rng: np.random.Generator,
    max_delta: float,
) -> _Tree:
    """Recursively build a balanced subtree of depth ``depth`` in direction ``v``."""
    if depth == 0:
        theta1, r1, grad1, u1 = leapfrog(
            theta, r, grad, v * eps, inv_mass, value_and_grad
        )
        h = u1 + _kinetic(r1, inv_mass)
        dh = h - h0
        diverged = (not np.isfinite(dh)) or (dh > max_delta)
        logw = -np.inf if diverged else -dh
        alpha = 0.0 if not np.isfinite(dh) else float(min(1.0, np.exp(-dh)))
        return _Tree(
            theta_minus=theta1,
            r_minus=r1,
            grad_minus=grad1,
            theta_plus=theta1,
            r_plus=r1,
            grad_plus=grad1,
            theta_cand=theta1,
            grad_cand=grad1,
            u_cand=u1,
            logw=logw,
            n_alpha=1,
            sum_alpha=alpha,
            n_leapfrog=1,
            valid=not diverged,
            diverged=diverged,
        )

    left = _build_tree(
        theta, r, grad, v, depth - 1, eps, h0, inv_mass, value_and_grad, rng, max_delta
    )
    if not left.valid:
        return left  # early stop; carries the diverged flag upward

    if v == -1:
        right = _build_tree(
            left.theta_minus,
            left.r_minus,
            left.grad_minus,
            v,
            depth - 1,
            eps,
            h0,
            inv_mass,
            value_and_grad,
            rng,
            max_delta,
        )
        theta_minus, r_minus, grad_minus = (
            right.theta_minus,
            right.r_minus,
            right.grad_minus,
        )
        theta_plus, r_plus, grad_plus = left.theta_plus, left.r_plus, left.grad_plus
    else:
        right = _build_tree(
            left.theta_plus,
            left.r_plus,
            left.grad_plus,
            v,
            depth - 1,
            eps,
            h0,
            inv_mass,
            value_and_grad,
            rng,
            max_delta,
        )
        theta_plus, r_plus, grad_plus = (
            right.theta_plus,
            right.r_plus,
            right.grad_plus,
        )
        theta_minus, r_minus, grad_minus = (
            left.theta_minus,
            left.r_minus,
            left.grad_minus,
        )

    logw = float(np.logaddexp(left.logw, right.logw))
    # multinomial pick within the subtree: choose right with prob w_right / w_total
    if right.valid and np.log(rng.uniform()) < (right.logw - logw):
        theta_cand, grad_cand, u_cand = right.theta_cand, right.grad_cand, right.u_cand
    else:
        theta_cand, grad_cand, u_cand = left.theta_cand, left.grad_cand, left.u_cand

    valid = right.valid and _no_u_turn(
        theta_minus, theta_plus, r_minus, r_plus, inv_mass
    )
    return _Tree(
        theta_minus=theta_minus,
        r_minus=r_minus,
        grad_minus=grad_minus,
        theta_plus=theta_plus,
        r_plus=r_plus,
        grad_plus=grad_plus,
        theta_cand=theta_cand,
        grad_cand=grad_cand,
        u_cand=u_cand,
        logw=logw,
        n_alpha=left.n_alpha + right.n_alpha,
        sum_alpha=left.sum_alpha + right.sum_alpha,
        n_leapfrog=left.n_leapfrog + right.n_leapfrog,
        valid=valid,
        diverged=left.diverged or right.diverged,
    )


def _nuts_transition(
    theta: np.ndarray,
    u: float,
    grad: np.ndarray,
    eps: float,
    inv_mass: np.ndarray,
    mass_std: np.ndarray,
    value_and_grad: ValueAndGrad,
    rng: np.random.Generator,
    max_tree_depth: int,
    max_delta: float,
) -> tuple[np.ndarray, float, np.ndarray, int, int, float, bool]:
    """One NUTS transition by biased progressive doubling.

    Returns ``(theta, U, grad, tree_depth, n_leapfrog, accept_stat, diverged)``.
    The selected state's gradient is returned so the next transition reuses it.
    """
    r0 = mass_std * rng.standard_normal(theta.size)
    h0 = u + _kinetic(r0, inv_mass)

    theta_minus = theta_plus = theta
    r_minus = r_plus = r0
    grad_minus = grad_plus = grad

    theta_s, grad_s, u_s = theta, grad, u
    logw_tree = 0.0  # initial point weight exp(-(h0 - h0)) = 1
    depth = 0
    n_leapfrog = 0
    sum_alpha = 0.0
    n_alpha = 0
    diverged = False

    while depth < max_tree_depth:
        v = -1 if rng.uniform() < 0.5 else 1
        if v == -1:
            tree = _build_tree(
                theta_minus,
                r_minus,
                grad_minus,
                v,
                depth,
                eps,
                h0,
                inv_mass,
                value_and_grad,
                rng,
                max_delta,
            )
            theta_minus, r_minus, grad_minus = (
                tree.theta_minus,
                tree.r_minus,
                tree.grad_minus,
            )
        else:
            tree = _build_tree(
                theta_plus,
                r_plus,
                grad_plus,
                v,
                depth,
                eps,
                h0,
                inv_mass,
                value_and_grad,
                rng,
                max_delta,
            )
            theta_plus, r_plus, grad_plus = tree.theta_plus, tree.r_plus, tree.grad_plus

        n_leapfrog += tree.n_leapfrog
        sum_alpha += tree.sum_alpha
        n_alpha += tree.n_alpha
        diverged = diverged or tree.diverged
        if not tree.valid:
            break

        # biased progressive sampling: favour the newly added subtree
        if np.log(rng.uniform()) < min(0.0, tree.logw - logw_tree):
            theta_s, grad_s, u_s = tree.theta_cand, tree.grad_cand, tree.u_cand
        logw_tree = float(np.logaddexp(logw_tree, tree.logw))

        if not _no_u_turn(theta_minus, theta_plus, r_minus, r_plus, inv_mass):
            break
        depth += 1

    accept_stat = sum_alpha / max(1, n_alpha)
    return theta_s, u_s, grad_s, depth, n_leapfrog, accept_stat, diverged


# --------------------------------------------------------------------------- #
# Step-size initialisation and dual averaging
# --------------------------------------------------------------------------- #
def find_reasonable_epsilon(
    value_and_grad: ValueAndGrad,
    theta: np.ndarray,
    u: float,
    grad: np.ndarray,
    inv_mass: np.ndarray,
    mass_std: np.ndarray,
    rng: np.random.Generator,
    *,
    max_iters: int = 100,
) -> tuple[float, int]:
    """Heuristic initial step size (Hoffman & Gelman Algorithm 4).

    Returns ``(step_size, n_leapfrog_used)`` so the warmup solve budget can be
    accounted (each leapfrog is one gradient evaluation).
    """
    eps = 1.0
    n_lf = 0
    r = mass_std * rng.standard_normal(theta.size)
    h0 = u + _kinetic(r, inv_mass)
    _, r1, _, u1 = leapfrog(theta, r, grad, eps, inv_mass, value_and_grad)
    n_lf += 1
    log_ratio = h0 - (u1 + _kinetic(r1, inv_mass))  # = -(h1 - h0)
    if not np.isfinite(log_ratio):
        log_ratio = -np.inf
    a = 1.0 if log_ratio > np.log(0.5) else -1.0
    for _ in range(max_iters):
        if a * log_ratio <= -a * np.log(2.0):
            break
        eps *= 2.0**a
        _, r1, _, u1 = leapfrog(theta, r, grad, eps, inv_mass, value_and_grad)
        n_lf += 1
        log_ratio = h0 - (u1 + _kinetic(r1, inv_mass))
        if not np.isfinite(log_ratio):
            log_ratio = -np.inf
    return float(eps), n_lf


@dataclass
class _DualAverage:
    """Nesterov dual-averaging step-size adaptation (Hoffman & Gelman Algorithm 5)."""

    mu: float
    target_accept: float = 0.8
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75
    bar_h: float = 0.0
    log_eps_bar: float = 0.0
    step: int = 0

    def update(self, accept_stat: float) -> float:
        """Consume one iteration's mean accept statistic; return the next step size."""
        self.step += 1
        m = self.step
        eta_h = 1.0 / (m + self.t0)
        self.bar_h = (1.0 - eta_h) * self.bar_h + eta_h * (
            self.target_accept - accept_stat
        )
        log_eps = self.mu - np.sqrt(m) / self.gamma * self.bar_h
        eta = m ** (-self.kappa)
        self.log_eps_bar = eta * log_eps + (1.0 - eta) * self.log_eps_bar
        return float(np.exp(log_eps))

    def averaged(self) -> float:
        """The dual-averaged step size to freeze for the sampling phase."""
        return float(np.exp(self.log_eps_bar))


# --------------------------------------------------------------------------- #
# Result record and driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NUTSState:
    """One yielded post-warmup NUTS state (mirrors :class:`mcmc.MCMCState`)."""

    iteration: int
    theta: np.ndarray
    potential: float
    log_density: float  # = -potential (parity with MCMCState)
    tree_depth: int
    n_leapfrog: int
    accept_prob: float
    step_size: float
    diverged: bool


def nuts_sample(
    value_and_grad: ValueAndGrad,
    theta0: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    *,
    n_warmup: int,
    step_size0: float | None = None,
    target_accept: float = 0.8,
    mass_matrix: np.ndarray | None = None,
    max_tree_depth: int = 10,
    max_delta: float = 1000.0,
    adapt_mass: bool = False,
    stats: dict[str, int] | None = None,
) -> Iterator[NUTSState]:
    """Yield ``n_iter`` post-warmup states from a NUTS chain.

    ``value_and_grad(theta) -> (U, grad_U)`` defines the target potential.
    Warmup runs internally: ``find_reasonable_epsilon`` (unless ``step_size0`` is
    given) then dual-averaging step-size adaptation over ``n_warmup`` iterations,
    after which the step size is frozen to its dual-averaged value. The metric is
    the identity unless ``mass_matrix`` (a diagonal, shape ``(N,)``) is supplied;
    ``adapt_mass`` estimates a single diagonal metric from warmup variance.

    If a mutable ``stats`` dict is supplied, it is populated with the warmup solve
    budget: ``"warmup_leapfrogs"`` (gradient evals over warmup transitions),
    ``"find_eps_leapfrogs"``, and ``"init_grads"`` (= 1). Each leapfrog and each
    initial evaluation is one ``value_and_grad`` call.

    All randomness is drawn from ``rng`` in a fixed order, so a fixed seed gives a
    bit-for-bit identical chain.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if n_warmup < 0:
        raise ValueError("n_warmup must be non-negative.")

    theta = np.asarray(theta0, dtype=np.float64)
    if theta.ndim != 1:
        raise ValueError("theta0 must be a one-dimensional array.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta0 must contain only finite values.")
    theta = theta.copy()
    N = theta.size

    if mass_matrix is None:
        inv_mass = np.ones(N, dtype=np.float64)
    else:
        mass = np.asarray(mass_matrix, dtype=np.float64)
        if mass.shape != (N,) or np.any(mass <= 0.0):
            raise ValueError("mass_matrix must be a positive (N,) diagonal.")
        inv_mass = 1.0 / mass
    mass_std = np.sqrt(1.0 / inv_mass)

    u, grad = value_and_grad(theta)

    find_eps_leapfrogs = 0
    if step_size0 is not None:
        eps = float(step_size0)
    else:
        eps, find_eps_leapfrogs = find_reasonable_epsilon(
            value_and_grad, theta, u, grad, inv_mass, mass_std, rng
        )
    dual = _DualAverage(mu=np.log(10.0 * eps), target_accept=target_accept)
    warmup_leapfrogs = 0
    if stats is not None:  # defaults (overwritten at the warmup boundary)
        stats["warmup_leapfrogs"] = 0
        stats["find_eps_leapfrogs"] = find_eps_leapfrogs
        stats["init_grads"] = 1

    # Welford accumulators for optional diagonal mass adaptation.
    mass_n = 0
    mass_mean = np.zeros(N, dtype=np.float64)
    mass_m2 = np.zeros(N, dtype=np.float64)
    mass_updated = False

    total = n_warmup + n_iter
    for m in range(1, total + 1):
        theta, u, grad, depth, n_lf, accept_stat, diverged = _nuts_transition(
            theta,
            u,
            grad,
            eps,
            inv_mass,
            mass_std,
            value_and_grad,
            rng,
            max_tree_depth,
            max_delta,
        )

        if m <= n_warmup:
            warmup_leapfrogs += n_lf
            eps = dual.update(accept_stat)
            if adapt_mass:
                mass_n += 1
                delta = theta - mass_mean
                mass_mean += delta / mass_n
                mass_m2 += delta * (theta - mass_mean)
                # re-estimate the metric once, at the warmup midpoint
                if (not mass_updated) and m == n_warmup // 2 and mass_n > 1:
                    var = mass_m2 / (mass_n - 1)
                    inv_mass = 1.0 / np.clip(var, 1e-8, None)
                    mass_std = np.sqrt(1.0 / inv_mass)
                    u, grad = value_and_grad(theta)
                    eps, extra_lf = find_reasonable_epsilon(
                        value_and_grad, theta, u, grad, inv_mass, mass_std, rng
                    )
                    find_eps_leapfrogs += extra_lf
                    dual = _DualAverage(
                        mu=np.log(10.0 * eps), target_accept=target_accept
                    )
                    mass_updated = True
            if m == n_warmup:
                eps = dual.averaged()
                if stats is not None:
                    stats["warmup_leapfrogs"] = warmup_leapfrogs
                    stats["find_eps_leapfrogs"] = find_eps_leapfrogs
                    stats["init_grads"] = 1
            continue

        yield NUTSState(
            iteration=m - n_warmup,
            theta=theta.copy(),
            potential=float(u),
            log_density=float(-u),
            tree_depth=int(depth),
            n_leapfrog=int(n_lf),
            accept_prob=float(accept_stat),
            step_size=float(eps),
            diverged=bool(diverged),
        )


def collect_nuts(
    states: list[NUTSState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect states into ``(theta_chain (n, N), diverged (n,), step_size (n,))``."""
    if not states:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float64),
        )
    chain = np.stack([s.theta for s in states]).astype(np.float64, copy=False)
    diverged = np.asarray([s.diverged for s in states], dtype=bool)
    step_size = np.asarray([s.step_size for s in states], dtype=np.float64)
    return chain, diverged, step_size
