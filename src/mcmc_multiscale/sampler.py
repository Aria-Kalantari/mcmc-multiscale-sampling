"""Headless single-subdomain conditioned sampler for M4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from mcmc_multiscale.bayes import GlobalKLE, log_prior_field
from mcmc_multiscale.conditioning.constraints import (
    build_A,
    build_c,
    select_conditioning_points,
)
from mcmc_multiscale.conditioning.gaussian_block import (
    BlockConditioner,
    build_precision,
    make_block_conditioner,
    propose_block,
)
from mcmc_multiscale.conditioning.nullspace import null_basis, project_null
from mcmc_multiscale.conditioning.particular import lu_pivot, svd_min_norm
from mcmc_multiscale.conditioning.project import stabilize
from mcmc_multiscale.conditioning.soft import soft_project
from mcmc_multiscale.config import Config
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.diagnostics import (
    constraint_residual,
    expected_gaussian_norm,
    interface_jump,
    relative_error,
    theta_norm,
)
from mcmc_multiscale.field import (
    field_from_theta,
    permeability_from_log_field,
    reshape_field,
)
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs
from mcmc_multiscale.observations import make_truth, restrict_pressure
from mcmc_multiscale.subdomain import (
    Subdomain,
    iter_coarse_subdomains,
    make_subdomain,
    make_subdomain_at,
    red_black_order,
)


@dataclass(frozen=True)
class ConditionedSamplerState:
    iteration: int

    G_accepted: np.ndarray
    k_accepted: np.ndarray
    pressure_accepted: np.ndarray
    log_likelihood_accepted: float

    G_candidate: np.ndarray
    k_candidate: np.ndarray
    pressure_candidate: np.ndarray
    log_likelihood_candidate: float

    accepted: bool
    acceptance_probability: float

    theta_local_accepted: np.ndarray
    theta_local_candidate: np.ndarray
    theta_proposed: np.ndarray
    theta_p: np.ndarray
    theta_n_candidate: np.ndarray

    theta_norm_accepted: float
    theta_norm_candidate: float
    theta_p_norm: float
    theta_n_candidate_norm: float
    null_space_norm_candidate: float
    null_space_norm_accepted: float
    hidden_null_norm: float
    expected_norm: float

    constraint_residual_candidate: float
    constraint_residual_accepted: float
    interface_jump_accepted: float
    interface_jump_candidate: float
    relative_k_error_accepted: float | None
    relative_k_error_candidate: float | None

    cond_A: float
    cond_B: float | None


@dataclass(frozen=True)
class RedBlackSamplerState:
    """One local update from the sequential red-black sweep schedule."""

    sweep: int
    color: int
    subdomain_row: int
    subdomain_col: int
    accepted: bool
    acceptance_probability: float
    theta_norm_candidate: float
    theta_norm_accepted: float
    expected_norm: float
    constraint_residual_candidate: float
    interface_jump_candidate: float
    interface_jump_accepted: float
    relative_k_error_accepted: float | None
    log_likelihood_accepted: float
    G_accepted: np.ndarray
    G_candidate: np.ndarray
    cond_A: float
    cond_B: float | None
    theta_p_norm: float
    null_space_norm_candidate: float
    null_space_norm_accepted: float
    hidden_null_norm: float


@dataclass(frozen=True)
class BlockGibbsSamplerState:
    """One core-block full-conditional update from a precision block-Gibbs sweep.

    The yielded diagnostics mirror the existing sampler states closely enough for
    the shared experiment harness: ``G_accepted`` is the current global field,
    ``theta_norm_accepted`` the global-KLE-projected coefficient norm, and the
    acceptance is exactly likelihood-only.
    """

    sweep: int
    block_index: int
    subdomain_row: int
    subdomain_col: int
    accepted: bool
    acceptance_probability: float
    log_likelihood_accepted: float
    log_likelihood_candidate: float
    theta_norm_accepted: float
    expected_norm: float
    block_nugget: float
    relative_k_error_accepted: float | None
    relative_k_error_candidate: float | None
    G_accepted: np.ndarray
    G_candidate: np.ndarray


@dataclass(frozen=True)
class _LocalUpdateData:
    row: int
    col: int
    color: int
    sub: Subdomain
    Phi_ext: np.ndarray
    lambda_local: np.ndarray
    A: np.ndarray
    cond_local_idx: np.ndarray


def _pressure_log_likelihood(
    pressure: np.ndarray,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_obs: float,
) -> float:
    if sigma_obs <= 0.0:
        raise ValueError("sigma_obs must be positive.")
    pred = restrict_pressure(pressure, sensor_idx)
    residual = pred - np.asarray(y_obs, dtype=np.float64)
    return float(-0.5 / sigma_obs**2 * np.dot(residual, residual))


def _proposal_from_current(
    theta: np.ndarray,
    beta: float,
    proposal: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if proposal == "pcn":
        if beta <= 0.0 or beta > 1.0:
            raise ValueError("pCN beta must satisfy 0 < beta <= 1.")
        xi = rng.standard_normal(theta.shape, dtype=np.float64)
        return np.sqrt(1.0 - beta**2) * theta + beta * xi
    if proposal == "random_walk":
        if beta <= 0.0:
            raise ValueError("random-walk beta must be positive.")
        xi = rng.standard_normal(theta.shape, dtype=np.float64)
        return theta + beta * xi
    raise ValueError("proposal must be 'pcn' or 'random_walk'.")


def _acceptance_probability(log_alpha: float) -> float:
    if np.isnan(log_alpha):
        raise ValueError("acceptance log ratio is nan.")
    if log_alpha >= 0.0:
        return 1.0
    if np.isneginf(log_alpha):
        return 0.0
    return float(np.exp(log_alpha))


def _validate_acceptance(acceptance: str) -> None:
    if acceptance not in {"posterior", "likelihood_only"}:
        raise ValueError("acceptance must be 'posterior' or 'likelihood_only'.")


def _validate_prior_mode(prior_mode: str, conditioning_mode: str) -> None:
    if prior_mode not in {"global_field", "null_space"}:
        raise ValueError("prior_mode must be 'global_field' or 'null_space'.")
    if prior_mode == "null_space" and conditioning_mode != "hard":
        raise ValueError("prior_mode='null_space' requires hard conditioning.")


def _null_space_log_prior_ratio(
    theta_current: np.ndarray,
    theta_candidate: np.ndarray,
    Z: np.ndarray,
) -> float:
    """Return the reference-style local null-coordinate log-prior ratio."""

    eta_current = Z.T @ np.asarray(theta_current, dtype=np.float64)
    eta_candidate = Z.T @ np.asarray(theta_candidate, dtype=np.float64)
    return float(
        -0.5 * (np.dot(eta_candidate, eta_candidate) - np.dot(eta_current, eta_current))
    )


def _acceptance_log_ratio(
    log_like_candidate: float,
    log_like_current: float,
    acceptance: str,
    G_candidate_vec: np.ndarray | None = None,
    G_current_vec: np.ndarray | None = None,
    global_kle: GlobalKLE | None = None,
    proposal_correction: float = 0.0,
    prior_mode: str = "global_field",
    theta_current: np.ndarray | None = None,
    theta_candidate: np.ndarray | None = None,
    Z: np.ndarray | None = None,
) -> float:
    """Return the selected debug or posterior-correct acceptance log ratio."""

    _validate_acceptance(acceptance)
    log_alpha = float(log_like_candidate - log_like_current)
    if acceptance == "likelihood_only":
        return log_alpha
    if prior_mode == "null_space":
        if theta_current is None or theta_candidate is None or Z is None:
            raise ValueError(
                "null-space prior acceptance requires local coefficients and Z."
            )
        # The pCN q-ratio is deliberately omitted in this reference-style
        # diagnostic variant. Adding it would cancel this null-prior ratio and
        # silently reduce the update to likelihood-only acceptance.
        return float(
            log_alpha + _null_space_log_prior_ratio(theta_current, theta_candidate, Z)
        )
    if prior_mode != "global_field":
        raise ValueError("prior_mode must be 'global_field' or 'null_space'.")
    if G_candidate_vec is None or G_current_vec is None or global_kle is None:
        raise ValueError("posterior acceptance requires fields and the global KLE.")
    return float(
        log_alpha
        + log_prior_field(G_candidate_vec, global_kle)
        - log_prior_field(G_current_vec, global_kle)
        + proposal_correction
    )


def _null_pcn_proposal_correction(
    theta_current: np.ndarray,
    theta_candidate: np.ndarray,
    Z: np.ndarray,
    proposal: str,
    conditioning_mode: str,
) -> float:
    """Return `log q(reverse) - log q(forward)` for hard null-space pCN."""

    if proposal != "pcn" or conditioning_mode != "hard":
        return 0.0
    eta_current = Z.T @ theta_current
    eta_candidate = Z.T @ theta_candidate
    return float(
        0.5 * (np.dot(eta_candidate, eta_candidate) - np.dot(eta_current, eta_current))
    )


def _particular_solution(
    A: np.ndarray, c: np.ndarray, theta_p_method: str
) -> tuple[np.ndarray, np.ndarray, int, float, float | None]:
    if theta_p_method == "svd":
        theta_p, Z, info = svd_min_norm(A, c)
        return theta_p, Z, info.rankA, info.cond_effective, None
    if theta_p_method == "lu":
        theta_p, info = lu_pivot(A, c)
        return theta_p, null_basis(A), info.rankA, info.cond_effective, info.cond_B
    if theta_p_method == "lu_stabilized":
        theta_p_lu, info = lu_pivot(A, c)
        Z = null_basis(A)
        return (
            stabilize(theta_p_lu, Z),
            Z,
            info.rankA,
            info.cond_effective,
            info.cond_B,
        )
    raise ValueError("theta_p_method must be 'lu', 'svd', or 'lu_stabilized'.")


def _validate_conditioning_options(
    theta_p_method: str,
    conditioning_mode: str,
    rhs_mode: str,
    rho: float | None,
) -> None:
    if rhs_mode not in {"data", "zero"}:
        raise ValueError("rhs_mode must be 'data' or 'zero'.")
    if conditioning_mode not in {"hard", "soft"}:
        raise ValueError("conditioning_mode must be 'hard' or 'soft'.")
    allowed_theta_methods = {"lu", "svd", "lu_stabilized"}
    if conditioning_mode == "soft":
        allowed_theta_methods = allowed_theta_methods | {"soft"}
    if theta_p_method not in allowed_theta_methods:
        raise ValueError(
            "theta_p_method must be 'lu', 'svd', or 'lu_stabilized'"
            + (
                " (or 'soft' for soft conditioning)."
                if conditioning_mode == "soft"
                else "."
            )
        )
    if conditioning_mode == "soft":
        if rho is None or not np.isfinite(float(rho)) or float(rho) <= 0.0:
            raise ValueError("soft conditioning requires rho > 0.")
    elif rho is not None and (not np.isfinite(float(rho)) or float(rho) <= 0.0):
        raise ValueError("rho must be positive when provided.")


def _solve_local_conditioning(
    A: np.ndarray,
    c_used: np.ndarray,
    Next: int,
    theta_p_method: str,
    conditioning_mode: str,
    rhs_mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    if conditioning_mode == "hard" and rhs_mode == "zero":
        theta_p = np.zeros(Next, dtype=np.float64)
        _, Z, _, cond_A, _ = _particular_solution(A, c_used, "svd")
        return theta_p, Z, cond_A, None
    if conditioning_mode == "hard":
        theta_p, Z, _, cond_A, cond_B = _particular_solution(A, c_used, theta_p_method)
        return theta_p, Z, cond_A, cond_B

    _, Z, _, cond_A, _ = _particular_solution(A, c_used, "svd")
    return np.zeros(Next, dtype=np.float64), Z, cond_A, None


def _condition_candidate(
    theta_proposed: np.ndarray,
    A: np.ndarray,
    c_used: np.ndarray,
    theta_p: np.ndarray,
    Z: np.ndarray,
    conditioning_mode: str,
    rhs_mode: str,
    rho: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if conditioning_mode == "soft":
        theta_local_candidate = soft_project(theta_proposed, A, c_used, float(rho))
        return theta_local_candidate, theta_local_candidate.copy(), np.nan

    theta_n_candidate = project_null(Z, theta_proposed)
    hidden_null_norm = 0.0 if rhs_mode == "zero" else theta_norm(Z.T @ theta_p)
    return theta_p + theta_n_candidate, theta_n_candidate, hidden_null_norm


def _precompute_red_black_updates(
    cfg: Config,
    Mb: int,
    global_pts: np.ndarray,
) -> list[_LocalUpdateData]:
    Next = cfg.Nc + Mb
    order = red_black_order(cfg)
    local_data: list[_LocalUpdateData] = []
    for color in (0, 1):
        for row, col in order[color]:
            sub = make_subdomain_at(cfg, row, col)
            local_pts = global_pts[sub.local_global_idx, :]
            C_local = exp_covariance(local_pts, cfg.sigma, cfg.corr_length)
            Phi_local, lambda_local = top_eigenpairs(C_local, Next)
            cond_local_idx = select_conditioning_points(
                local_pts, sub.core_local_idx, sub.buffer_local_idx, Mb
            )
            Phi_ext = Phi_local[:, :Next]
            A = build_A(Phi_ext, np.sqrt(lambda_local), cond_local_idx)
            local_data.append(
                _LocalUpdateData(
                    row=row,
                    col=col,
                    color=color,
                    sub=sub,
                    Phi_ext=Phi_ext,
                    lambda_local=lambda_local,
                    A=A,
                    cond_local_idx=cond_local_idx,
                )
            )
    return local_data


def conditioned_sampler(
    cfg: Config,
    n_iter: int,
    Mb: int,
    theta_p_method: str,
    rng: np.random.Generator,
    beta: float | None = None,
    update_scheme: str = "single",
    proposal: str = "pcn",
    rhs_mode: str = "data",
    conditioning_mode: str = "hard",
    rho: float | None = None,
    acceptance: str = "likelihood_only",
    prior_mode: str | None = None,
) -> Iterator[ConditionedSamplerState]:
    """Run the M4 single-subdomain repeated-conditioning harness.

    The M4 instability path uses `theta_p + Z @ (Z.T @ theta_proposed)`, not
    the shifted affine projection. This intentionally preserves any hidden
    null-space component of an arbitrary LU particular solution. The low-level
    default remains likelihood-only so the M4/M5 reproduction harnesses are
    unchanged; pass `acceptance="posterior"` for the M8 global-prior baseline.
    """

    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if update_scheme != "single":
        raise NotImplementedError("M4 implements only update_scheme='single'.")
    _validate_conditioning_options(theta_p_method, conditioning_mode, rhs_mode, rho)
    _validate_acceptance(acceptance)
    prior_mode_value = cfg.prior_mode if prior_mode is None else prior_mode
    if acceptance == "posterior":
        _validate_prior_mode(prior_mode_value, conditioning_mode)
    beta_value = cfg.beta if beta is None else float(beta)

    truth = make_truth(cfg, rng)
    _, _, _, _, global_pts = cell_centered_grid(cfg.nx, cfg.ny)
    C_global = exp_covariance(global_pts, cfg.sigma, cfg.corr_length)
    Phi_global, lambda_global = top_eigenpairs(C_global, cfg.n_global_modes)
    global_kle = (Phi_global, lambda_global)

    theta_global_current = rng.standard_normal(cfg.n_global_modes, dtype=np.float64)
    G_current_vec = field_from_theta(Phi_global, lambda_global, theta_global_current)
    G_accepted = reshape_field(G_current_vec, cfg.ny, cfg.nx)
    k_accepted = permeability_from_log_field(G_accepted)
    fwd = ForwardModel(cfg)
    pressure_accepted = fwd.solve(k_accepted)
    log_like_accepted = _pressure_log_likelihood(
        pressure_accepted, truth.y_obs, truth.sensor_idx, cfg.sigma_obs
    )

    sub = make_subdomain(cfg)
    local_pts = global_pts[sub.local_global_idx, :]
    Next = cfg.Nc + Mb
    C_local = exp_covariance(local_pts, cfg.sigma, cfg.corr_length)
    Phi_local, lambda_local = top_eigenpairs(C_local, Next)
    sqrt_lam = np.sqrt(lambda_local)
    cond_local_idx = select_conditioning_points(
        local_pts, sub.core_local_idx, sub.buffer_local_idx, Mb
    )
    Phi_ext = Phi_local[:, :Next]
    A = build_A(Phi_ext, sqrt_lam, cond_local_idx)

    theta_local_accepted = rng.standard_normal(Next, dtype=np.float64)
    expected_norm = expected_gaussian_norm(Next)

    for iteration in range(1, n_iter + 1):
        accepted_vec = G_accepted.ravel(order="F")
        c_data = build_c(accepted_vec, sub.local_global_idx, cond_local_idx)
        c_used = np.zeros_like(c_data) if rhs_mode == "zero" else c_data
        theta_p, Z, cond_A, cond_B = _solve_local_conditioning(
            A, c_used, Next, theta_p_method, conditioning_mode, rhs_mode
        )

        theta_proposed = _proposal_from_current(
            theta_local_accepted, beta_value, proposal, rng
        )
        theta_local_candidate, theta_n_candidate, hidden_null_norm = (
            _condition_candidate(
                theta_proposed,
                A,
                c_used,
                theta_p,
                Z,
                conditioning_mode,
                rhs_mode,
                rho,
            )
        )
        proposal_correction = _null_pcn_proposal_correction(
            theta_current=theta_local_accepted,
            theta_candidate=theta_local_candidate,
            Z=Z,
            proposal=proposal,
            conditioning_mode=conditioning_mode,
        )

        G_local_candidate = field_from_theta(
            Phi_ext, lambda_local, theta_local_candidate
        )
        candidate_vec = accepted_vec.copy()
        candidate_vec[sub.core_global_idx] = G_local_candidate[sub.core_local_idx]
        G_candidate = reshape_field(candidate_vec, cfg.ny, cfg.nx)
        k_candidate = permeability_from_log_field(G_candidate)
        pressure_candidate = fwd.solve(k_candidate)
        log_like_candidate = _pressure_log_likelihood(
            pressure_candidate, truth.y_obs, truth.sensor_idx, cfg.sigma_obs
        )

        log_alpha = _acceptance_log_ratio(
            log_like_candidate=log_like_candidate,
            log_like_current=log_like_accepted,
            acceptance=acceptance,
            G_candidate_vec=candidate_vec,
            G_current_vec=accepted_vec,
            global_kle=global_kle,
            proposal_correction=proposal_correction,
            prior_mode=prior_mode_value,
            theta_current=theta_local_accepted,
            theta_candidate=theta_local_candidate,
            Z=Z,
        )
        accept_prob = _acceptance_probability(log_alpha)
        accepted = bool(np.log(rng.uniform()) < min(0.0, log_alpha))

        if accepted:
            G_accepted = G_candidate.copy()
            k_accepted = k_candidate.copy()
            pressure_accepted = pressure_candidate.copy()
            log_like_accepted = log_like_candidate
            theta_local_accepted = theta_local_candidate.copy()

        c_after_data = build_c(
            G_accepted.ravel(order="F"), sub.local_global_idx, cond_local_idx
        )
        c_after = np.zeros_like(c_after_data) if rhs_mode == "zero" else c_after_data
        yield ConditionedSamplerState(
            iteration=iteration,
            G_accepted=G_accepted.copy(),
            k_accepted=k_accepted.copy(),
            pressure_accepted=pressure_accepted.copy(),
            log_likelihood_accepted=float(log_like_accepted),
            G_candidate=G_candidate.copy(),
            k_candidate=k_candidate.copy(),
            pressure_candidate=pressure_candidate.copy(),
            log_likelihood_candidate=float(log_like_candidate),
            accepted=accepted,
            acceptance_probability=accept_prob,
            theta_local_accepted=theta_local_accepted.copy(),
            theta_local_candidate=theta_local_candidate.copy(),
            theta_proposed=theta_proposed.copy(),
            theta_p=theta_p.copy(),
            theta_n_candidate=theta_n_candidate.copy(),
            theta_norm_accepted=theta_norm(theta_local_accepted),
            theta_norm_candidate=theta_norm(theta_local_candidate),
            theta_p_norm=theta_norm(theta_p),
            theta_n_candidate_norm=theta_norm(theta_n_candidate),
            null_space_norm_candidate=theta_norm(Z.T @ theta_local_candidate),
            null_space_norm_accepted=theta_norm(Z.T @ theta_local_accepted),
            hidden_null_norm=hidden_null_norm,
            expected_norm=expected_norm,
            constraint_residual_candidate=constraint_residual(
                A, theta_local_candidate, c_used
            ),
            constraint_residual_accepted=constraint_residual(
                A, theta_local_accepted, c_after
            ),
            interface_jump_accepted=interface_jump(G_accepted, sub),
            interface_jump_candidate=interface_jump(G_candidate, sub),
            relative_k_error_accepted=relative_error(k_accepted, truth.k_true),
            relative_k_error_candidate=relative_error(k_candidate, truth.k_true),
            cond_A=float(cond_A),
            cond_B=None if cond_B is None else float(cond_B),
        )


def red_black_conditioned_sampler(
    cfg: Config,
    n_sweeps: int,
    Mb: int,
    theta_p_method: str,
    rng: np.random.Generator,
    beta: float | None = None,
    proposal: str = "pcn",
    rhs_mode: str = "data",
    conditioning_mode: str = "hard",
    rho: float | None = None,
    acceptance: str = "likelihood_only",
    prior_mode: str | None = None,
) -> Iterator[RedBlackSamplerState]:
    """Run deterministic sequential red-black sweeps over all coarse subdomains.

    Coarse coordinates in yielded states are zero-based. Each color pass builds
    all conditioning right-hand sides from one frozen global field snapshot,
    then applies same-color local updates sequentially in sorted row/column
    order. Pass `acceptance="posterior"` for the M8 global-prior baseline.
    """

    if n_sweeps < 1:
        raise ValueError("n_sweeps must be at least 1.")
    _validate_conditioning_options(theta_p_method, conditioning_mode, rhs_mode, rho)
    _validate_acceptance(acceptance)
    prior_mode_value = cfg.prior_mode if prior_mode is None else prior_mode
    if acceptance == "posterior":
        _validate_prior_mode(prior_mode_value, conditioning_mode)
    beta_value = cfg.beta if beta is None else float(beta)

    truth = make_truth(cfg, rng)
    _, _, _, _, global_pts = cell_centered_grid(cfg.nx, cfg.ny)
    C_global = exp_covariance(global_pts, cfg.sigma, cfg.corr_length)
    Phi_global, lambda_global = top_eigenpairs(C_global, cfg.n_global_modes)
    global_kle = (Phi_global, lambda_global)

    theta_global_current = rng.standard_normal(cfg.n_global_modes, dtype=np.float64)
    G_current_vec = field_from_theta(Phi_global, lambda_global, theta_global_current)
    G_accepted = reshape_field(G_current_vec, cfg.ny, cfg.nx)
    k_accepted = permeability_from_log_field(G_accepted)
    fwd = ForwardModel(cfg)
    pressure_accepted = fwd.solve(k_accepted)
    log_like_accepted = _pressure_log_likelihood(
        pressure_accepted, truth.y_obs, truth.sensor_idx, cfg.sigma_obs
    )

    local_updates = _precompute_red_black_updates(cfg, Mb, global_pts)
    local_by_color: dict[int, list[_LocalUpdateData]] = {
        0: [data for data in local_updates if data.color == 0],
        1: [data for data in local_updates if data.color == 1],
    }
    Next = cfg.Nc + Mb
    theta_local_by_subdomain = {
        (data.row, data.col): rng.standard_normal(Next, dtype=np.float64)
        for data in local_updates
    }
    expected_norm = expected_gaussian_norm(Next)

    for sweep in range(1, n_sweeps + 1):
        for color in (0, 1):
            frozen_for_color = G_accepted.copy()
            frozen_vec = frozen_for_color.ravel(order="F")

            for data in local_by_color[color]:
                key = (data.row, data.col)
                c_data = build_c(
                    frozen_vec, data.sub.local_global_idx, data.cond_local_idx
                )
                c_used = np.zeros_like(c_data) if rhs_mode == "zero" else c_data
                theta_p, Z, cond_A, cond_B = _solve_local_conditioning(
                    data.A,
                    c_used,
                    Next,
                    theta_p_method,
                    conditioning_mode,
                    rhs_mode,
                )

                theta_local_current = theta_local_by_subdomain[key]
                theta_proposed = _proposal_from_current(
                    theta_local_current, beta_value, proposal, rng
                )
                theta_local_candidate, _, hidden_null_norm = _condition_candidate(
                    theta_proposed,
                    data.A,
                    c_used,
                    theta_p,
                    Z,
                    conditioning_mode,
                    rhs_mode,
                    rho,
                )
                proposal_correction = _null_pcn_proposal_correction(
                    theta_current=theta_local_current,
                    theta_candidate=theta_local_candidate,
                    Z=Z,
                    proposal=proposal,
                    conditioning_mode=conditioning_mode,
                )

                G_local_candidate = field_from_theta(
                    data.Phi_ext, data.lambda_local, theta_local_candidate
                )
                accepted_vec = G_accepted.ravel(order="F")
                candidate_vec = accepted_vec.copy()
                candidate_vec[data.sub.core_global_idx] = G_local_candidate[
                    data.sub.core_local_idx
                ]
                G_candidate = reshape_field(candidate_vec, cfg.ny, cfg.nx)
                k_candidate = permeability_from_log_field(G_candidate)
                pressure_candidate = fwd.solve(k_candidate)
                log_like_candidate = _pressure_log_likelihood(
                    pressure_candidate, truth.y_obs, truth.sensor_idx, cfg.sigma_obs
                )

                log_alpha = _acceptance_log_ratio(
                    log_like_candidate=log_like_candidate,
                    log_like_current=log_like_accepted,
                    acceptance=acceptance,
                    G_candidate_vec=candidate_vec,
                    G_current_vec=accepted_vec,
                    global_kle=global_kle,
                    proposal_correction=proposal_correction,
                    prior_mode=prior_mode_value,
                    theta_current=theta_local_current,
                    theta_candidate=theta_local_candidate,
                    Z=Z,
                )
                accept_prob = _acceptance_probability(log_alpha)
                accepted = bool(np.log(rng.uniform()) < min(0.0, log_alpha))

                if accepted:
                    G_accepted = G_candidate.copy()
                    k_accepted = k_candidate.copy()
                    pressure_accepted = pressure_candidate.copy()
                    log_like_accepted = log_like_candidate
                    theta_local_by_subdomain[key] = theta_local_candidate.copy()

                theta_local_accepted = theta_local_by_subdomain[key]
                yield RedBlackSamplerState(
                    sweep=sweep,
                    color=color,
                    subdomain_row=data.row,
                    subdomain_col=data.col,
                    accepted=accepted,
                    acceptance_probability=accept_prob,
                    theta_norm_candidate=theta_norm(theta_local_candidate),
                    theta_norm_accepted=theta_norm(theta_local_accepted),
                    expected_norm=expected_norm,
                    constraint_residual_candidate=constraint_residual(
                        data.A, theta_local_candidate, c_used
                    ),
                    interface_jump_candidate=interface_jump(G_candidate, data.sub),
                    interface_jump_accepted=interface_jump(G_accepted, data.sub),
                    relative_k_error_accepted=relative_error(k_accepted, truth.k_true),
                    log_likelihood_accepted=float(log_like_accepted),
                    G_accepted=G_accepted.copy(),
                    G_candidate=G_candidate.copy(),
                    cond_A=float(cond_A),
                    cond_B=None if cond_B is None else float(cond_B),
                    theta_p_norm=theta_norm(theta_p),
                    null_space_norm_candidate=theta_norm(Z.T @ theta_local_candidate),
                    null_space_norm_accepted=theta_norm(Z.T @ theta_local_accepted),
                    hidden_null_norm=hidden_null_norm,
                )


def _block_scan_order(cfg: Config, scan_order: str) -> list[tuple[int, int]]:
    """Return the coarse (row, col) visiting order for a block-Gibbs sweep.

    ``"systematic"`` is lexicographic row-major order. ``"red_black"`` visits
    all even-parity ``(row + col)`` cores, then all odd-parity cores, each block
    still updated sequentially from its current full conditional. Under the dense
    KLE+nugget precision ``Q`` same-color cores are NOT conditionally independent,
    so this is an alternative sequential ordering, not a parallel/checkerboard
    independence scheme. Invariance is order-independent either way.
    """

    if scan_order == "systematic":
        return iter_coarse_subdomains(cfg)
    if scan_order == "red_black":
        colored = red_black_order(cfg)
        return [*colored[0], *colored[1]]
    raise ValueError("scan_order must be 'systematic' or 'red_black'.")


def block_gibbs_sampler(
    cfg: Config,
    n_sweeps: int,
    rng: np.random.Generator,
    beta: float | None = None,
    block_nugget: float | None = None,
    scan_order: str = "systematic",
) -> Iterator[BlockGibbsSamplerState]:
    """Run precision block-Gibbs sweeps over the core blocks.

    This is the posterior-correct multiscale update derived in
    ``docs/conditioned_posterior_derivation.md``. Each core block (the core cells
    of one coarse subdomain; the cores tile the grid) is updated from its exact
    Gaussian full conditional under the proper full-rank prior ``N(0, C_tau)``,
    ``C_tau = Phi diag(lam) Phi.T + tau2 I``, using a pCN proposal around the
    conditional mean. Because that proposal is reversible w.r.t. the block's base
    Gaussian, the Metropolis acceptance is exactly likelihood-only, and a sweep
    over all cores targets ``pi(G | Y) ~ exp(-Phi_mis) N(0, C_tau)``.

    ``scan_order`` selects the within-sweep visiting order of the cores,
    ``"systematic"`` (default, lexicographic) or ``"red_black"`` (even then odd
    ``(row + col)`` parity). Both are sequential systematic-scan Gibbs and target
    the same posterior; red-black is only an ordering, not a parallel scheme
    (same-color cores are coupled under the dense precision).

    This is additive: it does not modify ``conditioned_sampler`` /
    ``red_black_conditioned_sampler`` or their ``likelihood_only`` /
    ``global_field`` / ``null_space`` acceptance modes. The rng draw order
    matches the conditioned samplers (truth prefix, then the initial global
    field) so the experiment-only matched-start replay adapter applies unchanged.
    """

    if n_sweeps < 1:
        raise ValueError("n_sweeps must be at least 1.")
    beta_value = cfg.beta if beta is None else float(beta)
    tau2 = cfg.block_nugget if block_nugget is None else float(block_nugget)
    if not np.isfinite(tau2) or tau2 <= 0.0:
        raise ValueError("block_nugget (tau^2) must be positive.")
    coarse_order = _block_scan_order(cfg, scan_order)

    truth = make_truth(cfg, rng)
    _, _, _, _, global_pts = cell_centered_grid(cfg.nx, cfg.ny)
    C_global = exp_covariance(global_pts, cfg.sigma, cfg.corr_length)
    Phi_global, lambda_global = top_eigenpairs(C_global, cfg.n_global_modes)
    sqrt_lambda_global = np.sqrt(lambda_global)

    precision = build_precision(Phi_global, lambda_global, tau2)
    blocks: list[tuple[int, int, BlockConditioner]] = []
    for row, col in coarse_order:
        sub = make_subdomain_at(cfg, row, col)
        conditioner = make_block_conditioner(precision, sub.core_global_idx)
        blocks.append((row, col, conditioner))

    theta_global_current = rng.standard_normal(cfg.n_global_modes, dtype=np.float64)
    G_vec = field_from_theta(Phi_global, lambda_global, theta_global_current)
    G_accepted = reshape_field(G_vec, cfg.ny, cfg.nx)
    k_accepted = permeability_from_log_field(G_accepted)
    fwd = ForwardModel(cfg)
    pressure_accepted = fwd.solve(k_accepted)
    log_like_accepted = _pressure_log_likelihood(
        pressure_accepted, truth.y_obs, truth.sensor_idx, cfg.sigma_obs
    )
    expected_norm = expected_gaussian_norm(cfg.n_global_modes)

    for sweep in range(1, n_sweeps + 1):
        for block_index, (row, col, conditioner) in enumerate(blocks):
            g_block_candidate = propose_block(
                precision, conditioner, G_vec, beta_value, rng
            )
            candidate_vec = G_vec.copy()
            candidate_vec[conditioner.global_idx] = g_block_candidate
            G_candidate = reshape_field(candidate_vec, cfg.ny, cfg.nx)
            k_candidate = permeability_from_log_field(G_candidate)
            pressure_candidate = fwd.solve(k_candidate)
            log_like_candidate = _pressure_log_likelihood(
                pressure_candidate, truth.y_obs, truth.sensor_idx, cfg.sigma_obs
            )

            # Acceptance is exactly likelihood-only: the conditional Gaussian and
            # the pCN proposal density cancel in the block MH ratio.
            log_alpha = float(log_like_candidate - log_like_accepted)
            accept_prob = _acceptance_probability(log_alpha)
            accepted = bool(np.log(rng.uniform()) < min(0.0, log_alpha))

            if accepted:
                G_vec = candidate_vec
                G_accepted = G_candidate
                k_accepted = k_candidate
                log_like_accepted = log_like_candidate

            theta_global_proj = (Phi_global.T @ G_vec) / sqrt_lambda_global
            yield BlockGibbsSamplerState(
                sweep=sweep,
                block_index=block_index,
                subdomain_row=row,
                subdomain_col=col,
                accepted=accepted,
                acceptance_probability=accept_prob,
                log_likelihood_accepted=float(log_like_accepted),
                log_likelihood_candidate=float(log_like_candidate),
                theta_norm_accepted=float(np.linalg.norm(theta_global_proj)),
                expected_norm=expected_norm,
                block_nugget=tau2,
                relative_k_error_accepted=relative_error(k_accepted, truth.k_true),
                relative_k_error_candidate=relative_error(k_candidate, truth.k_true),
                G_accepted=G_accepted.copy(),
                G_candidate=G_candidate.copy(),
            )
