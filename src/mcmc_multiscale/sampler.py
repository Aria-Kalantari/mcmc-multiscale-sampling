"""Headless single-subdomain conditioned sampler for M4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from mcmc_multiscale.conditioning.constraints import (
    build_A,
    build_c,
    select_conditioning_points,
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
from mcmc_multiscale.subdomain import make_subdomain


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
) -> Iterator[ConditionedSamplerState]:
    """Run the M4 single-subdomain repeated-conditioning harness.

    The M4 instability path uses `theta_p + Z @ (Z.T @ theta_proposed)`, not
    the shifted affine projection. This intentionally preserves any hidden
    null-space component of an arbitrary LU particular solution.
    """

    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if update_scheme != "single":
        raise NotImplementedError("M4 implements only update_scheme='single'.")
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
    beta_value = cfg.beta if beta is None else float(beta)

    truth = make_truth(cfg, rng)
    _, _, _, _, global_pts = cell_centered_grid(cfg.nx, cfg.ny)
    C_global = exp_covariance(global_pts, cfg.sigma, cfg.corr_length)
    Phi_global, lambda_global = top_eigenpairs(C_global, cfg.n_global_modes)

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
        Z = null_basis(A)

        if conditioning_mode == "hard" and rhs_mode == "zero":
            theta_p = np.zeros(Next, dtype=np.float64)
            _, _, _, cond_A, _ = _particular_solution(A, c_used, "svd")
            cond_B = None
        elif conditioning_mode == "hard":
            theta_p, Z, _, cond_A, cond_B = _particular_solution(
                A, c_used, theta_p_method
            )
        else:
            _, _, _, cond_A, _ = _particular_solution(A, c_used, "svd")
            cond_B = None
            theta_p = np.zeros(Next, dtype=np.float64)

        theta_proposed = _proposal_from_current(
            theta_local_accepted, beta_value, proposal, rng
        )
        if conditioning_mode == "soft":
            theta_local_candidate = soft_project(theta_proposed, A, c_used, float(rho))
            theta_n_candidate = theta_local_candidate.copy()
            hidden_null_norm = np.nan
        else:
            theta_n_candidate = project_null(Z, theta_proposed)
            theta_local_candidate = theta_p + theta_n_candidate
            hidden_null_norm = 0.0 if rhs_mode == "zero" else theta_norm(Z.T @ theta_p)

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

        log_alpha = log_like_candidate - log_like_accepted
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
