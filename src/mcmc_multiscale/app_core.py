"""Streamlit-free run and summary helpers for M4/M5 sampler outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mcmc_multiscale.config import Config
from mcmc_multiscale.sampler import ConditionedSamplerState, conditioned_sampler


@dataclass(frozen=True)
class MethodRunConfig:
    """Configuration for one headless conditioned-sampler run."""

    label: str
    theta_p_method: str
    conditioning_mode: str = "hard"
    rhs_mode: str = "data"
    rho: float | None = None


@dataclass(frozen=True)
class RunSummary:
    """Numerical summary of one conditioned-sampler run."""

    label: str
    n_iter: int
    Mb: int
    beta: float
    seed: int
    acceptance_rate: float
    accepted_count: int
    expected_norm: float
    max_candidate_theta_norm: float
    final_candidate_theta_norm: float
    max_accepted_theta_norm: float
    final_accepted_theta_norm: float
    candidate_norm_over_expected: float
    accepted_norm_over_expected: float
    mean_residual: float
    max_residual: float
    mean_interface_jump_candidate: float
    mean_interface_jump_accepted: float
    mean_relative_k_error_candidate: float | None
    final_relative_k_error_candidate: float | None
    mean_relative_k_error_accepted: float | None
    final_relative_k_error_accepted: float | None
    mean_hidden_null_norm: float | None
    max_hidden_null_norm: float | None
    mean_cond_A: float
    max_cond_A: float
    mean_cond_B: float | None
    max_cond_B: float | None
    rho: float | None = None


def _nan_array(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def _nan_mean_or_none(values: Sequence[float | None]) -> float | None:
    arr = _nan_array(values)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def _nan_max_or_none(values: Sequence[float | None]) -> float | None:
    arr = _nan_array(values)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmax(arr))


def _nan_final_or_none(values: Sequence[float | None]) -> float | None:
    arr = _nan_array(values)
    if arr.size == 0 or np.isnan(arr[-1]):
        return None
    return float(arr[-1])


def summarize_states(
    states: Sequence[ConditionedSamplerState],
    label: str,
    Mb: int,
    beta: float,
    seed: int,
    rho: float | None = None,
) -> RunSummary:
    """Summarize sampler states without any UI dependency."""

    if len(states) == 0:
        raise ValueError("states must contain at least one sampler state.")

    candidate_norms = np.asarray(
        [state.theta_norm_candidate for state in states], dtype=np.float64
    )
    accepted_norms = np.asarray(
        [state.theta_norm_accepted for state in states], dtype=np.float64
    )
    accepted = np.asarray([state.accepted for state in states], dtype=bool)
    residuals = np.asarray(
        [state.constraint_residual_candidate for state in states], dtype=np.float64
    )
    candidate_jumps = np.asarray(
        [state.interface_jump_candidate for state in states], dtype=np.float64
    )
    accepted_jumps = np.asarray(
        [state.interface_jump_accepted for state in states], dtype=np.float64
    )
    cond_A = np.asarray([state.cond_A for state in states], dtype=np.float64)
    expected_norm = float(states[-1].expected_norm)

    max_candidate = float(np.max(candidate_norms))
    max_accepted = float(np.max(accepted_norms))

    return RunSummary(
        label=label,
        n_iter=len(states),
        Mb=Mb,
        beta=beta,
        seed=seed,
        acceptance_rate=float(np.mean(accepted)),
        accepted_count=int(np.count_nonzero(accepted)),
        expected_norm=expected_norm,
        max_candidate_theta_norm=max_candidate,
        final_candidate_theta_norm=float(candidate_norms[-1]),
        max_accepted_theta_norm=max_accepted,
        final_accepted_theta_norm=float(accepted_norms[-1]),
        candidate_norm_over_expected=max_candidate / expected_norm,
        accepted_norm_over_expected=max_accepted / expected_norm,
        mean_residual=float(np.mean(residuals)),
        max_residual=float(np.max(residuals)),
        mean_interface_jump_candidate=float(np.mean(candidate_jumps)),
        mean_interface_jump_accepted=float(np.mean(accepted_jumps)),
        mean_relative_k_error_candidate=_nan_mean_or_none(
            [state.relative_k_error_candidate for state in states]
        ),
        final_relative_k_error_candidate=_nan_final_or_none(
            [state.relative_k_error_candidate for state in states]
        ),
        mean_relative_k_error_accepted=_nan_mean_or_none(
            [state.relative_k_error_accepted for state in states]
        ),
        final_relative_k_error_accepted=_nan_final_or_none(
            [state.relative_k_error_accepted for state in states]
        ),
        mean_hidden_null_norm=_nan_mean_or_none(
            [state.hidden_null_norm for state in states]
        ),
        max_hidden_null_norm=_nan_max_or_none(
            [state.hidden_null_norm for state in states]
        ),
        mean_cond_A=float(np.mean(cond_A)),
        max_cond_A=float(np.max(cond_A)),
        mean_cond_B=_nan_mean_or_none([state.cond_B for state in states]),
        max_cond_B=_nan_max_or_none([state.cond_B for state in states]),
        rho=rho,
    )


def run_method(
    cfg: Config,
    method: MethodRunConfig,
    n_iter: int,
    Mb: int,
    beta: float,
    seed: int,
    proposal: str = "pcn",
) -> tuple[list[ConditionedSamplerState], RunSummary]:
    """Run one configured sampler method and return states plus summary."""

    states = list(
        conditioned_sampler(
            cfg=cfg,
            n_iter=n_iter,
            Mb=Mb,
            theta_p_method=method.theta_p_method,
            rng=np.random.default_rng(seed),
            beta=beta,
            update_scheme="single",
            proposal=proposal,
            rhs_mode=method.rhs_mode,
            conditioning_mode=method.conditioning_mode,
            rho=method.rho,
        )
    )
    return states, summarize_states(
        states=states,
        label=method.label,
        Mb=Mb,
        beta=beta,
        seed=seed,
        rho=method.rho,
    )


def run_methods(
    cfg: Config,
    methods: Sequence[MethodRunConfig],
    n_iter: int,
    Mb: int,
    beta: float,
    seed: int,
    proposal: str = "pcn",
) -> dict[str, tuple[list[ConditionedSamplerState], RunSummary]]:
    """Run several methods with identical sampler settings."""

    return {
        method.label: run_method(
            cfg=cfg,
            method=method,
            n_iter=n_iter,
            Mb=Mb,
            beta=beta,
            seed=seed,
            proposal=proposal,
        )
        for method in methods
    }


def run_lu_svd_comparison(
    cfg: Config,
    n_iter: int,
    Mb: int,
    beta: float,
    seed: int,
    proposal: str = "pcn",
) -> dict[str, tuple[list[ConditionedSamplerState], RunSummary]]:
    """Run the M4 LU-vs-SVD comparison through the shared helper."""

    return run_methods(
        cfg=cfg,
        methods=[
            MethodRunConfig("hard_data_lu", "lu"),
            MethodRunConfig("hard_data_svd", "svd"),
        ],
        n_iter=n_iter,
        Mb=Mb,
        beta=beta,
        seed=seed,
        proposal=proposal,
    )


def default_m5_methods(rho_values: Sequence[float]) -> list[MethodRunConfig]:
    """Return the standard M5 stability-fix comparison methods."""

    methods = [
        MethodRunConfig("hard_data_lu", "lu"),
        MethodRunConfig("hard_data_svd", "svd"),
        MethodRunConfig("hard_data_lu_stabilized", "lu_stabilized"),
        MethodRunConfig("hard_zero_svd", "svd", rhs_mode="zero"),
    ]
    methods.extend(
        MethodRunConfig(
            label=f"soft_data_rho_{float(rho):.0e}",
            theta_p_method="soft",
            conditioning_mode="soft",
            rho=float(rho),
        )
        for rho in rho_values
    )
    return methods
