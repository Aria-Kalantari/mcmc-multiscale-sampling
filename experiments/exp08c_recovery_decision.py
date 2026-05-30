"""Decisive red-black versus global-pCN recovery comparison."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.exp08_convergence_diagnostics import (  # noqa: E402
    ScalarSummary,
    _TruthReplayGenerator,
    _global_kle,
    _project_coefficients,
    _scalar_summary,
    _truth_replay_prefix,
)
from mcmc_multiscale.bayes import log_posterior, log_prior  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.diagnostics import (  # noqa: E402
    credible_interval_coverage,
    posterior_summary,
    relative_error,
    sampling_efficiency,
)
from mcmc_multiscale.field import (  # noqa: E402
    field_from_theta,
    permeability_from_log_field,
    reshape_field,
)
from mcmc_multiscale.forward import ForwardModel  # noqa: E402
from mcmc_multiscale.mcmc import MCMCState, metropolis_hastings  # noqa: E402
from mcmc_multiscale.observations import TruthData  # noqa: E402
from mcmc_multiscale.proposals import make_pcn_proposal  # noqa: E402
from mcmc_multiscale.sampler import (  # noqa: E402
    BlockGibbsSamplerState,
    RedBlackSamplerState,
    block_gibbs_sampler,
    red_black_conditioned_sampler,
)


@dataclass(frozen=True)
class DecisionProfile:
    label: str
    red_black_sweeps: int
    global_iter: int
    wall_cap_seconds: float
    sample_stride: int
    trajectory_points: int
    stabilization_window: int
    stabilization_tolerance: float
    recovery_gap_tolerance: float
    misfit_trend_fraction: float


@dataclass(frozen=True)
class DecisionChain:
    fields: np.ndarray
    misfit: np.ndarray
    coefficients: np.ndarray
    theta_norm: np.ndarray
    accepted_state_theta_norm: np.ndarray
    null_space_norm: np.ndarray
    theta_p_norm: np.ndarray
    sample_updates: np.ndarray
    sample_solves: np.ndarray
    acceptance_rate: float
    updates: int
    forward_solves: int
    wall_seconds: float
    stopped_by: str


@dataclass(frozen=True)
class TrajectoryPoint:
    mean_updates_per_chain: float
    total_forward_solves: int
    relative_k_error: float
    mean_misfit: float
    misfit_over_noise_floor: float
    max_r_hat: float
    mean_accepted_state_theta_norm: float = np.nan
    mean_null_space_norm: float = np.nan
    mean_theta_p_norm: float = np.nan


@dataclass(frozen=True)
class DecisionSummary:
    label: str
    relative_k_error: float
    trajectory_shape: str
    relative_k_stable: bool
    relative_k_window_change: float
    max_relative_k_checkpoint_change: float
    misfit_trajectory_shape: str
    final_accepted_state_theta_norm: float
    max_accepted_state_theta_norm: float
    final_null_space_norm: float
    max_null_space_norm: float
    mean_theta_p_norm: float
    max_theta_p_norm: float
    tail_mean_misfit: float
    min_misfit: float
    misfit_over_noise_floor: float
    coverage: float
    acceptance_rate: float
    endpoint_max_r_hat: float
    best_max_r_hat: float
    conservative_total_ess: float
    total_forward_solves: int
    wall_seconds: float
    ess_per_1000_solves: float
    ess_per_second: float
    stopped_by: str
    updates_per_chain: tuple[int, ...]
    scalar_summaries: tuple[ScalarSummary, ...]
    trajectory: tuple[TrajectoryPoint, ...]


def _resolved_profile(args: argparse.Namespace) -> DecisionProfile:
    if args.decide and args.resolve:
        raise ValueError("choose at most one of --decide and --resolve.")
    if args.resolve:
        defaults = DecisionProfile(
            "resolving", 2_000, 40_000, 7_200.0, 32, 12, 4, 0.005, 0.02, 0.05
        )
    elif args.decide:
        defaults = DecisionProfile(
            "decision", 500, 10_000, 600.0, 8, 8, 4, 0.005, 0.02, 0.05
        )
    else:
        defaults = DecisionProfile(
            "responsive", 2, 64, 60.0, 1, 4, 4, 0.005, 0.02, 0.05
        )
    profile = DecisionProfile(
        label=defaults.label,
        red_black_sweeps=(
            defaults.red_black_sweeps
            if args.red_black_sweeps is None
            else args.red_black_sweeps
        ),
        global_iter=(
            defaults.global_iter if args.global_iter is None else args.global_iter
        ),
        wall_cap_seconds=(
            defaults.wall_cap_seconds
            if args.wall_cap_seconds is None
            else args.wall_cap_seconds
        ),
        sample_stride=(
            defaults.sample_stride if args.sample_stride is None else args.sample_stride
        ),
        trajectory_points=(
            defaults.trajectory_points
            if args.trajectory_points is None
            else args.trajectory_points
        ),
        stabilization_window=(
            defaults.stabilization_window
            if args.stabilization_window is None
            else args.stabilization_window
        ),
        stabilization_tolerance=(
            defaults.stabilization_tolerance
            if args.stabilization_tolerance is None
            else args.stabilization_tolerance
        ),
        recovery_gap_tolerance=(
            defaults.recovery_gap_tolerance
            if args.recovery_gap_tolerance is None
            else args.recovery_gap_tolerance
        ),
        misfit_trend_fraction=(
            defaults.misfit_trend_fraction
            if args.misfit_trend_fraction is None
            else args.misfit_trend_fraction
        ),
    )
    if profile.red_black_sweeps < 1 or profile.global_iter < 1:
        raise ValueError("chain lengths must be positive.")
    if profile.wall_cap_seconds <= 0.0:
        raise ValueError("wall-cap-seconds must be positive.")
    if profile.sample_stride < 1:
        raise ValueError("sample-stride must be positive.")
    if profile.trajectory_points < 2:
        raise ValueError("trajectory-points must be at least 2.")
    if profile.stabilization_window < 2:
        raise ValueError("stabilization-window must be at least 2.")
    if profile.stabilization_tolerance <= 0.0:
        raise ValueError("stabilization-tolerance must be positive.")
    if profile.recovery_gap_tolerance < 0.0:
        raise ValueError("recovery-gap-tolerance must be non-negative.")
    if profile.misfit_trend_fraction <= 0.0:
        raise ValueError("misfit-trend-fraction must be positive.")
    return profile


def _consume_states(
    states: Iterator[RedBlackSamplerState] | Iterator[MCMCState],
    max_updates: int,
    burn_fraction: float,
    sample_stride: int,
    solve_offset: int,
    wall_cap_seconds: float,
    field_and_misfit,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[float],
    list[float],
    list[float],
    list[int],
    list[int],
    list[bool],
    int,
    float,
    str,
]:
    burn = int(max_updates * burn_fraction)
    fields: list[np.ndarray] = []
    misfit: list[float] = []
    accepted_state_theta_norm: list[float] = []
    null_space_norm: list[float] = []
    theta_p_norm: list[float] = []
    sample_updates: list[int] = []
    sample_solves: list[int] = []
    accepted: list[bool] = []
    last_sample: tuple[np.ndarray, float, float, float, float, int] | None = None
    started = perf_counter()

    for idx, state in enumerate(states, start=1):
        accepted.append(state.accepted)
        if idx > burn:
            (
                field,
                state_misfit,
                state_theta_norm,
                state_null_space_norm,
                state_theta_p_norm,
            ) = field_and_misfit(state)
            sample = (
                field,
                state_misfit,
                state_theta_norm,
                state_null_space_norm,
                state_theta_p_norm,
                idx + solve_offset,
            )
            last_sample = sample
            if (idx - burn - 1) % sample_stride == 0:
                fields.append(field)
                misfit.append(state_misfit)
                accepted_state_theta_norm.append(state_theta_norm)
                null_space_norm.append(state_null_space_norm)
                theta_p_norm.append(state_theta_p_norm)
                sample_updates.append(idx)
                sample_solves.append(idx + solve_offset)
        if perf_counter() - started >= wall_cap_seconds:
            stopped_by = "wall_cap"
            break
    else:
        stopped_by = "budget"

    wall_seconds = perf_counter() - started
    updates = len(accepted)
    if last_sample is None:
        raise RuntimeError(
            "wall cap stopped a chain before burn-in completed; increase the cap "
            "or reduce the requested budget."
        )
    if not sample_solves or sample_solves[-1] != last_sample[5]:
        fields.append(last_sample[0])
        misfit.append(last_sample[1])
        accepted_state_theta_norm.append(last_sample[2])
        null_space_norm.append(last_sample[3])
        theta_p_norm.append(last_sample[4])
        sample_updates.append(last_sample[5] - solve_offset)
        sample_solves.append(last_sample[5])
    return (
        fields,
        misfit,
        accepted_state_theta_norm,
        null_space_norm,
        theta_p_norm,
        sample_updates,
        sample_solves,
        accepted,
        updates,
        wall_seconds,
        stopped_by,
    )


def _finish_chain(
    fields: list[np.ndarray],
    misfit: list[float],
    accepted_state_theta_norm: list[float],
    null_space_norm: list[float],
    theta_p_norm: list[float],
    sample_updates: list[int],
    sample_solves: list[int],
    accepted: list[bool],
    updates: int,
    wall_seconds: float,
    stopped_by: str,
    solve_offset: int,
    Phi: np.ndarray,
    lam: np.ndarray,
) -> DecisionChain:
    fields_arr = np.stack(fields)
    coefficients = _project_coefficients(fields_arr, Phi, lam)
    return DecisionChain(
        fields=fields_arr,
        misfit=np.asarray(misfit, dtype=np.float64),
        coefficients=coefficients,
        theta_norm=np.linalg.norm(coefficients, axis=1),
        accepted_state_theta_norm=np.asarray(
            accepted_state_theta_norm, dtype=np.float64
        ),
        null_space_norm=np.asarray(null_space_norm, dtype=np.float64),
        theta_p_norm=np.asarray(theta_p_norm, dtype=np.float64),
        sample_updates=np.asarray(sample_updates, dtype=np.int64),
        sample_solves=np.asarray(sample_solves, dtype=np.int64),
        acceptance_rate=float(np.mean(accepted)),
        updates=updates,
        forward_solves=updates + solve_offset,
        wall_seconds=wall_seconds,
        stopped_by=stopped_by,
    )


def _red_black_chain(
    cfg: Config,
    truth_draw: np.ndarray,
    noise_draw: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    chain_seed: int,
    n_sweeps: int,
    Mb: int,
    beta: float,
    burn_fraction: float,
    sample_stride: int,
    wall_cap_seconds: float,
    prior_mode: str,
) -> DecisionChain:
    n_subdomains = cfg.n_coarse_x * cfg.n_coarse_y
    rng = _TruthReplayGenerator(
        theta_true_draw=truth_draw,
        noise_draw=noise_draw,
        chain_seed=chain_seed,
        n_initial_draws=1 + n_subdomains,
        initial_scale=1.0,
    )
    states = red_black_conditioned_sampler(
        cfg=cfg,
        n_sweeps=n_sweeps,
        Mb=Mb,
        theta_p_method="svd",
        rng=rng,  # type: ignore[arg-type]
        beta=beta,
        acceptance="posterior",
        prior_mode=prior_mode,
    )

    def field_and_misfit(
        state: RedBlackSamplerState,
    ) -> tuple[np.ndarray, float, float, float, float]:
        return (
            state.G_accepted.copy(),
            -state.log_likelihood_accepted,
            state.theta_norm_accepted,
            state.null_space_norm_accepted,
            state.theta_p_norm,
        )

    consumed = _consume_states(
        states=states,
        max_updates=n_sweeps * n_subdomains,
        burn_fraction=burn_fraction,
        sample_stride=sample_stride,
        solve_offset=2,
        wall_cap_seconds=wall_cap_seconds,
        field_and_misfit=field_and_misfit,
    )
    return _finish_chain(*consumed, solve_offset=2, Phi=Phi, lam=lam)


def _block_gibbs_chain(
    cfg: Config,
    truth_draw: np.ndarray,
    noise_draw: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    chain_seed: int,
    n_sweeps: int,
    beta: float,
    burn_fraction: float,
    sample_stride: int,
    wall_cap_seconds: float,
    block_nugget: float,
) -> DecisionChain:
    """Run a precision block-Gibbs chain on a matched unit-scale prior start.

    The block-Gibbs sampler draws the truth prefix and then one initial global
    field, identical to the red-black setup, so the replay adapter delivers a
    matched start. It carries no local null coordinates, so the null-space and
    theta_p columns are reported as NaN like the global-pCN chain.
    """

    n_subdomains = cfg.n_coarse_x * cfg.n_coarse_y
    rng = _TruthReplayGenerator(
        theta_true_draw=truth_draw,
        noise_draw=noise_draw,
        chain_seed=chain_seed,
        n_initial_draws=1,
        initial_scale=1.0,
    )
    states = block_gibbs_sampler(
        cfg=cfg,
        n_sweeps=n_sweeps,
        rng=rng,  # type: ignore[arg-type]
        beta=beta,
        block_nugget=block_nugget,
    )

    def field_and_misfit(
        state: BlockGibbsSamplerState,
    ) -> tuple[np.ndarray, float, float, float, float]:
        return (
            state.G_accepted.copy(),
            -state.log_likelihood_accepted,
            state.theta_norm_accepted,
            np.nan,
            np.nan,
        )

    consumed = _consume_states(
        states=states,
        max_updates=n_sweeps * n_subdomains,
        burn_fraction=burn_fraction,
        sample_stride=sample_stride,
        solve_offset=1,
        wall_cap_seconds=wall_cap_seconds,
        field_and_misfit=field_and_misfit,
    )
    return _finish_chain(*consumed, solve_offset=1, Phi=Phi, lam=lam)


def _global_pcn_chain(
    cfg: Config,
    truth: TruthData,
    Phi: np.ndarray,
    lam: np.ndarray,
    chain_seed: int,
    n_iter: int,
    beta: float,
    burn_fraction: float,
    sample_stride: int,
    wall_cap_seconds: float,
) -> DecisionChain:
    rng = np.random.default_rng(chain_seed)
    fwd = ForwardModel(cfg)

    def log_density(theta: np.ndarray) -> float:
        return log_posterior(
            theta,
            Phi,
            lam,
            fwd,
            truth.y_obs,
            truth.sensor_idx,
            cfg.sigma_obs,
            cfg.ny,
            cfg.nx,
        )

    states = metropolis_hastings(
        log_density_fn=log_density,
        proposal_fn=make_pcn_proposal(beta),
        theta0=rng.standard_normal(cfg.n_global_modes, dtype=np.float64),
        n_iter=n_iter,
        rng=rng,
        log_prior_fn=log_prior,
    )

    def field_and_misfit(
        state: MCMCState,
    ) -> tuple[np.ndarray, float, float, float, float]:
        field = reshape_field(field_from_theta(Phi, lam, state.theta), cfg.ny, cfg.nx)
        return (
            field,
            -(state.log_density - log_prior(state.theta)),
            float(np.linalg.norm(state.theta)),
            np.nan,
            np.nan,
        )

    consumed = _consume_states(
        states=states,
        max_updates=n_iter,
        burn_fraction=burn_fraction,
        sample_stride=sample_stride,
        solve_offset=1,
        wall_cap_seconds=wall_cap_seconds,
        field_and_misfit=field_and_misfit,
    )
    return _finish_chain(*consumed, solve_offset=1, Phi=Phi, lam=lam)


def _trim_chains(chains: Sequence[DecisionChain]) -> tuple[DecisionChain, ...]:
    min_samples = min(chain.fields.shape[0] for chain in chains)
    if min_samples < 2:
        raise RuntimeError("each chain must retain at least two diagnostic samples.")
    return tuple(
        DecisionChain(
            fields=chain.fields[:min_samples],
            misfit=chain.misfit[:min_samples],
            coefficients=chain.coefficients[:min_samples],
            theta_norm=chain.theta_norm[:min_samples],
            accepted_state_theta_norm=chain.accepted_state_theta_norm[:min_samples],
            null_space_norm=chain.null_space_norm[:min_samples],
            theta_p_norm=chain.theta_p_norm[:min_samples],
            sample_updates=chain.sample_updates[:min_samples],
            sample_solves=chain.sample_solves[:min_samples],
            acceptance_rate=chain.acceptance_rate,
            updates=chain.updates,
            forward_solves=chain.forward_solves,
            wall_seconds=chain.wall_seconds,
            stopped_by=chain.stopped_by,
        )
        for chain in chains
    )


def _scalar_summaries(
    chains: Sequence[DecisionChain], prefix: int | None = None
) -> tuple[ScalarSummary, ...]:
    end = chains[0].misfit.size if prefix is None else prefix
    values = [
        _scalar_summary("misfit", np.stack([chain.misfit[:end] for chain in chains])),
        _scalar_summary(
            "theta_norm", np.stack([chain.theta_norm[:end] for chain in chains])
        ),
    ]
    values.extend(
        _scalar_summary(
            f"theta_global_{idx}",
            np.stack([chain.coefficients[:end, idx] for chain in chains]),
        )
        for idx in range(min(3, chains[0].coefficients.shape[1]))
    )
    return tuple(values)


def _nan_mean(values: np.ndarray) -> float:
    return float(np.nan) if np.all(np.isnan(values)) else float(np.nanmean(values))


def _nan_max(values: np.ndarray) -> float:
    return float(np.nan) if np.all(np.isnan(values)) else float(np.nanmax(values))


def _trajectory(
    chains: Sequence[DecisionChain],
    truth: TruthData,
    noise_floor: float,
    n_points: int,
) -> tuple[TrajectoryPoint, ...]:
    n_samples = chains[0].fields.shape[0]
    prefixes = sorted(
        {
            max(2, int(round(fraction * n_samples)))
            for fraction in np.linspace(1.0 / n_points, 1.0, n_points)
        }
    )
    points: list[TrajectoryPoint] = []
    for prefix in prefixes:
        fields = np.concatenate([chain.fields[:prefix] for chain in chains], axis=0)
        G_mean, _ = posterior_summary(fields)
        scalar_summaries = _scalar_summaries(chains, prefix=prefix)
        points.append(
            TrajectoryPoint(
                mean_updates_per_chain=float(
                    np.mean([chain.sample_updates[prefix - 1] for chain in chains])
                ),
                total_forward_solves=int(
                    np.sum([chain.sample_solves[prefix - 1] for chain in chains])
                ),
                relative_k_error=relative_error(
                    permeability_from_log_field(G_mean), truth.k_true
                ),
                mean_misfit=float(
                    np.mean([chain.misfit[prefix - 1] for chain in chains])
                ),
                misfit_over_noise_floor=float(
                    np.mean([chain.misfit[prefix - 1] for chain in chains])
                    / noise_floor
                ),
                max_r_hat=max(summary.r_hat for summary in scalar_summaries),
                mean_accepted_state_theta_norm=_nan_mean(
                    np.asarray(
                        [
                            chain.accepted_state_theta_norm[prefix - 1]
                            for chain in chains
                        ],
                        dtype=np.float64,
                    )
                ),
                mean_null_space_norm=_nan_mean(
                    np.asarray(
                        [chain.null_space_norm[prefix - 1] for chain in chains],
                        dtype=np.float64,
                    )
                ),
                mean_theta_p_norm=_nan_mean(
                    np.asarray(
                        [chain.theta_p_norm[prefix - 1] for chain in chains],
                        dtype=np.float64,
                    )
                ),
            )
        )
    return tuple(points)


def _trajectory_shape(points: Sequence[TrajectoryPoint]) -> str:
    if len(points) < 2:
        return "insufficient"
    final_window = points[-min(4, len(points)) :]
    change = final_window[0].relative_k_error - final_window[-1].relative_k_error
    if change > 0.01:
        return "descending"
    if change < -0.01:
        return "rising"
    return "plateau"


def _relative_k_stabilization(
    points: Sequence[TrajectoryPoint], window: int, tolerance: float
) -> tuple[bool, float, float]:
    """Report whether final relative-k checkpoint steps stay within tolerance."""
    if len(points) < 2:
        return False, float("nan"), float("nan")
    final_window = points[-min(window, len(points)) :]
    values = np.asarray(
        [point.relative_k_error for point in final_window], dtype=np.float64
    )
    changes = np.abs(np.diff(values))
    return (
        bool(np.all(changes < tolerance)),
        float(values[-1] - values[0]),
        float(np.max(changes)),
    )


def _misfit_trajectory_shape(
    points: Sequence[TrajectoryPoint],
    noise_floor: float,
    window: int,
    trend_fraction: float,
) -> str:
    """Classify the final data-misfit checkpoint direction against its floor."""
    if len(points) < 2:
        return "insufficient"
    final_window = points[-min(window, len(points)) :]
    change = final_window[0].mean_misfit - final_window[-1].mean_misfit
    tolerance = trend_fraction * noise_floor
    if change > tolerance:
        return "descending"
    if change < -tolerance:
        return "rising"
    return "plateau"


def _summarize(
    label: str,
    chains: Sequence[DecisionChain],
    truth: TruthData,
    noise_floor: float,
    level: float,
    trajectory_points: int,
    stabilization_window: int,
    stabilization_tolerance: float,
    misfit_trend_fraction: float,
) -> DecisionSummary:
    trimmed = _trim_chains(chains)
    fields = np.concatenate([chain.fields for chain in trimmed], axis=0)
    G_mean, _ = posterior_summary(fields)
    scalar_summaries = _scalar_summaries(trimmed)
    conservative_total_ess = min(summary.total_ess for summary in scalar_summaries)
    total_forward_solves = sum(chain.forward_solves for chain in trimmed)
    wall_seconds = sum(chain.wall_seconds for chain in trimmed)
    ess_per_1000_solves, ess_per_second = sampling_efficiency(
        conservative_total_ess, total_forward_solves, wall_seconds
    )
    trajectory = _trajectory(
        trimmed, truth, noise_floor=noise_floor, n_points=trajectory_points
    )
    (
        relative_k_stable,
        relative_k_window_change,
        max_relative_k_checkpoint_change,
    ) = _relative_k_stabilization(
        trajectory, window=stabilization_window, tolerance=stabilization_tolerance
    )
    tail = max(1, trimmed[0].misfit.size // 10)
    tail_misfit = np.concatenate([chain.misfit[-tail:] for chain in trimmed])
    all_misfit = np.concatenate([chain.misfit for chain in trimmed])
    accepted_state_theta_norm = np.concatenate(
        [chain.accepted_state_theta_norm for chain in trimmed]
    )
    null_space_norm = np.concatenate([chain.null_space_norm for chain in trimmed])
    theta_p_norm = np.concatenate([chain.theta_p_norm for chain in trimmed])
    return DecisionSummary(
        label=label,
        relative_k_error=relative_error(
            permeability_from_log_field(G_mean), truth.k_true
        ),
        trajectory_shape=_trajectory_shape(trajectory),
        relative_k_stable=relative_k_stable,
        relative_k_window_change=relative_k_window_change,
        max_relative_k_checkpoint_change=max_relative_k_checkpoint_change,
        misfit_trajectory_shape=_misfit_trajectory_shape(
            trajectory,
            noise_floor=noise_floor,
            window=stabilization_window,
            trend_fraction=misfit_trend_fraction,
        ),
        final_accepted_state_theta_norm=_nan_mean(
            np.asarray(
                [chain.accepted_state_theta_norm[-1] for chain in trimmed],
                dtype=np.float64,
            )
        ),
        max_accepted_state_theta_norm=_nan_max(accepted_state_theta_norm),
        final_null_space_norm=_nan_mean(
            np.asarray(
                [chain.null_space_norm[-1] for chain in trimmed], dtype=np.float64
            )
        ),
        max_null_space_norm=_nan_max(null_space_norm),
        mean_theta_p_norm=_nan_mean(theta_p_norm),
        max_theta_p_norm=_nan_max(theta_p_norm),
        tail_mean_misfit=float(np.mean(tail_misfit)),
        min_misfit=float(np.min(all_misfit)),
        misfit_over_noise_floor=float(np.mean(tail_misfit) / noise_floor),
        coverage=credible_interval_coverage(fields, truth.G_true, level=level),
        acceptance_rate=float(np.mean([chain.acceptance_rate for chain in trimmed])),
        endpoint_max_r_hat=max(summary.r_hat for summary in scalar_summaries),
        best_max_r_hat=min(point.max_r_hat for point in trajectory),
        conservative_total_ess=conservative_total_ess,
        total_forward_solves=total_forward_solves,
        wall_seconds=wall_seconds,
        ess_per_1000_solves=ess_per_1000_solves,
        ess_per_second=ess_per_second,
        stopped_by=(
            "wall_cap"
            if any(chain.stopped_by == "wall_cap" for chain in trimmed)
            else "budget"
        ),
        updates_per_chain=tuple(chain.updates for chain in trimmed),
        scalar_summaries=scalar_summaries,
        trajectory=trajectory,
    )


def _verdict(
    red_black: DecisionSummary,
    global_pcn: DecisionSummary,
    noise_floor: float,
    level: float | None = None,
    recovery_gap_tolerance: float = 0.02,
) -> str:
    del level
    both_stable = red_black.relative_k_stable and global_pcn.relative_k_stable
    red_black_below = (
        red_black.relative_k_error
        < global_pcn.relative_k_error - recovery_gap_tolerance
    )
    red_black_at_or_above = (
        red_black.relative_k_error
        >= global_pcn.relative_k_error - recovery_gap_tolerance
    )
    red_black_misfit_progressing = (
        red_black.misfit_trajectory_shape == "descending"
        or red_black.tail_mean_misfit <= 1.5 * noise_floor
    )

    if red_black.trajectory_shape == "rising" and global_pcn.relative_k_stable:
        return (
            "NULL-SPACE PRIOR INSUFFICIENT: the diagnostic bounds local "
            "null-coordinate norms but red-black relative-k still reverses "
            "while global pCN plateaus. Record this for the conditioned-prior "
            "derivation; do not force a positive recovery conclusion."
        )
    if both_stable and red_black_below and red_black_misfit_progressing:
        return (
            "CONDITIONING RECOVERS BETTER: red-black relative-k flattened below "
            "the global-pCN plateau while its data misfit is still descending "
            "toward the noise floor. Route section 3.8(b) is not needed; "
            "proceed to M10/M11."
        )
    if both_stable and red_black_at_or_above:
        return (
            "RECONSIDER CONDITIONING: red-black relative-k flattened at or "
            "above the global-pCN plateau. Revisit the conditioned formulation "
            "before proceeding."
        )
    return (
        "UNRESOLVED: at least one final relative-k checkpoint window has not "
        "flattened, or red-black has flattened lower without a descending "
        "data-misfit tail. Increase --max-seconds, --sweeps, and --pcn-iters "
        "until both final checkpoint windows stabilize, then apply the same "
        "recovery-floor and misfit-direction criteria."
    )


def _print_summary_table(
    summaries: Sequence[DecisionSummary], noise_floor: float
) -> None:
    print(
        "scheme              stop      updates/chain             rel-k shape      "
        "flat d_relk max_step misfit_shape tail_misfit min_misfit misfit/floor "
        "coverage accept stateN nullN nullMax thetaP end_Rhat best_Rhat min_ESS "
        "solves seconds ESS/1k ESS/sec"
    )
    for summary in summaries:
        updates = ",".join(str(value) for value in summary.updates_per_chain)
        print(
            f"{summary.label:<19} {summary.stopped_by:<9} {updates:<25} "
            f"{summary.relative_k_error:6.4f} {summary.trajectory_shape:<10} "
            f"{str(summary.relative_k_stable):<5} "
            f"{summary.relative_k_window_change:7.4f} "
            f"{summary.max_relative_k_checkpoint_change:8.4f} "
            f"{summary.misfit_trajectory_shape:<12} "
            f"{summary.tail_mean_misfit:10.3f} {summary.min_misfit:9.3f} "
            f"{summary.misfit_over_noise_floor:11.3f} {summary.coverage:8.3f} "
            f"{summary.acceptance_rate:6.3f} "
            f"{summary.final_accepted_state_theta_norm:6.2f} "
            f"{summary.final_null_space_norm:5.2f} "
            f"{summary.max_null_space_norm:7.2f} {summary.mean_theta_p_norm:6.2f} "
            f"{summary.endpoint_max_r_hat:8.3f} "
            f"{summary.best_max_r_hat:9.3f} {summary.conservative_total_ess:7.2f} "
            f"{summary.total_forward_solves:6d} {summary.wall_seconds:7.2f} "
            f"{summary.ess_per_1000_solves:6.3f} {summary.ess_per_second:7.3f}"
        )
    print(f"nominal observation-noise misfit floor: {noise_floor:.3f}")


def _print_trajectories(summaries: Sequence[DecisionSummary]) -> None:
    print(
        "scheme              updates/chain solves rel-k(mean G) mean_misfit "
        "misfit/floor state_norm null_norm theta_p max_Rhat"
    )
    for summary in summaries:
        for point in summary.trajectory:
            print(
                f"{summary.label:<19} {point.mean_updates_per_chain:13.1f} "
                f"{point.total_forward_solves:6d} {point.relative_k_error:13.4e} "
                f"{point.mean_misfit:11.3f} {point.misfit_over_noise_floor:11.3f} "
                f"{point.mean_accepted_state_theta_norm:10.3f} "
                f"{point.mean_null_space_norm:9.3f} "
                f"{point.mean_theta_p_norm:7.3f} {point.max_r_hat:8.3f}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decide",
        action="store_true",
        help="Run the opt-in generous recovery-decision profile.",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Run the opt-in resolving profile with larger budgets and cap.",
    )
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--red-black-sweeps", "--sweeps", type=int, default=None)
    parser.add_argument("--global-iter", "--pcn-iters", type=int, default=None)
    parser.add_argument("--wall-cap-seconds", "--max-seconds", type=float, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--trajectory-points", "--checkpoints", type=int, default=None)
    parser.add_argument("--stabilization-window", type=int, default=None)
    parser.add_argument("--stabilization-tolerance", type=float, default=None)
    parser.add_argument("--recovery-gap-tolerance", type=float, default=None)
    parser.add_argument("--misfit-trend-fraction", type=float, default=None)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--burn-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    parser.add_argument(
        "--block-gibbs",
        action="store_true",
        help="Also run the opt-in precision block-Gibbs scheme.",
    )
    parser.add_argument(
        "--block-nugget",
        type=float,
        default=None,
        help="Override the block-Gibbs nugget tau^2 (default Config.block_nugget).",
    )
    args = parser.parse_args()

    if args.n_chains < 4:
        raise ValueError("n-chains must be at least 4 for the recovery decision.")
    if args.burn_fraction < 0.0 or args.burn_fraction >= 1.0:
        raise ValueError("burn-fraction must satisfy 0 <= burn-fraction < 1.")
    profile = _resolved_profile(args)

    cfg = Config(seed=args.seed, beta=args.beta, n_chains=args.n_chains)
    truth, theta_true_draw, noise_draw = _truth_replay_prefix(cfg)
    Phi, lam = _global_kle(cfg)
    seeds = [args.seed + 1_000 + idx for idx in range(args.n_chains)]
    per_chain_cap = profile.wall_cap_seconds / args.n_chains
    noise_floor = 0.5 * cfg.n_obs_x * cfg.n_obs_y
    block_nugget = (
        cfg.block_nugget if args.block_nugget is None else float(args.block_nugget)
    )

    red_black_global_field = [
        _red_black_chain(
            cfg=cfg,
            truth_draw=theta_true_draw,
            noise_draw=noise_draw,
            Phi=Phi,
            lam=lam,
            chain_seed=chain_seed,
            n_sweeps=profile.red_black_sweeps,
            Mb=args.mb,
            beta=args.beta,
            burn_fraction=args.burn_fraction,
            sample_stride=profile.sample_stride,
            wall_cap_seconds=per_chain_cap,
            prior_mode="global_field",
        )
        for chain_seed in seeds
    ]
    red_black_null_space = [
        _red_black_chain(
            cfg=cfg,
            truth_draw=theta_true_draw,
            noise_draw=noise_draw,
            Phi=Phi,
            lam=lam,
            chain_seed=chain_seed,
            n_sweeps=profile.red_black_sweeps,
            Mb=args.mb,
            beta=args.beta,
            burn_fraction=args.burn_fraction,
            sample_stride=profile.sample_stride,
            wall_cap_seconds=per_chain_cap,
            prior_mode="null_space",
        )
        for chain_seed in seeds
    ]
    global_pcn = [
        _global_pcn_chain(
            cfg=cfg,
            truth=truth,
            Phi=Phi,
            lam=lam,
            chain_seed=chain_seed,
            n_iter=profile.global_iter,
            beta=args.beta,
            burn_fraction=args.burn_fraction,
            sample_stride=profile.sample_stride,
            wall_cap_seconds=per_chain_cap,
        )
        for chain_seed in seeds
    ]
    summaries = [
        _summarize(
            "rb_global_field",
            red_black_global_field,
            truth,
            noise_floor,
            args.coverage_level,
            profile.trajectory_points,
            profile.stabilization_window,
            profile.stabilization_tolerance,
            profile.misfit_trend_fraction,
        ),
        _summarize(
            "rb_null_space",
            red_black_null_space,
            truth,
            noise_floor,
            args.coverage_level,
            profile.trajectory_points,
            profile.stabilization_window,
            profile.stabilization_tolerance,
            profile.misfit_trend_fraction,
        ),
        _summarize(
            "global_pcn",
            global_pcn,
            truth,
            noise_floor,
            args.coverage_level,
            profile.trajectory_points,
            profile.stabilization_window,
            profile.stabilization_tolerance,
            profile.misfit_trend_fraction,
        ),
    ]

    block_gibbs_summary: DecisionSummary | None = None
    if args.block_gibbs:
        block_gibbs = [
            _block_gibbs_chain(
                cfg=cfg,
                truth_draw=theta_true_draw,
                noise_draw=noise_draw,
                Phi=Phi,
                lam=lam,
                chain_seed=chain_seed,
                n_sweeps=profile.red_black_sweeps,
                beta=args.beta,
                burn_fraction=args.burn_fraction,
                sample_stride=profile.sample_stride,
                wall_cap_seconds=per_chain_cap,
                block_nugget=block_nugget,
            )
            for chain_seed in seeds
        ]
        block_gibbs_summary = _summarize(
            "block_gibbs",
            block_gibbs,
            truth,
            noise_floor,
            args.coverage_level,
            profile.trajectory_points,
            profile.stabilization_window,
            profile.stabilization_tolerance,
            profile.misfit_trend_fraction,
        )
        summaries.append(block_gibbs_summary)

    print("M9C RECOVERY DECISION")
    print(
        f"profile={profile.label}; grid={cfg.ny} x {cfg.nx}; "
        f"n_chains={args.n_chains}; Mb={args.mb}; beta={args.beta}; seed={args.seed}"
    )
    print(
        f"red_black_sweeps={profile.red_black_sweeps}; "
        f"global_iter={profile.global_iter}; wall_cap_seconds={profile.wall_cap_seconds}; "
        f"sample_stride={profile.sample_stride}; burn_fraction={args.burn_fraction:.3f}; "
        f"stabilization_window={profile.stabilization_window}; "
        f"stabilization_tolerance={profile.stabilization_tolerance}"
    )
    print(
        "Comparison note: single-subdomain is excluded because it freezes the "
        "rest of the field. Both compared schemes use matched unit-scale prior "
        "starts and the same synthetic truth."
    )
    print(
        "Prior note: rb_global_field uses delta_like + delta_global_field_prior "
        "+ hard-null pCN q-ratio. rb_null_space uses the non-degenerate "
        "reference-style delta_like - 0.5 * delta(||Z.T theta||^2); adding the "
        "pCN q-ratio there would cancel the null-prior term into a no-op."
    )
    if block_gibbs_summary is not None:
        print(
            "Block-Gibbs note: the opt-in block_gibbs scheme is the "
            "posterior-correct precision update (exact full conditionals of "
            f"N(0, C_tau), tau^2={block_nugget:g}); its acceptance is exactly "
            "likelihood-only and a sweep targets pi(G | Y)."
        )
    print()
    print("RECOVERY AND COMPUTE-FAIR SUMMARY")
    _print_summary_table(summaries, noise_floor)
    print()
    print("TRAJECTORY CHECKPOINTS")
    _print_trajectories(summaries)
    print("Verdict:")
    print(
        f"  {_verdict(summaries[1], summaries[2], noise_floor, recovery_gap_tolerance=profile.recovery_gap_tolerance)}"
    )

    if block_gibbs_summary is not None:
        global_pcn_summary = summaries[2]
        reversal_cured = (
            block_gibbs_summary.relative_k_stable
            and block_gibbs_summary.trajectory_shape != "rising"
            and block_gibbs_summary.relative_k_error
            <= global_pcn_summary.relative_k_error + profile.recovery_gap_tolerance
        )
        print()
        print("Block-Gibbs recovery check (vs global pCN):")
        print(
            f"  block_gibbs rel-k={block_gibbs_summary.relative_k_error:.4f} "
            f"(shape={block_gibbs_summary.trajectory_shape}, "
            f"flat={block_gibbs_summary.relative_k_stable}, "
            f"tail_misfit={block_gibbs_summary.tail_mean_misfit:.3f}); "
            f"global_pcn rel-k={global_pcn_summary.relative_k_error:.4f}."
        )
        print(
            "  REVERSAL CURED: block-Gibbs relative-k flattened at or below the "
            "global-pCN plateau without rising."
            if reversal_cured
            else "  NOT YET CURED: block-Gibbs relative-k has not flattened at or "
            "below global pCN; increase --max-seconds/--sweeps or retune "
            "--block-nugget."
        )


if __name__ == "__main__":
    main()
