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
    RedBlackSamplerState,
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


@dataclass(frozen=True)
class DecisionChain:
    fields: np.ndarray
    misfit: np.ndarray
    coefficients: np.ndarray
    theta_norm: np.ndarray
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


@dataclass(frozen=True)
class DecisionSummary:
    label: str
    relative_k_error: float
    trajectory_shape: str
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
    defaults = (
        DecisionProfile("decision", 500, 10_000, 600.0, 8, 8)
        if args.decide
        else DecisionProfile("responsive", 2, 64, 60.0, 1, 4)
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
    )
    if profile.red_black_sweeps < 1 or profile.global_iter < 1:
        raise ValueError("chain lengths must be positive.")
    if profile.wall_cap_seconds <= 0.0:
        raise ValueError("wall-cap-seconds must be positive.")
    if profile.sample_stride < 1:
        raise ValueError("sample-stride must be positive.")
    if profile.trajectory_points < 2:
        raise ValueError("trajectory-points must be at least 2.")
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
    sample_updates: list[int] = []
    sample_solves: list[int] = []
    accepted: list[bool] = []
    last_sample: tuple[np.ndarray, float, int] | None = None
    started = perf_counter()

    for idx, state in enumerate(states, start=1):
        accepted.append(state.accepted)
        if idx > burn:
            field, state_misfit = field_and_misfit(state)
            sample = (field, state_misfit, idx + solve_offset)
            last_sample = sample
            if (idx - burn - 1) % sample_stride == 0:
                fields.append(field)
                misfit.append(state_misfit)
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
    if not sample_solves or sample_solves[-1] != last_sample[2]:
        fields.append(last_sample[0])
        misfit.append(last_sample[1])
        sample_updates.append(last_sample[2] - solve_offset)
        sample_solves.append(last_sample[2])
    return (
        fields,
        misfit,
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
    )

    def field_and_misfit(state: RedBlackSamplerState) -> tuple[np.ndarray, float]:
        return state.G_accepted.copy(), -state.log_likelihood_accepted

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

    def field_and_misfit(state: MCMCState) -> tuple[np.ndarray, float]:
        field = reshape_field(field_from_theta(Phi, lam, state.theta), cfg.ny, cfg.nx)
        return field, -(state.log_density - log_prior(state.theta))

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


def _summarize(
    label: str,
    chains: Sequence[DecisionChain],
    truth: TruthData,
    noise_floor: float,
    level: float,
    trajectory_points: int,
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
    tail = max(1, trimmed[0].misfit.size // 10)
    tail_misfit = np.concatenate([chain.misfit[-tail:] for chain in trimmed])
    all_misfit = np.concatenate([chain.misfit for chain in trimmed])
    return DecisionSummary(
        label=label,
        relative_k_error=relative_error(
            permeability_from_log_field(G_mean), truth.k_true
        ),
        trajectory_shape=_trajectory_shape(trajectory),
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
    level: float,
) -> str:
    converged = (
        red_black.endpoint_max_r_hat <= 1.05 and global_pcn.endpoint_max_r_hat <= 1.05
    )
    red_black_fits = red_black.tail_mean_misfit <= 1.5 * noise_floor
    global_fits = global_pcn.tail_mean_misfit <= 1.5 * noise_floor
    similar_recovery = (
        abs(red_black.relative_k_error - global_pcn.relative_k_error) <= 0.1
    )
    calibrated = (
        abs(red_black.coverage - level) <= 0.1
        and abs(global_pcn.coverage - level) <= 0.1
    )
    materially_worse_coverage = global_pcn.coverage - red_black.coverage > 0.1
    plateau_gap = (
        red_black.trajectory_shape == "plateau"
        and red_black.relative_k_error > global_pcn.relative_k_error + 0.1
    )

    if converged and red_black_fits and global_fits and similar_recovery and calibrated:
        return (
            "DATA-LIMITED: both schemes reach the noise floor, recovery floors "
            "are similar, and coverage is calibrated. Conditioning's value is "
            "compute efficiency; proceed to M10/M11 without route (b)."
        )
    if converged and (
        (not red_black_fits and global_fits) or plateau_gap or materially_worse_coverage
    ):
        return (
            "CONDITIONING DEFECT: converged red-black diagnostics trail global "
            "pCN in data fit, plateaued recovery, or coverage. Route section "
            "3.8(b) is justified as the next milestone."
        )
    return (
        "AMBIGUOUS: at least one scheme is not converged or the trajectories "
        "have not separated decisively. Increase --wall-cap-seconds and chain "
        "budgets until endpoint max R_hat <= 1.05 for both schemes and the "
        "last two relative-k checkpoints are stable; then apply the same "
        "data-fit, recovery-gap, and coverage criteria."
    )


def _print_summary_table(
    summaries: Sequence[DecisionSummary], noise_floor: float
) -> None:
    print(
        "scheme              stop      updates/chain             rel-k shape      "
        "tail_misfit min_misfit misfit/floor coverage accept end_Rhat best_Rhat "
        "min_ESS solves seconds ESS/1k ESS/sec"
    )
    for summary in summaries:
        updates = ",".join(str(value) for value in summary.updates_per_chain)
        print(
            f"{summary.label:<19} {summary.stopped_by:<9} {updates:<25} "
            f"{summary.relative_k_error:6.4f} {summary.trajectory_shape:<10} "
            f"{summary.tail_mean_misfit:10.3f} {summary.min_misfit:9.3f} "
            f"{summary.misfit_over_noise_floor:11.3f} {summary.coverage:8.3f} "
            f"{summary.acceptance_rate:6.3f} {summary.endpoint_max_r_hat:8.3f} "
            f"{summary.best_max_r_hat:9.3f} {summary.conservative_total_ess:7.2f} "
            f"{summary.total_forward_solves:6d} {summary.wall_seconds:7.2f} "
            f"{summary.ess_per_1000_solves:6.3f} {summary.ess_per_second:7.3f}"
        )
    print(f"nominal observation-noise misfit floor: {noise_floor:.3f}")


def _print_trajectories(summaries: Sequence[DecisionSummary]) -> None:
    print(
        "scheme              updates/chain solves rel-k(mean G) mean_misfit "
        "misfit/floor max_Rhat"
    )
    for summary in summaries:
        for point in summary.trajectory:
            print(
                f"{summary.label:<19} {point.mean_updates_per_chain:13.1f} "
                f"{point.total_forward_solves:6d} {point.relative_k_error:13.4e} "
                f"{point.mean_misfit:11.3f} {point.misfit_over_noise_floor:11.3f} "
                f"{point.max_r_hat:8.3f}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decide",
        action="store_true",
        help="Run the opt-in generous recovery-decision profile.",
    )
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--red-black-sweeps", type=int, default=None)
    parser.add_argument("--global-iter", type=int, default=None)
    parser.add_argument("--wall-cap-seconds", type=float, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--trajectory-points", type=int, default=None)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--burn-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--coverage-level", type=float, default=0.9)
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

    red_black = [
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
            "posterior_red_black",
            red_black,
            truth,
            noise_floor,
            args.coverage_level,
            profile.trajectory_points,
        ),
        _summarize(
            "global_pcn",
            global_pcn,
            truth,
            noise_floor,
            args.coverage_level,
            profile.trajectory_points,
        ),
    ]

    print("M9C RECOVERY DECISION")
    print(
        f"profile={profile.label}; grid={cfg.ny} x {cfg.nx}; "
        f"n_chains={args.n_chains}; Mb={args.mb}; beta={args.beta}; seed={args.seed}"
    )
    print(
        f"red_black_sweeps={profile.red_black_sweeps}; "
        f"global_iter={profile.global_iter}; wall_cap_seconds={profile.wall_cap_seconds}; "
        f"sample_stride={profile.sample_stride}; burn_fraction={args.burn_fraction:.3f}"
    )
    print(
        "Comparison note: single-subdomain is excluded because it freezes the "
        "rest of the field. Both compared schemes use matched unit-scale prior "
        "starts and the same synthetic truth."
    )
    print()
    print("RECOVERY AND COMPUTE-FAIR SUMMARY")
    _print_summary_table(summaries, noise_floor)
    print()
    print("TRAJECTORY CHECKPOINTS")
    _print_trajectories(summaries)
    print("Verdict:")
    print(f"  {_verdict(summaries[0], summaries[1], noise_floor, args.coverage_level)}")


if __name__ == "__main__":
    main()
