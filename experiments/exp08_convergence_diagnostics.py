"""M9 convergence diagnostics and recovery comparison."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.bayes import log_posterior, log_prior  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.diagnostics import (  # noqa: E402
    credible_interval_coverage,
    effective_sample_size,
    gelman_rubin,
    integrated_autocorr_time,
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
from mcmc_multiscale.grid import cell_centered_grid  # noqa: E402
from mcmc_multiscale.kle import top_eigenpairs  # noqa: E402
from mcmc_multiscale.mcmc import metropolis_hastings  # noqa: E402
from mcmc_multiscale.observations import TruthData, make_truth  # noqa: E402
from mcmc_multiscale.proposals import make_pcn_proposal  # noqa: E402
from mcmc_multiscale.sampler import (  # noqa: E402
    conditioned_sampler,
    red_black_conditioned_sampler,
)


@dataclass(frozen=True)
class ScalarSummary:
    label: str
    r_hat: float
    mean_iat: float
    total_ess: float


@dataclass(frozen=True)
class SchemeSummary:
    label: str
    relative_k_error: float
    coverage: float
    acceptance_rate: float
    n_retained_per_chain: int
    total_forward_solves: int
    wall_seconds: float
    conservative_total_ess: float
    max_r_hat: float
    ess_per_1000_solves: float
    ess_per_second: float
    scalar_summaries: tuple[ScalarSummary, ...]


@dataclass(frozen=True)
class _ChainSamples:
    fields: np.ndarray
    misfit: np.ndarray
    theta_norm: np.ndarray
    coefficients: np.ndarray
    acceptance_rate: float
    forward_solves: int
    wall_seconds: float


@dataclass(frozen=True)
class _RunLengths:
    profile: str
    single_iter: int
    red_black_sweeps: int
    global_iter: int


class _TruthReplayGenerator:
    """Replay one common synthetic truth, then delegate independent draws.

    M8 uses one generator for truth creation and chain updates. M9 must compare
    independent chains targeting the same posterior without changing sampler
    code, so this experiment-only adapter replays the shared truth prefix and
    delegates all subsequent draws to a seeded `numpy.random.Generator`.
    """

    def __init__(
        self,
        theta_true_draw: np.ndarray,
        noise_draw: np.ndarray,
        chain_seed: int,
        n_initial_draws: int,
        initial_scale: float,
    ) -> None:
        self._truth_prefix = [theta_true_draw.copy(), noise_draw.copy()]
        self._delegate = np.random.default_rng(chain_seed)
        self._n_initial_draws = n_initial_draws
        self._initial_scale = initial_scale

    def standard_normal(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype: type[np.float64] = np.float64,
    ) -> np.ndarray:
        if self._truth_prefix:
            replay = self._truth_prefix.pop(0)
            if replay.shape != np.empty(size).shape:
                raise ValueError(
                    "truth replay draw shape does not match sampler request."
                )
            return replay.astype(dtype, copy=True)

        draw = self._delegate.standard_normal(size=size, dtype=dtype)
        if self._n_initial_draws > 0:
            draw = self._initial_scale * draw
            self._n_initial_draws -= 1
        return draw

    def uniform(self) -> float:
        return float(self._delegate.uniform())


def _global_kle(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    covariance = exp_covariance(points, cfg.sigma, cfg.corr_length)
    return top_eigenpairs(covariance, cfg.n_global_modes)


def _truth_replay_prefix(cfg: Config) -> tuple[TruthData, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    truth = make_truth(cfg, np.random.default_rng(cfg.seed))
    theta_true_draw = rng.standard_normal(cfg.n_global_modes, dtype=np.float64)
    noise_draw = rng.standard_normal(truth.y_obs.shape, dtype=np.float64)
    return truth, theta_true_draw, noise_draw


def _retained_start(n_samples: int, burn_fraction: float) -> int:
    burn = int(n_samples * burn_fraction)
    if burn >= n_samples:
        raise ValueError("burn-in must leave at least one retained sample.")
    return burn


def _project_coefficients(
    fields: np.ndarray, Phi: np.ndarray, lam: np.ndarray
) -> np.ndarray:
    vectors = np.stack([field.ravel(order="F") for field in fields])
    return (vectors @ Phi) / np.sqrt(lam)


def _conditioned_chain(
    cfg: Config,
    truth_draw: np.ndarray,
    noise_draw: np.ndarray,
    chain_seed: int,
    update_scheme: str,
    n_iter: int,
    Mb: int,
    beta: float,
    burn_fraction: float,
    initial_scale: float,
    Phi: np.ndarray,
    lam: np.ndarray,
) -> _ChainSamples:
    n_subdomains = cfg.n_coarse_x * cfg.n_coarse_y
    n_initial_draws = 2 if update_scheme == "single" else 1 + n_subdomains
    rng = _TruthReplayGenerator(
        theta_true_draw=truth_draw,
        noise_draw=noise_draw,
        chain_seed=chain_seed,
        n_initial_draws=n_initial_draws,
        initial_scale=initial_scale,
    )
    if update_scheme == "single":
        expected_updates = n_iter
        states = conditioned_sampler(
            cfg=cfg,
            n_iter=n_iter,
            Mb=Mb,
            theta_p_method="svd",
            rng=rng,  # type: ignore[arg-type]
            beta=beta,
            acceptance="posterior",
        )
    elif update_scheme == "red_black":
        expected_updates = n_iter * n_subdomains
        states = red_black_conditioned_sampler(
            cfg=cfg,
            n_sweeps=n_iter,
            Mb=Mb,
            theta_p_method="svd",
            rng=rng,  # type: ignore[arg-type]
            beta=beta,
            acceptance="posterior",
        )
    else:
        raise ValueError("update_scheme must be 'single' or 'red_black'.")

    burn = _retained_start(expected_updates, burn_fraction)
    fields_retained: list[np.ndarray] = []
    misfit_retained: list[float] = []
    accepted: list[bool] = []
    started = perf_counter()
    for idx, state in enumerate(states):
        accepted.append(state.accepted)
        if idx >= burn:
            fields_retained.append(state.G_accepted)
            misfit_retained.append(-state.log_likelihood_accepted)
    wall_seconds = perf_counter() - started
    if len(accepted) != expected_updates:
        raise RuntimeError("conditioned sampler yielded an unexpected update count.")

    fields = np.stack(fields_retained)
    coefficients = _project_coefficients(fields, Phi, lam)
    return _ChainSamples(
        fields=fields,
        misfit=np.asarray(misfit_retained, dtype=np.float64),
        theta_norm=np.linalg.norm(coefficients, axis=1),
        coefficients=coefficients,
        acceptance_rate=float(np.mean(accepted)),
        forward_solves=expected_updates + 2,
        wall_seconds=wall_seconds,
    )


def _global_pcn_chain(
    cfg: Config,
    truth: TruthData,
    Phi: np.ndarray,
    lam: np.ndarray,
    chain_seed: int,
    n_iter: int,
    beta: float,
    burn_fraction: float,
    initial_scale: float,
) -> _ChainSamples:
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
        theta0=initial_scale
        * rng.standard_normal(cfg.n_global_modes, dtype=np.float64),
        n_iter=n_iter,
        rng=rng,
        log_prior_fn=log_prior,
    )
    burn = _retained_start(n_iter, burn_fraction)
    coefficient_retained: list[np.ndarray] = []
    misfit_retained: list[float] = []
    accepted: list[bool] = []
    started = perf_counter()
    for idx, state in enumerate(states):
        accepted.append(state.accepted)
        if idx >= burn:
            coefficient_retained.append(state.theta)
            misfit_retained.append(-(state.log_density - log_prior(state.theta)))
    wall_seconds = perf_counter() - started
    if len(accepted) != n_iter:
        raise RuntimeError("global pCN sampler yielded an unexpected update count.")

    coefficients = np.stack(coefficient_retained)
    fields = np.stack(
        [
            reshape_field(field_from_theta(Phi, lam, theta), cfg.ny, cfg.nx)
            for theta in coefficients
        ]
    )
    return _ChainSamples(
        fields=fields,
        misfit=np.asarray(misfit_retained, dtype=np.float64),
        theta_norm=np.linalg.norm(coefficients, axis=1),
        coefficients=coefficients,
        acceptance_rate=float(np.mean(accepted)),
        forward_solves=n_iter + 1,
        wall_seconds=wall_seconds,
    )


def _scalar_summary(label: str, chains: np.ndarray) -> ScalarSummary:
    return ScalarSummary(
        label=label,
        r_hat=gelman_rubin(chains),
        mean_iat=float(np.mean([integrated_autocorr_time(chain) for chain in chains])),
        total_ess=float(np.sum([effective_sample_size(chain) for chain in chains])),
    )


def _summarize_scheme(
    label: str,
    chains: Sequence[_ChainSamples],
    truth: TruthData,
    level: float,
) -> SchemeSummary:
    fields = np.concatenate([chain.fields for chain in chains], axis=0)
    G_mean, _ = posterior_summary(fields)
    k_from_mean_G = permeability_from_log_field(G_mean)
    scalar_summaries = [
        _scalar_summary("misfit", np.stack([chain.misfit for chain in chains])),
        _scalar_summary("theta_norm", np.stack([chain.theta_norm for chain in chains])),
    ]
    n_coefficients = min(3, chains[0].coefficients.shape[1])
    scalar_summaries.extend(
        _scalar_summary(
            f"theta_global_{idx}",
            np.stack([chain.coefficients[:, idx] for chain in chains]),
        )
        for idx in range(n_coefficients)
    )
    scalar_summaries_tuple = tuple(scalar_summaries)
    conservative_total_ess = min(
        summary.total_ess for summary in scalar_summaries_tuple
    )
    total_forward_solves = sum(chain.forward_solves for chain in chains)
    wall_seconds = sum(chain.wall_seconds for chain in chains)
    ess_per_1000_solves, ess_per_second = sampling_efficiency(
        conservative_total_ess, total_forward_solves, wall_seconds
    )
    return SchemeSummary(
        label=label,
        relative_k_error=relative_error(k_from_mean_G, truth.k_true),
        coverage=credible_interval_coverage(fields, truth.G_true, level=level),
        acceptance_rate=float(np.mean([chain.acceptance_rate for chain in chains])),
        n_retained_per_chain=chains[0].fields.shape[0],
        total_forward_solves=total_forward_solves,
        wall_seconds=wall_seconds,
        conservative_total_ess=conservative_total_ess,
        max_r_hat=max(summary.r_hat for summary in scalar_summaries_tuple),
        ess_per_1000_solves=ess_per_1000_solves,
        ess_per_second=ess_per_second,
        scalar_summaries=scalar_summaries_tuple,
    )


def _print_summaries(summaries: Sequence[SchemeSummary], level: float) -> None:
    print(
        "scheme             rel-k(mean G) coverage accept retained/chain "
        "scalar          R_hat mean_IAT total_ESS"
    )
    for summary in summaries:
        for idx, scalar in enumerate(summary.scalar_summaries):
            scheme = summary.label if idx == 0 else ""
            rel_k = f"{summary.relative_k_error:.4e}" if idx == 0 else ""
            coverage = f"{summary.coverage:.3f}" if idx == 0 else ""
            acceptance = f"{summary.acceptance_rate:.3f}" if idx == 0 else ""
            retained = str(summary.n_retained_per_chain) if idx == 0 else ""
            print(
                f"{scheme:<18} {rel_k:>13} {coverage:>8} {acceptance:>6} "
                f"{retained:>14} {scalar.label:<14} {scalar.r_hat:6.3f} "
                f"{scalar.mean_iat:8.2f} {scalar.total_ess:9.2f}"
            )
    print(f"central credible-interval level: {level:.2f}")


def _print_compute_fair_table(summaries: Sequence[SchemeSummary]) -> None:
    print(
        "scheme              rel-k(mean G) coverage max_Rhat min_total_ESS "
        "forward_solves wall_seconds ESS/1k_solves ESS/second"
    )
    for summary in summaries:
        print(
            f"{summary.label:<19} {summary.relative_k_error:13.4e} "
            f"{summary.coverage:8.3f} {summary.max_r_hat:8.3f} "
            f"{summary.conservative_total_ess:13.2f} "
            f"{summary.total_forward_solves:14d} {summary.wall_seconds:12.2f} "
            f"{summary.ess_per_1000_solves:13.3f} "
            f"{summary.ess_per_second:10.3f}"
        )
    print(
        "Compute normalization uses the minimum total ESS across reported "
        "scalars and counts actual scheme-local forward solves, including "
        "initialization. Wall time includes scheme-local setup."
    )


def _verdict(summaries: Sequence[SchemeSummary], level: float) -> str:
    single, red_black, baseline = summaries
    if any(summary.max_r_hat > 1.05 for summary in summaries):
        return (
            "INCONCLUSIVE: at least one scheme has R_hat above 1.05. The "
            "compute-normalized table is diagnostic only until longer chains "
            "converge; route (b) remains justified for investigation, not yet "
            "as a proven fix."
        )
    coverage_near_nominal = (
        abs(red_black.coverage - level) <= 0.1 and abs(baseline.coverage - level) <= 0.1
    )
    conditioned_best = min(single.relative_k_error, red_black.relative_k_error)
    baseline_similar = abs(conditioned_best - baseline.relative_k_error) <= 0.1
    if coverage_near_nominal and baseline_similar:
        return (
            "DATA-LIMITED: converged calibrated coverage and similar recovery "
            "support genuine posterior uncertainty. Use ESS per solve and per "
            "second to judge conditioning's speed value."
        )
    if (
        red_black.coverage < level - 0.2
        or baseline.relative_k_error + 0.1 < conditioned_best
    ):
        return (
            "DEFECT SIGNAL: converged conditioned recovery trails global pCN "
            "materially or coverage is poor. Route (b) is justified."
        )
    return (
        "INCONCLUSIVE: the short run does not cleanly separate data-limited "
        "uncertainty from a conditioned-sampler defect."
    )


def _resolved_lengths(args: argparse.Namespace) -> _RunLengths:
    defaults = (
        _RunLengths("long", 2_000, 100, 5_000)
        if args.long
        else _RunLengths("responsive", 200, 10, 500)
    )
    lengths = _RunLengths(
        profile=defaults.profile,
        single_iter=(
            defaults.single_iter if args.single_iter is None else args.single_iter
        ),
        red_black_sweeps=(
            defaults.red_black_sweeps
            if args.red_black_sweeps is None
            else args.red_black_sweeps
        ),
        global_iter=(
            defaults.global_iter if args.global_iter is None else args.global_iter
        ),
    )
    if (
        lengths.single_iter < 1
        or lengths.red_black_sweeps < 1
        or lengths.global_iter < 1
    ):
        raise ValueError("all chain lengths must be positive.")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Use the opt-in long convergence profile.",
    )
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--single-iter", type=int, default=None)
    parser.add_argument("--red-black-sweeps", type=int, default=None)
    parser.add_argument("--global-iter", type=int, default=None)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--burn-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--initial-scale", type=float, default=2.0)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    args = parser.parse_args()

    if args.n_chains < 2:
        raise ValueError("n-chains must be at least 2 for R_hat.")
    if args.initial_scale <= 0.0:
        raise ValueError("initial-scale must be positive.")
    lengths = _resolved_lengths(args)

    cfg = Config(seed=args.seed, beta=args.beta, n_chains=args.n_chains)
    truth, theta_true_draw, noise_draw = _truth_replay_prefix(cfg)
    Phi, lam = _global_kle(cfg)
    seeds = [args.seed + 1_000 + idx for idx in range(args.n_chains)]

    single = [
        _conditioned_chain(
            cfg,
            theta_true_draw,
            noise_draw,
            chain_seed,
            "single",
            lengths.single_iter,
            args.mb,
            args.beta,
            args.burn_fraction,
            args.initial_scale,
            Phi,
            lam,
        )
        for chain_seed in seeds
    ]
    red_black = [
        _conditioned_chain(
            cfg,
            theta_true_draw,
            noise_draw,
            chain_seed,
            "red_black",
            lengths.red_black_sweeps,
            args.mb,
            args.beta,
            args.burn_fraction,
            args.initial_scale,
            Phi,
            lam,
        )
        for chain_seed in seeds
    ]
    baseline = [
        _global_pcn_chain(
            cfg,
            truth,
            Phi,
            lam,
            chain_seed + 1_000,
            lengths.global_iter,
            args.beta,
            args.burn_fraction,
            args.initial_scale,
        )
        for chain_seed in seeds
    ]
    summaries = [
        _summarize_scheme("posterior_single", single, truth, args.coverage_level),
        _summarize_scheme("posterior_red_black", red_black, truth, args.coverage_level),
        _summarize_scheme("global_pcn", baseline, truth, args.coverage_level),
    ]

    print("M9 CONVERGENCE DIAGNOSTICS")
    print(
        f"profile={lengths.profile}; grid={cfg.ny} x {cfg.nx}; "
        f"n_chains={args.n_chains}; Mb={args.mb}; beta={args.beta}; "
        f"seed={args.seed}; initial_scale={args.initial_scale}"
    )
    print(
        f"single_iter={lengths.single_iter}; "
        f"red_black_sweeps={lengths.red_black_sweeps}; "
        f"global_iter={lengths.global_iter}; burn_fraction={args.burn_fraction:.3f}"
    )
    print(
        "Conditioned-chain setup: one shared synthetic truth is replayed before "
        "distinct seeded chain streams begin."
    )
    print()
    _print_summaries(summaries, args.coverage_level)
    print()
    print("COMPUTE-FAIR COMPARISON")
    _print_compute_fair_table(summaries)
    print()
    print("Verdict:")
    print(f"  {_verdict(summaries, args.coverage_level)}")
    print(
        "  Use longer chains before making a recovery claim when R_hat or ESS "
        "shows that the responsive default run is not converged."
    )


if __name__ == "__main__":
    main()
