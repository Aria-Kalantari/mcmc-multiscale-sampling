"""M11 acceleration study: block-Gibbs (systematic / red-black) vs global pCN.

Headline question: at modest stochastic dimension (N = n_global_modes in
{64, 90}), does a local/block update reach a given Gelman-Rubin R_hat in fewer
forward solves / less wall-time than global pCN -- especially with a red-black
scan order? Deliverables: (1) convergence speed (R_hat vs forward solves, ESS
per 1000 solves, ESS per second), (2) posterior-mean field plausibility
heatmaps, (3) interface-jump RMS comparing a naive independent-local-KLE stitch
against the conditioned (block-Gibbs) and global-pCN posterior means.

Measurement only: it imports the samplers unchanged and adds no numerical
kernel. The fast default is responsive; the real comparison is opt-in via
``--long``. Honest reporting: under the dense KLE+nugget precision, red-black is
only a sequential ordering (same-color cores are coupled), so a limited or
absent speedup is the expected, reported outcome.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.exp08_convergence_diagnostics import (  # noqa: E402
    _TruthReplayGenerator,
    _project_coefficients,
    _truth_replay_prefix,
)
from mcmc_multiscale.bayes import log_posterior, log_prior  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.diagnostics import (  # noqa: E402
    effective_sample_size,
    gelman_rubin,
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
from mcmc_multiscale.observations import TruthData  # noqa: E402
from mcmc_multiscale.proposals import make_pcn_proposal  # noqa: E402
from mcmc_multiscale.sampler import block_gibbs_sampler  # noqa: E402
from mcmc_multiscale.subdomain import (  # noqa: E402
    interface_jump_rms,
    iter_coarse_subdomains,
    make_subdomain_at,
)

SCALARS = ("misfit", "theta_norm", "theta_0", "theta_1", "theta_2")
R_HAT_THRESHOLD = 1.1


@dataclass(frozen=True)
class ChainSamples:
    solves: np.ndarray  # (n_samples,) cumulative forward solves at each sample
    fields: np.ndarray  # (n_samples, ny, nx) accepted log-field
    misfit: np.ndarray  # (n_samples,)
    coefficients: np.ndarray  # (n_samples, N) global-KLE projection
    wall_seconds: float


@dataclass(frozen=True)
class MethodResult:
    label: str
    chains: tuple[ChainSamples, ...]
    rhat_solves: np.ndarray  # (n_checkpoints,)
    rhat_curve: np.ndarray  # (n_checkpoints,) max-scalar R_hat (2nd-half)
    best_rhat: float
    endpoint_rhat: float
    solves_to_threshold: int | None
    conservative_total_ess: float
    total_forward_solves: int
    wall_seconds: float
    ess_per_1000_solves: float
    ess_per_second: float
    posterior_mean_field: np.ndarray
    relative_k_error: float


# --- per-chain sampling ------------------------------------------------------


def _scalar_table(chain: ChainSamples) -> dict[str, np.ndarray]:
    coefs = chain.coefficients
    return {
        "misfit": chain.misfit,
        "theta_norm": np.linalg.norm(coefs, axis=1),
        "theta_0": coefs[:, 0],
        "theta_1": coefs[:, 1],
        "theta_2": coefs[:, 2],
    }


def _block_gibbs_chain(
    cfg: Config,
    truth_draw: np.ndarray,
    noise_draw: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    chain_seed: int,
    n_sweeps: int,
    beta: float,
    tau2: float,
    scan_order: str,
    initial_scale: float,
) -> ChainSamples:
    n_sub = cfg.n_coarse_x * cfg.n_coarse_y
    rng = _TruthReplayGenerator(
        theta_true_draw=truth_draw,
        noise_draw=noise_draw,
        chain_seed=chain_seed,
        n_initial_draws=1,
        initial_scale=initial_scale,
    )
    states = block_gibbs_sampler(
        cfg=cfg,
        n_sweeps=n_sweeps,
        rng=rng,  # type: ignore[arg-type]
        beta=beta,
        block_nugget=tau2,
        scan_order=scan_order,
    )
    solves: list[int] = []
    misfit: list[float] = []
    fields: list[np.ndarray] = []
    started = perf_counter()
    updates = 0
    for state in states:
        updates += 1
        if state.block_index == n_sub - 1:  # one sample per completed sweep
            solves.append(1 + updates)  # 1 initial solve + per-update solves
            misfit.append(-state.log_likelihood_accepted)
            fields.append(state.G_accepted)
    wall = perf_counter() - started
    fields_arr = np.stack(fields)
    return ChainSamples(
        solves=np.asarray(solves, dtype=np.int64),
        fields=fields_arr,
        misfit=np.asarray(misfit, dtype=np.float64),
        coefficients=_project_coefficients(fields_arr, Phi, lam),
        wall_seconds=wall,
    )


def _global_pcn_chain(
    cfg: Config,
    truth: TruthData,
    Phi: np.ndarray,
    lam: np.ndarray,
    chain_seed: int,
    n_iter: int,
    beta: float,
    initial_scale: float,
    stride: int,
) -> ChainSamples:
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
    solves: list[int] = []
    misfit: list[float] = []
    fields: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    started = perf_counter()
    for idx, state in enumerate(states, start=1):
        if idx % stride == 0:  # sample at the block-Gibbs solve cadence
            solves.append(1 + idx)
            misfit.append(-(state.log_density - log_prior(state.theta)))
            fields.append(
                reshape_field(field_from_theta(Phi, lam, state.theta), cfg.ny, cfg.nx)
            )
            coefficients.append(np.asarray(state.theta, dtype=np.float64))
    wall = perf_counter() - started
    return ChainSamples(
        solves=np.asarray(solves, dtype=np.int64),
        fields=np.stack(fields),
        misfit=np.asarray(misfit, dtype=np.float64),
        coefficients=np.stack(coefficients),
        wall_seconds=wall,
    )


# --- convergence / efficiency reduction --------------------------------------


def _rhat_at(scalars: list[dict[str, np.ndarray]], upto: int) -> float:
    """Max over scalars of split-style R_hat on the second half of each prefix."""
    start = upto // 2
    worst = 1.0
    for name in SCALARS:
        block = np.stack([table[name][start:upto] for table in scalars])
        worst = max(worst, gelman_rubin(block))
    return worst


def _summarize_method(
    label: str,
    chains: tuple[ChainSamples, ...],
    truth: TruthData,
    n_checkpoints: int,
) -> MethodResult:
    n_samples = min(chain.solves.size for chain in chains)
    chains = tuple(
        ChainSamples(
            solves=chain.solves[:n_samples],
            fields=chain.fields[:n_samples],
            misfit=chain.misfit[:n_samples],
            coefficients=chain.coefficients[:n_samples],
            wall_seconds=chain.wall_seconds,
        )
        for chain in chains
    )
    scalars = [_scalar_table(chain) for chain in chains]

    checkpoints = sorted(
        {
            int(round(value))
            for value in np.geomspace(max(4, n_samples // 16), n_samples, n_checkpoints)
        }
    )
    rhat_solves = np.asarray(
        [int(chains[0].solves[upto - 1]) for upto in checkpoints], dtype=np.int64
    )
    rhat_curve = np.asarray([_rhat_at(scalars, upto) for upto in checkpoints])
    solves_to_threshold: int | None = None
    for solves, rhat in zip(rhat_solves, rhat_curve):
        if rhat <= R_HAT_THRESHOLD:
            solves_to_threshold = int(solves)
            break

    half = n_samples // 2
    total_ess = {
        name: float(
            np.sum([effective_sample_size(table[name][half:]) for table in scalars])
        )
        for name in SCALARS
    }
    conservative_total_ess = min(total_ess.values())
    total_forward_solves = int(sum(chain.solves[-1] for chain in chains))
    wall_seconds = float(sum(chain.wall_seconds for chain in chains))
    ess_per_1000_solves, ess_per_second = sampling_efficiency(
        conservative_total_ess, total_forward_solves, wall_seconds
    )

    pooled = np.concatenate([chain.fields[half:] for chain in chains], axis=0)
    posterior_mean_field = pooled.mean(axis=0)
    relative_k_error = relative_error(
        permeability_from_log_field(posterior_mean_field), truth.k_true
    )
    return MethodResult(
        label=label,
        chains=chains,
        rhat_solves=rhat_solves,
        rhat_curve=rhat_curve,
        best_rhat=float(np.min(rhat_curve)),
        endpoint_rhat=float(rhat_curve[-1]),
        solves_to_threshold=solves_to_threshold,
        conservative_total_ess=conservative_total_ess,
        total_forward_solves=total_forward_solves,
        wall_seconds=wall_seconds,
        ess_per_1000_solves=ess_per_1000_solves,
        ess_per_second=ess_per_second,
        posterior_mean_field=posterior_mean_field,
        relative_k_error=relative_k_error,
    )


# --- task 3: naive stitch + interface jump + figures -------------------------


def _naive_local_prior_draw(cfg: Config, n_ext: int, seed: int) -> np.ndarray:
    """Independent per-subdomain local-KLE prior draw, cores stitched (no coupling).

    Each subdomain draws its own local-KLE generative field from an independent
    seed and its core is copied in. With no cross-block conditioning the cores
    carry independent information, so the stitched field is discontinuous at the
    subdomain seams -- the faithful "no conditioning" baseline. It shares the
    generative-prior amplitude (sigma=1), so the interface-jump RMS is comparable
    to the posterior-mean fields.
    """
    rng = np.random.default_rng(seed)
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    out = np.zeros(cfg.nx * cfg.ny, dtype=np.float64)
    for row, col in iter_coarse_subdomains(cfg):
        sub = make_subdomain_at(cfg, row, col)
        local_pts = points[sub.local_global_idx, :]
        c_local = exp_covariance(local_pts, cfg.sigma, cfg.corr_length)
        n_keep = min(n_ext, sub.local_global_idx.size)
        phi_local, lam_local = top_eigenpairs(c_local, n_keep)
        eta = rng.standard_normal(n_keep)
        field_ext = phi_local @ (np.sqrt(lam_local) * eta)  # independent prior draw
        out[sub.core_global_idx] = field_ext[sub.core_local_idx]
    return reshape_field(out, cfg.ny, cfg.nx)


def _naive_local_projection(
    cfg: Config, truth_field: np.ndarray, n_ext: int
) -> np.ndarray:
    """Per-subdomain local-KLE projection of the truth, cores stitched (reference).

    Each core is the orthogonal projection of the truth (on that subdomain's
    extended region) onto its own local KLE subspace. Reported alongside the prior
    draw to show that a rich local basis tracking the same smooth truth is *not*
    discontinuous -- the seams come from independent information, not from local
    representation per se.
    """
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    truth_vec = truth_field.ravel(order="F")
    out = np.zeros_like(truth_vec)
    for row, col in iter_coarse_subdomains(cfg):
        sub = make_subdomain_at(cfg, row, col)
        local_pts = points[sub.local_global_idx, :]
        c_local = exp_covariance(local_pts, cfg.sigma, cfg.corr_length)
        n_keep = min(n_ext, sub.local_global_idx.size)
        phi_local, _ = top_eigenpairs(c_local, n_keep)
        field_ext = truth_vec[sub.local_global_idx]
        recon = phi_local @ (phi_local.T @ field_ext)  # projection onto local KLE
        out[sub.core_global_idx] = recon[sub.core_local_idx]
    return reshape_field(out, cfg.ny, cfg.nx)


def _mean_interface_jump(field: np.ndarray, cfg: Config) -> float:
    """Mean over all coarse cores of the per-core boundary interface-jump RMS."""
    return float(
        np.mean(
            [
                interface_jump_rms(field, make_subdomain_at(cfg, row, col))
                for row, col in iter_coarse_subdomains(cfg)
            ]
        )
    )


def _save_heatmap(
    field: np.ndarray, title: str, path: Path, vlim: tuple[float, float]
) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    image = ax.imshow(field, origin="lower", cmap="viridis", vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_rhat_curve(results: list[MethodResult], n_modes: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for result in results:
        ax.plot(
            result.rhat_solves, result.rhat_curve, marker="o", ms=3, label=result.label
        )
    ax.axhline(R_HAT_THRESHOLD, ls="--", lw=1.0, color="gray", label="R_hat=1.1")
    ax.set_xlabel("forward solves")
    ax.set_ylabel("max-scalar R_hat (2nd half)")
    ax.set_yscale("log")
    ax.set_title(f"Convergence speed, N={n_modes}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- printing ----------------------------------------------------------------


def _print_convergence_table(results: list[MethodResult]) -> None:
    print(
        "method            solves/chn end_Rhat best_Rhat solves@Rhat<=1.1 min_ESS "
        "ESS/1k ESS/sec wall_s relk(meanG)"
    )
    for result in results:
        threshold = (
            "not_reached"
            if result.solves_to_threshold is None
            else str(result.solves_to_threshold)
        )
        solves_per_chain = result.total_forward_solves // len(result.chains)
        print(
            f"{result.label:<17} {solves_per_chain:10d} {result.endpoint_rhat:8.3f} "
            f"{result.best_rhat:9.3f} {threshold:>16} "
            f"{result.conservative_total_ess:7.2f} {result.ess_per_1000_solves:6.3f} "
            f"{result.ess_per_second:7.4f} {result.wall_seconds:6.1f} "
            f"{result.relative_k_error:10.4f}"
        )


def _print_rhat_curve(results: list[MethodResult]) -> None:
    print("R_hat vs forward solves (max-scalar, second-half):")
    print("  solves      " + "".join(f"{result.label:>18}" for result in results))
    solves_axis = results[0].rhat_solves
    for idx, solves in enumerate(solves_axis):
        row = "".join(f"{result.rhat_curve[idx]:18.3f}" for result in results)
        print(f"  {int(solves):<10d}{row}")


def _acceleration_verdict(results_by_modes: dict[int, list[MethodResult]]) -> list[str]:
    lines: list[str] = []
    for n_modes, results in results_by_modes.items():
        by_label = {result.label: result for result in results}
        pcn = by_label["global_pcn"]
        best_block = min(
            (by_label["bg_systematic"], by_label["bg_red_black"]),
            key=lambda r: r.endpoint_rhat,
        )
        rb = by_label["bg_red_black"]
        systematic = by_label["bg_systematic"]
        converged = max(r.endpoint_rhat for r in results) <= R_HAT_THRESHOLD
        faster_solve = best_block.ess_per_1000_solves >= pcn.ess_per_1000_solves
        faster_second = best_block.ess_per_second >= pcn.ess_per_second
        rb_vs_sys = (
            "red-black mixes better than systematic"
            if rb.endpoint_rhat < systematic.endpoint_rhat - 0.05
            else "red-black is not better than systematic"
        )
        lines.append(
            f"N={n_modes}: best block = {best_block.label} (end R_hat "
            f"{best_block.endpoint_rhat:.2f}) vs global pCN (end R_hat "
            f"{pcn.endpoint_rhat:.2f}); ESS/1k-solves "
            f"{best_block.ess_per_1000_solves:.3f} vs {pcn.ess_per_1000_solves:.3f}, "
            f"ESS/sec {best_block.ess_per_second:.4f} vs {pcn.ess_per_second:.4f}. "
            + (
                "Block-Gibbs matches/beats global pCN per solve"
                if faster_solve
                else "Block-Gibbs does NOT beat global pCN per solve"
            )
            + " and "
            + ("per second" if faster_second else "trails per second")
            + f"; {rb_vs_sys}."
            + ("" if converged else " (R_hat>1.1: efficiency is indicative only.)")
        )
    lines.append(
        "Honest reading: under the dense KLE+nugget precision, red-black is only a "
        "sequential ordering (same-color cores are coupled), so a limited or absent "
        "speedup is expected; a genuine local-update acceleration needs a "
        "sparse/local prior with nearest-neighbour precision, not this dense KLE one."
    )
    return lines


# --- driver ------------------------------------------------------------------


def _run_modes(
    n_modes: int,
    args: argparse.Namespace,
    n_sweeps: int,
    outdir: Path,
) -> list[MethodResult]:
    cfg = Config(seed=args.seed, beta=args.beta, n_global_modes=n_modes)
    n_sub = cfg.n_coarse_x * cfg.n_coarse_y
    truth, theta_true_draw, noise_draw = _truth_replay_prefix(cfg)
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    covariance = exp_covariance(points, cfg.sigma, cfg.corr_length)
    Phi, lam = top_eigenpairs(covariance, cfg.n_global_modes)
    seeds = [args.seed + 1_000 + idx for idx in range(args.n_chains)]
    pcn_iter = n_sweeps * n_sub

    def block_chains(scan_order: str) -> tuple[ChainSamples, ...]:
        return tuple(
            _block_gibbs_chain(
                cfg,
                theta_true_draw,
                noise_draw,
                Phi,
                lam,
                seed,
                n_sweeps,
                args.beta,
                args.block_nugget,
                scan_order,
                args.initial_scale,
            )
            for seed in seeds
        )

    pcn_chains = tuple(
        _global_pcn_chain(
            cfg,
            truth,
            Phi,
            lam,
            seed + 5_000,
            pcn_iter,
            args.beta,
            args.initial_scale,
            n_sub,
        )
        for seed in seeds
    )
    results = [
        _summarize_method(
            "bg_systematic", block_chains("systematic"), truth, args.checkpoints
        ),
        _summarize_method(
            "bg_red_black", block_chains("red_black"), truth, args.checkpoints
        ),
        _summarize_method("global_pcn", pcn_chains, truth, args.checkpoints),
    ]

    print(f"\n=== N (n_global_modes) = {n_modes}; lam_N = {lam[-1]:.3f} ===")
    print(
        f"chains={args.n_chains}; sweeps={n_sweeps} (={pcn_iter} pcn-iters); "
        f"beta={args.beta}; tau2={args.block_nugget:g}; initial_scale={args.initial_scale}"
    )
    _print_convergence_table(results)
    print()
    _print_rhat_curve(results)
    _save_rhat_curve(results, n_modes, outdir / f"exp09_rhat_vs_solves_N{n_modes}.png")

    # Task 3: heatmaps + interface-jump comparison.
    n_ext = cfg.Nc + args.mb
    vlim = (float(truth.G_true.min()), float(truth.G_true.max()))
    _save_heatmap(
        truth.G_true, "truth log-k", outdir / f"exp09_field_truth_N{n_modes}.png", vlim
    )
    naive_draw = _naive_local_prior_draw(cfg, n_ext, seed=args.seed + n_modes)
    naive_projection = _naive_local_projection(cfg, truth.G_true, n_ext)
    _save_heatmap(
        naive_draw,
        "naive independent local-KLE draw",
        outdir / f"exp09_field_naive_draw_N{n_modes}.png",
        vlim,
    )
    jump_rows = [
        ("naive_local_prior_draw", _mean_interface_jump(naive_draw, cfg)),
        ("naive_local_projection", _mean_interface_jump(naive_projection, cfg)),
    ]
    for result in results:
        _save_heatmap(
            result.posterior_mean_field,
            f"{result.label} posterior mean",
            outdir / f"exp09_field_{result.label}_N{n_modes}.png",
            vlim,
        )
        jump_rows.append(
            (result.label, _mean_interface_jump(result.posterior_mean_field, cfg))
        )
    jump_rows.append(("truth", _mean_interface_jump(truth.G_true, cfg)))

    print(
        "\nInterface-jump RMS (mean over coarse cores; lower = smoother seams; "
        "naive draw = no conditioning):"
    )
    for name, value in jump_rows:
        print(f"  {name:<26} {value:.4f}")
    print(f"heatmaps + R_hat curve written to {outdir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", action="store_true", help="Opt-in long comparison.")
    parser.add_argument("--modes", type=str, default="64,90")
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=None)
    parser.add_argument("--checkpoints", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--block-nugget", type=float, default=1e-2)
    parser.add_argument("--initial-scale", type=float, default=2.0)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    if args.n_chains < 2:
        raise ValueError("n-chains must be at least 2 for R_hat.")
    if args.block_nugget <= 0.0:
        raise ValueError("block-nugget must be positive.")
    modes = tuple(int(value) for value in args.modes.split(","))
    n_sweeps = args.sweeps if args.sweeps is not None else (300 if args.long else 8)
    outdir = Path(args.outdir) if args.outdir else (ROOT / "outputs" / "exp09")
    outdir.mkdir(parents=True, exist_ok=True)

    print("M11 ACCELERATION STUDY: block-Gibbs (systematic / red-black) vs global pCN")
    print(
        f"profile={'long' if args.long else 'responsive'}; modes={modes}; outdir={outdir}"
    )

    results_by_modes = {
        n_modes: _run_modes(n_modes, args, n_sweeps, outdir) for n_modes in modes
    }
    print("\nACCELERATION VERDICT")
    for line in _acceleration_verdict(results_by_modes):
        print(f"  {line}")


if __name__ == "__main__":
    main()
