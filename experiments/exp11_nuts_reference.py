"""M13 NUTS gold-standard reference: the converged posterior the accelerators
are judged against.

The M9 convergence experiments stalled -- nothing reached R_hat <~ 1.1, so the
posterior-mean rel-k "floor" (~0.5-0.8) and interval coverage could not be
attributed: is 0.6 the true posterior floor, or an unconverged chain? This
experiment removes the ambiguity with a hand-rolled NUTS sampler (gradient-based
Hamiltonian dynamics on the whitened potential U(theta) = 0.5||theta||^2 +
misfit) run to tight convergence (target R_hat <= 1.01) at N in {64, 90}. It
reports the reference posterior-mean rel-k floor, 90% credible-interval coverage,
the data-misfit floor (~N_obs/2), R_hat and ESS -- the numbers block-Gibbs,
global pCN and LIS are benchmarked against.

Preconditioning (exactness-preserving). The sigma_obs = 0.01 likelihood makes the
posterior highly anisotropic (Laplace precision I+H has condition ~1e3-1e4), so
identity-metric NUTS collapses to a tiny step size and maxes out the tree depth.
We therefore sample in whitened coordinates phi with theta = C phi, where
C = chol((I+H)^{-1}) is the Laplace-covariance factor from a cheap adjoint
Gauss-Newton Hessian at the MAP. A linear reparametrisation (equivalently a fixed
dense mass matrix M = I+H) leaves the *target distribution exactly unchanged* --
it is a preconditioner, not an approximation -- so the sampler remains a
gold-standard: theta = C phi are exact posterior draws. The one-time MAP + Hessian
cost is charged to the solve budget (as in exp10 LIS). Pass ``--precondition
none`` for the raw identity-metric sampler (correct but impractically slow here).

Role is trust, not speed: one forward + one adjoint solve per gradient
(``--gradient single_solve``, the default) makes the per-solve budget honest.

    python -m experiments.exp11_nuts_reference --setup     --modes 64,90
    python -m experiments.exp11_nuts_reference --run --long --modes 64,90
    python -m experiments.exp11_nuts_reference --aggregate  --modes 64,90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale import lis  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.diagnostics import (  # noqa: E402
    credible_interval_coverage,
    effective_sample_size,
    gelman_rubin,
    relative_error,
)
from mcmc_multiscale.field import (  # noqa: E402
    field_from_theta,
    permeability_from_log_field,
    reshape_field,
)
from mcmc_multiscale.forward import ForwardModel  # noqa: E402
from mcmc_multiscale.grid import cell_centered_grid  # noqa: E402
from mcmc_multiscale.kle import top_eigenpairs  # noqa: E402
from mcmc_multiscale.nuts import make_potential, nuts_sample  # noqa: E402
from mcmc_multiscale.observations import make_truth  # noqa: E402

SCALARS = ("misfit", "theta_norm", "theta_0", "theta_1", "theta_2")
R_HAT_THRESHOLD = 1.01
DEFAULT_OUT = ROOT / "outputs" / "exp11"


# --------------------------------------------------------------------------- #
# shared setup (cached per N): KLE, truth, Laplace preconditioner
# --------------------------------------------------------------------------- #
def _kle(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, _, pts = cell_centered_grid(cfg.nx, cfg.ny)
    return top_eigenpairs(
        exp_covariance(pts, cfg.sigma, cfg.corr_length), cfg.n_global_modes
    )


def setup(n_modes: int, outdir: Path) -> None:
    cfg = Config(n_global_modes=n_modes)
    truth = make_truth(cfg)
    Phi, lam = _kle(cfg)
    # Laplace preconditioner at the MAP: C = chol((I + H)^{-1}), H = J^T J / s2.
    # This whitens the near-Gaussian posterior; theta = C phi is exact (a fixed
    # linear reparametrisation), so the sampler stays a gold-standard reference.
    t0 = perf_counter()
    theta_map, map_solves = lis.gauss_newton_map(
        cfg, Phi, lam, truth.sensor_idx, truth.y_obs
    )
    J = lis.darcy_jacobian_adjoint(cfg, Phi, lam, theta_map, truth.sensor_idx)
    H = lis.gn_hessian(J, cfg.sigma_obs)
    setup_wall = perf_counter() - t0
    setup_solves = int(map_solves + 1 + truth.sensor_idx.size)
    Sigma = np.linalg.inv(np.eye(n_modes) + H)
    precond_C = np.linalg.cholesky(Sigma)
    mu = np.linalg.eigvalsh(np.eye(n_modes) + H)
    np.savez(
        outdir / f"setup_N{n_modes}.npz",
        Phi=Phi,
        lam=lam,
        y_obs=truth.y_obs,
        sensor_idx=truth.sensor_idx,
        k_true=truth.k_true,
        G_true=truth.G_true,
        theta_true=truth.theta_true,
        theta_map=theta_map,
        precond_C=precond_C,
        setup_solves=setup_solves,
        setup_wall=setup_wall,
    )
    print(
        f"[setup N={n_modes}] truth + {n_modes}-mode KLE; MAP+Hessian in "
        f"{setup_wall:.1f}s ({setup_solves} solves); Laplace (I+H) cond="
        f"{mu.max() / mu.min():.1e}; noise floor N_obs/2="
        f"{0.5 * cfg.n_obs_x * cfg.n_obs_y:.1f}"
    )


def _load_setup(n_modes: int, outdir: Path) -> dict:
    path = outdir / f"setup_N{n_modes}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --setup first.")
    return dict(np.load(path))


# --------------------------------------------------------------------------- #
# run: n_chains NUTS chains from matched over-dispersed starts
# --------------------------------------------------------------------------- #
def run(
    n_modes: int,
    n_iter: int,
    n_warmup: int,
    n_chains: int,
    outdir: Path,
    gradient: str,
    precondition: str,
    target_accept: float,
    max_tree_depth: int,
    initial_scale: float,
    seed: int,
) -> None:
    cfg = Config(n_global_modes=n_modes)
    S = _load_setup(n_modes, outdir)
    Phi, lam, y, sidx = S["Phi"], S["lam"], S["y_obs"], S["sensor_idx"]
    fwd = ForwardModel(cfg)
    n_obs = int(sidx.size)
    solves_per_grad = 2 if gradient == "single_solve" else n_obs + 2

    _, _, vg_theta = make_potential(cfg, Phi, lam, sidx, y, fwd, gradient=gradient)

    use_pc = precondition == "laplace"
    if use_pc:
        C = np.asarray(S["precond_C"], dtype=np.float64)
        phi_map = np.linalg.solve(C, np.asarray(S["theta_map"], dtype=np.float64))

        def value_and_grad(phi: np.ndarray) -> tuple[float, np.ndarray]:
            u, g = vg_theta(C @ phi)
            return u, C.T @ g

        setup_solves = int(S["setup_solves"])
        setup_wall = float(S["setup_wall"])
    else:
        C = None
        value_and_grad = vg_theta
        setup_solves = 0
        setup_wall = 0.0

    seeds = [seed + 100 + i for i in range(n_chains)]
    coeffs = np.empty((n_chains, n_iter, n_modes), dtype=np.float64)
    misfit = np.empty((n_chains, n_iter), dtype=np.float64)
    leapfrogs = np.empty((n_chains, n_iter), dtype=np.int64)
    tree_depth = np.empty((n_chains, n_iter), dtype=np.int64)
    accept = np.empty((n_chains, n_iter), dtype=np.float64)
    step_size = np.empty(n_chains, dtype=np.float64)
    divergences = np.zeros(n_chains, dtype=np.int64)
    warmup_solves = np.zeros(n_chains, dtype=np.int64)

    t0 = perf_counter()
    for c, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        z0 = np.random.default_rng(sd + 7).standard_normal(n_modes)
        # over-dispersed start: in whitened coords the posterior is ~N(phi_map, I),
        # so phi_map + scale*z is dispersed by `scale` posterior-stds (R_hat is
        # meaningful) while staying in-basin. Raw coords disperse at prior scale.
        start = (phi_map + initial_scale * z0) if use_pc else initial_scale * z0
        stats: dict[str, int] = {}
        for i, st in enumerate(
            nuts_sample(
                value_and_grad,
                start,
                n_iter,
                rng,
                n_warmup=n_warmup,
                target_accept=target_accept,
                max_tree_depth=max_tree_depth,
                stats=stats,
            )
        ):
            theta = C @ st.theta if use_pc else st.theta
            coeffs[c, i] = theta
            misfit[c, i] = st.potential - 0.5 * float(theta @ theta)
            leapfrogs[c, i] = st.n_leapfrog
            tree_depth[c, i] = st.tree_depth
            accept[c, i] = st.accept_prob
            divergences[c] += int(st.diverged)
            step_size[c] = st.step_size
        warmup_solves[c] = solves_per_grad * (
            stats.get("init_grads", 1)
            + stats.get("find_eps_leapfrogs", 0)
            + stats.get("warmup_leapfrogs", 0)
        )
    sample_wall = perf_counter() - t0

    out = outdir / f"chains_nuts_N{n_modes}.npz"
    np.savez(
        out,
        coeffs=coeffs,
        misfit=misfit,
        leapfrogs=leapfrogs,
        tree_depth=tree_depth,
        accept=accept,
        step_size=step_size,
        divergences=divergences,
        warmup_solves=warmup_solves,
        setup_solves=setup_solves,
        setup_wall=setup_wall,
        sample_wall=sample_wall,
        solves_per_grad=solves_per_grad,
        n_chains=n_chains,
        n_iter=n_iter,
        n_warmup=n_warmup,
        gradient=gradient,
        precondition=precondition,
    )
    print(
        f"[run nuts N={n_modes}] chains={n_chains} iters={n_iter} warmup={n_warmup} "
        f"grad={gradient} precond={precondition} accept={np.round(accept.mean(axis=1), 2)} "
        f"depth~{tree_depth.mean():.1f} div={divergences.sum()} "
        f"sample_wall={sample_wall:.1f}s -> {out.name}"
    )


# --------------------------------------------------------------------------- #
# aggregation: R_hat-vs-solves, ESS, rel-k floor, coverage, misfit floor
# --------------------------------------------------------------------------- #
def _scalar_tables(
    coeffs: np.ndarray, misfit: np.ndarray
) -> list[dict[str, np.ndarray]]:
    out = []
    for c in range(coeffs.shape[0]):
        out.append(
            {
                "misfit": misfit[c],
                "theta_norm": np.linalg.norm(coeffs[c], axis=1),
                "theta_0": coeffs[c, :, 0],
                "theta_1": coeffs[c, :, 1],
                "theta_2": coeffs[c, :, 2],
            }
        )
    return out


def _rhat_at(tables: list[dict[str, np.ndarray]], upto: int) -> float:
    start = upto // 2
    worst = 1.0
    for name in SCALARS:
        block = np.stack([t[name][start:upto] for t in tables])
        worst = max(worst, gelman_rubin(block))
    return worst


def aggregate(n_modes: int, outdir: Path, n_checkpoints: int = 14) -> dict:
    path = outdir / f"chains_nuts_N{n_modes}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --run first.")
    data = dict(np.load(path))
    S = _load_setup(n_modes, outdir)
    Phi, lam, k_true, G_true = S["Phi"], S["lam"], S["k_true"], S["G_true"]
    cfg = Config(n_global_modes=n_modes)
    n_obs = cfg.n_obs_x * cfg.n_obs_y
    noise_floor = 0.5 * n_obs

    coeffs, misfit, leapfrogs = data["coeffs"], data["misfit"], data["leapfrogs"]
    n_iter = int(data["n_iter"])
    solves_per_grad = int(data["solves_per_grad"])
    setup_solves = int(data.get("setup_solves", 0))
    warmup_solves = int(np.sum(data["warmup_solves"]))
    fixed_solves = setup_solves + warmup_solves  # one-time cost before sampling
    tables = _scalar_tables(coeffs, misfit)

    checkpoints = sorted(
        {
            int(round(v))
            for v in np.geomspace(max(4, n_iter // 32), n_iter, n_checkpoints)
        }
    )
    # variable leapfrogs/iter: cumulative sampling solves = solves_per_grad *
    # sum over chains of leapfrogs up to the checkpoint, plus one-time setup+warmup.
    cum_lf = np.cumsum(leapfrogs, axis=1)  # (n_chains, n_iter)
    total_solves = np.array(
        [
            fixed_solves + solves_per_grad * int(cum_lf[:, cp - 1].sum())
            for cp in checkpoints
        ],
        dtype=np.int64,
    )
    rhat = np.array([_rhat_at(tables, cp) for cp in checkpoints])
    solves_to_thr = None
    for s, r in zip(total_solves, rhat):
        if r <= R_HAT_THRESHOLD:
            solves_to_thr = int(s)
            break

    half = n_iter // 2
    total_ess = min(
        float(np.sum([effective_sample_size(t[name][half:]) for t in tables]))
        for name in SCALARS
    )
    budget_solves = fixed_solves + solves_per_grad * int(cum_lf[:, -1].sum())

    # posterior recovery (second-half pooled)
    pooled = coeffs[:, half:, :].reshape(-1, n_modes)
    mean_theta = pooled.mean(axis=0)
    Gmean = reshape_field(field_from_theta(Phi, lam, mean_theta), cfg.ny, cfg.nx)
    k_mean = permeability_from_log_field(Gmean)
    relk = relative_error(k_mean, k_true)

    # 90% credible-interval coverage of G_true (subsample the pooled field draws)
    stride = max(1, pooled.shape[0] // 800)
    fields = np.stack(
        [
            reshape_field(field_from_theta(Phi, lam, th), cfg.ny, cfg.nx)
            for th in pooled[::stride]
        ]
    )
    coverage = credible_interval_coverage(fields, G_true, level=0.9)

    misfit_floor = float(misfit[:, half:].mean())
    summary = {
        "n_modes": n_modes,
        "endpoint_rhat": float(rhat[-1]),
        "best_rhat": float(rhat.min()),
        "solves_to_threshold": solves_to_thr,
        "total_ess": total_ess,
        "ess_per_1k": 1000.0 * total_ess / budget_solves,
        "budget_solves": budget_solves,
        "setup_solves": setup_solves,
        "warmup_solves": warmup_solves,
        "accept": float(data["accept"].mean()),
        "mean_tree_depth": float(data["tree_depth"].mean()),
        "divergences": int(np.sum(data["divergences"])),
        "relk": relk,
        "coverage": coverage,
        "misfit_floor": misfit_floor,
        "noise_floor": noise_floor,
        "total_solves": total_solves,
        "rhat": rhat,
    }

    thr = "not_reached" if solves_to_thr is None else str(solves_to_thr)
    print(f"\n=== N = {n_modes} ===  (target R_hat <= {R_HAT_THRESHOLD})")
    print(
        f"  end R_hat        : {summary['endpoint_rhat']:.4f}  (best {summary['best_rhat']:.4f})"
    )
    print(
        f"  solves @ R_hat<=1.01: {thr}  (budget {budget_solves}, setup {setup_solves}, warmup {warmup_solves})"
    )
    print(
        f"  accept / depth   : {summary['accept']:.3f} / {summary['mean_tree_depth']:.2f}  "
        f"divergences={summary['divergences']}"
    )
    print(
        f"  ESS (min scalar) : {total_ess:.0f}  ({summary['ess_per_1k']:.2f} / 1k solves)"
    )
    print(f"  rel-k floor      : {relk:.4f}")
    print(f"  90% coverage     : {coverage:.3f}")
    print(
        f"  misfit floor     : {misfit_floor:.2f}  (noise floor N_obs/2 = {noise_floor:.1f})"
    )

    _plot_rhat(summary, outdir / f"exp11_rhat_vs_solves_N{n_modes}.png")
    _plot_misfit(
        data, noise_floor, outdir / f"exp11_misfit_trace_N{n_modes}.png", n_modes
    )
    _plot_fields(S["k_true"], k_mean, outdir / f"exp11_fields_N{n_modes}.png", n_modes)
    return summary


# --------------------------------------------------------------------------- #
def _plot_rhat(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(summary["total_solves"], summary["rhat"], marker="o", ms=3, label="NUTS")
    ax.axhline(
        R_HAT_THRESHOLD, ls="--", lw=1, color="gray", label=f"R_hat={R_HAT_THRESHOLD}"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total forward solves (all chains, incl. setup+warmup)")
    ax.set_ylabel("max-scalar R_hat (2nd half)")
    ax.set_title(f"NUTS convergence, N={summary['n_modes']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path.name}")


def _plot_misfit(data: dict, noise_floor: float, path: Path, n_modes: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    misfit = data["misfit"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for c in range(misfit.shape[0]):
        ax.plot(misfit[c], lw=0.8, alpha=0.8, label=f"chain {c}")
    ax.axhline(
        noise_floor, ls="--", lw=1, color="k", label=f"N_obs/2={noise_floor:.0f}"
    )
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("data misfit")
    ax.set_title(f"Data-misfit trace, N={n_modes}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path.name}")


def _plot_fields(
    k_true: np.ndarray, k_mean: np.ndarray, path: Path, n_modes: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = float(min(k_true.min(), k_mean.min()))
    vmax = float(max(k_true.max(), k_mean.max()))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, field, title in (
        (axes[0], k_true, "truth k"),
        (axes[1], k_mean, "posterior-mean k"),
    ):
        im = ax.imshow(field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}, N={n_modes}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  figure -> {path.name}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--modes", type=str, default="64,90")
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument(
        "--gradient",
        type=str,
        default="single_solve",
        choices=["single_solve", "full_jacobian"],
    )
    p.add_argument(
        "--precondition", type=str, default="laplace", choices=["laplace", "none"]
    )
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--initial-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--quick", action="store_true", help="fast smoke run (default)")
    p.add_argument("--long", action="store_true", help="full convergence run")
    p.add_argument("--iters", type=int, default=None, help="override sampling iters")
    p.add_argument("--warmup", type=int, default=None, help="override warmup iters")
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    if args.n_chains < 2:
        raise SystemExit("--n-chains must be >= 2 for Gelman-Rubin R_hat.")

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    modes = [int(m) for m in args.modes.split(",")]

    n_iter = args.iters if args.iters is not None else (2000 if args.long else 300)
    n_warmup = args.warmup if args.warmup is not None else (1000 if args.long else 150)

    if args.setup:
        for n in modes:
            setup(n, outdir)
        return
    if args.run:
        for n in modes:
            run(
                n,
                n_iter,
                n_warmup,
                args.n_chains,
                outdir,
                gradient=args.gradient,
                precondition=args.precondition,
                target_accept=args.target_accept,
                max_tree_depth=args.max_tree_depth,
                initial_scale=args.initial_scale,
                seed=args.seed,
            )
        return
    if args.aggregate:
        for n in modes:
            aggregate(n, outdir)
        return
    raise SystemExit("pass --setup, --run, or --aggregate")


if __name__ == "__main__":
    main()
