"""M12 acceleration study: Likelihood-Informed-Subspace (LIS) pCN vs global pCN.

Question (the open problem in the project brief): at modest stochastic dimension
(N in {64, 90}) can a *correct* sampler beat global pCN per forward solve? Block-
Gibbs on the dense KLE prior could not -- spatial blocks fight the prior's
correlations and cost a full solve per tiny move, and they do not touch pCN's
real bottleneck: the low-dimensional **likelihood-informed** directions, where the
posterior is far narrower than the prior so prior-scale pCN steps overshoot.

LIS attacks exactly that bottleneck. It blocks the KLE coordinates by *spectrum*,
not space: an informed subspace (top eigenvectors of the prior-preconditioned
Gauss-Newton Hessian, built once from a cheap Darcy adjoint at the MAP) is sampled
by pCN about the Laplace approximation, while the orthogonal complement gets
ordinary pCN. Because orthogonal subspaces of the isotropic prior are independent,
the construction is exactly posterior-invariant (gates in tests/test_lis.py) at
one forward solve per step.

Compute-fair accounting: the one-time setup (MAP + adjoint Hessian) is charged to
the LIS solve budget. Headline metric: total forward solves (and seconds) to reach
Gelman-Rubin R_hat <= 1.1, plus ESS / 1000 solves and ESS / second. All chains use
matched over-dispersed starts.

Runs one (method, N) per invocation (kept under a short wall budget) and caches the
shared setup; `--aggregate` then builds the R_hat curves, tables and figures.

    python -m experiments.exp10_lis_acceleration --setup --modes 64,90
    python -m experiments.exp10_lis_acceleration --method global_pcn --modes 64 --iters 1500
    python -m experiments.exp10_lis_acceleration --method lis_adjoint --modes 64 --iters 600
    python -m experiments.exp10_lis_acceleration --aggregate --modes 64,90
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

from mcmc_multiscale.bayes import log_posterior, log_prior  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.diagnostics import (  # noqa: E402
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
from mcmc_multiscale.mcmc import metropolis_hastings  # noqa: E402
from mcmc_multiscale.observations import make_truth  # noqa: E402
from mcmc_multiscale.proposals import make_pcn_proposal  # noqa: E402
from mcmc_multiscale import lis  # noqa: E402

SCALARS = ("misfit", "theta_norm", "theta_0", "theta_1", "theta_2")
R_HAT_THRESHOLD = 1.1
DEFAULT_OUT = ROOT / "outputs" / "exp10"


# --------------------------------------------------------------------------- #
# shared setup (cached per N): KLE, truth, MAP, adjoint informed subspace
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
    t0 = perf_counter()
    theta_map, map_solves = lis.gauss_newton_map(
        cfg, Phi, lam, truth.sensor_idx, truth.y_obs
    )
    sub, sub_solves = lis.build_informed_subspace_adjoint(
        cfg, Phi, lam, theta_map, truth.sensor_idx, mu_threshold=1.0
    )
    setup_wall = perf_counter() - t0
    # full eigenbasis (all modes) so the ablation can pick any rank from cache
    H = lis.gn_hessian(
        lis.darcy_jacobian_adjoint(cfg, Phi, lam, theta_map, truth.sensor_idx),
        cfg.sigma_obs,
    )
    mu_all, V = np.linalg.eigh(H)
    order = np.argsort(mu_all)[::-1]
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
        U=V[:, order],
        mu=np.clip(mu_all[order], 0.0, None),
        setup_solves=map_solves + sub_solves,
        setup_wall=setup_wall,
    )
    print(
        f"[setup N={n_modes}] MAP misfit-region reached; informed rank(mu>1)={sub.rank}; "
        f"setup_solves={map_solves + sub_solves} (MAP {map_solves} + Hessian {sub_solves}); "
        f"setup_wall={setup_wall:.1f}s; post-std top6="
        f"{np.round(np.sqrt(sub.post_var[:6]), 3)}"
    )


def _load_setup(n_modes: int, outdir: Path) -> dict:
    path = outdir / f"setup_N{n_modes}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --setup first.")
    return dict(np.load(path))


# --------------------------------------------------------------------------- #
# one method, n_chains chains, matched dispersed starts
# --------------------------------------------------------------------------- #
def _make_proposal(
    method: str, S: dict, cfg: Config, rank: int, b_inf: float, b_comp: float, pilot
):
    if method == "global_pcn":
        return make_pcn_proposal(cfg.beta if b_comp is None else b_comp)
    if method == "lis_adjoint":
        sub = lis.InformedSubspace(
            U=S["U"][:, :rank],
            mean=S["theta_map"],
            post_var=1.0 / (1.0 + S["mu"][:rank]),
            eig=S["mu"][:rank],
            source="hessian",
        )
        return lis.make_lis_proposal(sub, b_inf, b_comp)
    if method == "lis_pilot":
        sub = lis.informed_subspace_from_samples(pilot, rank=rank)
        return lis.make_lis_proposal(sub, b_inf, b_comp)
    raise ValueError(method)


def run_method(
    method: str,
    n_modes: int,
    n_iter: int,
    n_chains: int,
    outdir: Path,
    rank: int,
    beta: float,
    beta_informed: float,
    beta_complement: float,
    initial_scale: float,
    seed: int,
    tag: str,
    pilot_iter: int,
) -> None:
    cfg = Config(n_global_modes=n_modes, beta=beta)
    S = _load_setup(n_modes, outdir)
    Phi, lam, y, sidx = S["Phi"], S["lam"], S["y_obs"], S["sensor_idx"]
    fwd = ForwardModel(cfg)

    def logpost(th):
        return log_posterior(th, Phi, lam, fwd, y, sidx, cfg.sigma_obs, cfg.ny, cfg.nx)

    setup_solves = 0
    setup_wall = 0.0
    pilot = None
    if method == "lis_adjoint":
        setup_solves = int(S["setup_solves"])
        setup_wall = float(S["setup_wall"])
    if (
        method == "lis_pilot"
    ):  # gradient-free: a global-pCN pilot estimates the subspace
        t0 = perf_counter()
        rng = np.random.default_rng(seed + 999)
        th = initial_scale * rng.standard_normal(n_modes)
        buf = []
        for st in metropolis_hastings(
            logpost,
            make_pcn_proposal(0.05),
            th,
            pilot_iter,
            rng,
            log_prior_fn=log_prior,
        ):
            buf.append(st.theta.copy())
        pilot = np.asarray(buf)[pilot_iter // 2 :]
        setup_solves = pilot_iter
        setup_wall = perf_counter() - t0

    seeds = [seed + 100 + i for i in range(n_chains)]
    coeffs = np.empty((n_chains, n_iter, n_modes), dtype=np.float64)
    misfit = np.empty((n_chains, n_iter), dtype=np.float64)
    accept = np.empty(n_chains)
    t0 = perf_counter()
    # Chains start from draws of the Laplace posterior N(MAP, (I+H)^{-1}): the LIS
    # proposal is a *local* sampler (tight reference), so an over-dispersed prior-scale
    # start makes pi/q* explode and the chain sticks. Laplace draws are over-dispersed
    # relative to the true posterior (R_hat is meaningful) yet in-basin. pCN gets the
    # same start (only helps it -> conservative for LIS). MAP/Hessian cost is charged.
    sqrt_post_var = 1.0 / np.sqrt(1.0 + S["mu"])
    for c, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        z0 = np.random.default_rng(sd + 7).standard_normal(n_modes)
        theta0 = S["theta_map"] + (S["U"] * sqrt_post_var) @ (S["U"].T @ z0)
        proposal = _make_proposal(
            method, S, cfg, rank, beta_informed, beta_complement, pilot
        )
        acc = 0
        for i, st in enumerate(
            metropolis_hastings(
                logpost, proposal, theta0, n_iter, rng, log_prior_fn=log_prior
            )
        ):
            coeffs[c, i] = st.theta
            misfit[c, i] = -(st.log_density - log_prior(st.theta))
            acc += st.accepted
        accept[c] = acc / n_iter
    sample_wall = perf_counter() - t0

    out = outdir / f"chains_{method}_{tag}_N{n_modes}.npz"
    np.savez(
        out,
        coeffs=coeffs,
        misfit=misfit,
        accept=accept,
        setup_solves=setup_solves,
        setup_wall=setup_wall,
        sample_wall=sample_wall,
        n_chains=n_chains,
        n_iter=n_iter,
        label=f"{method}_{tag}",
    )
    print(
        f"[run {method}_{tag} N={n_modes}] chains={n_chains} iters={n_iter} "
        f"accept={np.round(accept, 2)} setup_solves={setup_solves} "
        f"sample_wall={sample_wall:.1f}s -> {out.name}"
    )


# --------------------------------------------------------------------------- #
# aggregation: R_hat-vs-solves, ESS/solve, ESS/sec, field error, figures
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


def _summarize(data: dict, n_checkpoints: int = 14) -> dict:
    coeffs, misfit = data["coeffs"], data["misfit"]
    n_chains, n_iter = int(data["n_chains"]), int(data["n_iter"])
    setup_solves = int(data["setup_solves"])
    tables = _scalar_tables(coeffs, misfit)
    checkpoints = sorted(
        {
            int(round(v))
            for v in np.geomspace(max(4, n_iter // 32), n_iter, n_checkpoints)
        }
    )
    # total forward solves across all chains incl. one-time setup
    total_solves = np.array(
        [setup_solves + n_chains * cp for cp in checkpoints], dtype=np.int64
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
    budget_solves = setup_solves + n_chains * n_iter
    wall = float(data["setup_wall"]) + float(data["sample_wall"])
    return {
        "label": str(data["label"]),
        "checkpoints": np.array(checkpoints),
        "total_solves": total_solves,
        "rhat": rhat,
        "endpoint_rhat": float(rhat[-1]),
        "best_rhat": float(rhat.min()),
        "solves_to_threshold": solves_to_thr,
        "total_ess": total_ess,
        "ess_per_1k": 1000.0 * total_ess / budget_solves,
        "ess_per_sec": total_ess / wall,
        "budget_solves": budget_solves,
        "wall": wall,
        "accept": float(np.mean(data["accept"])),
        "mean_theta": coeffs[:, half:, :].reshape(-1, coeffs.shape[2]).mean(0),
    }


def aggregate(n_modes: int, outdir: Path) -> list[dict]:
    files = sorted(outdir.glob(f"chains_*_N{n_modes}.npz"))
    if not files:
        raise SystemExit(f"no chains_*_N{n_modes}.npz in {outdir}; run methods first.")
    S = _load_setup(n_modes, outdir)
    Phi, lam, k_true = S["Phi"], S["lam"], S["k_true"]
    cfg = Config(n_global_modes=n_modes)
    order = ["global_pcn", "lis_pilot", "lis_adjoint"]
    summaries = [_summarize(dict(np.load(f))) for f in files]
    summaries.sort(
        key=lambda s: (
            next((i for i, p in enumerate(order) if s["label"].startswith(p)), 9),
            s["label"],
        )
    )

    print(
        f"\n=== N = {n_modes} ===  (R_hat threshold {R_HAT_THRESHOLD}; setup charged to LIS budget)"
    )
    print(
        f"{'method':<20}{'accept':>7}{'end_Rhat':>9}{'best_Rhat':>10}"
        f"{'solves@1.1':>12}{'ESS/1k':>8}{'ESS/sec':>9}{'wall_s':>8}{'relk(meanG)':>12}"
    )
    for s in summaries:
        Gmean = reshape_field(
            field_from_theta(Phi, lam, s["mean_theta"]), cfg.ny, cfg.nx
        )
        relk = relative_error(permeability_from_log_field(Gmean), k_true)
        thr = (
            "not_reached"
            if s["solves_to_threshold"] is None
            else str(s["solves_to_threshold"])
        )
        print(
            f"{s['label']:<20}{s['accept']:>7.2f}{s['endpoint_rhat']:>9.2f}{s['best_rhat']:>10.2f}"
            f"{thr:>12}{s['ess_per_1k']:>8.2f}{s['ess_per_sec']:>9.3f}{s['wall']:>8.1f}{relk:>12.4f}"
        )
        s["relk"] = relk

    _plot_rhat(summaries, n_modes, outdir / f"exp10_rhat_vs_solves_N{n_modes}.png")
    return summaries


def _plot_rhat(summaries: list[dict], n_modes: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for s in summaries:
        ax.plot(s["total_solves"], s["rhat"], marker="o", ms=3, label=s["label"])
    ax.axhline(
        R_HAT_THRESHOLD, ls="--", lw=1, color="gray", label=f"R_hat={R_HAT_THRESHOLD}"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total forward solves (all chains, incl. LIS setup)")
    ax.set_ylabel("max-scalar R_hat (2nd half)")
    ax.set_title(f"Convergence speed, N={n_modes}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["global_pcn", "lis_adjoint", "lis_pilot"],
    )
    p.add_argument("--modes", type=str, default="64,90")
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument(
        "--rank", type=int, default=0, help="0 => auto (mu>1) for adjoint, 30 for pilot"
    )
    p.add_argument("--beta", type=float, default=0.05, help="global pCN step")
    p.add_argument("--beta-informed", type=float, default=0.6)
    p.add_argument("--beta-complement", type=float, default=0.45)
    p.add_argument("--initial-scale", type=float, default=2.0)
    p.add_argument("--pilot-iter", type=int, default=800)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    modes = [int(m) for m in args.modes.split(",")]

    if args.setup:
        for n in modes:
            setup(n, outdir)
        return
    if args.aggregate:
        for n in modes:
            aggregate(n, outdir)
        return
    if args.method is None:
        raise SystemExit("pass --setup, --method, or --aggregate")

    for n in modes:
        S = _load_setup(n, outdir)
        rank = args.rank
        if rank == 0:
            rank = int((S["mu"] > 1.0).sum()) if args.method == "lis_adjoint" else 30
        tag = args.tag or (
            f"r{rank}" if args.method != "global_pcn" else f"b{args.beta}"
        )
        run_method(
            args.method,
            n,
            args.iters,
            args.n_chains,
            outdir,
            rank=rank,
            beta=args.beta,
            beta_informed=args.beta_informed,
            beta_complement=args.beta_complement,
            initial_scale=args.initial_scale,
            seed=args.seed,
            tag=tag,
            pilot_iter=args.pilot_iter,
        )


if __name__ == "__main__":
    main()
