"""exp12 -- M14 conditioning diagnostics (mechanism probes, NOT recovery claims).

SPEC 0 [V4] closure 1 is settled: the repeated-conditioning route cannot be
repaired (incompatible local-KLE conditionals). Everything here is diagnostic /
mechanism-evidence for the analysis paper. No result may be framed as a recovery.

Two subcommands, one per probe:

  (a) standardize -- replace the log-perm field G by (G - mean(G)) / std(G)
      before the forward solve. This is scale-invariant in the KLE amplitude, so
      the standardized likelihood is blind to ||theta||: it does NOT target
      pi(G|Y). The scale-invariance is proved exactly in
      tests/test_conditioning_diagnostics.py; here it is illustrated.

  (b) refresh -- sweep cond_refresh_period K in {1,2,4,8,16} on
      red_black_conditioned_sampler in the global_field posterior regime that
      exhibits the rel-k reversal, and plot the rel-k trajectory per K. A larger
      K is expected to DELAY the reversal onset. Mitigation, not a cure.

Run (from the repository root):

  python -m experiments.exp12_conditioning_diagnostics standardize
  python -m experiments.exp12_conditioning_diagnostics refresh
  python -m experiments.exp12_conditioning_diagnostics refresh --ks 1,2,4,8,16 \
      --n-sweeps 400
  python -m experiments.exp12_conditioning_diagnostics refresh --full-grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.bayes import log_prior  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.diagnostics import relative_error  # noqa: E402
from mcmc_multiscale.field import (  # noqa: E402
    field_from_theta,
    permeability_from_log_field,
    reshape_field,
)
from mcmc_multiscale.forward import ForwardModel  # noqa: E402
from mcmc_multiscale.grid import cell_centered_grid  # noqa: E402
from mcmc_multiscale.kle import top_eigenpairs  # noqa: E402
from mcmc_multiscale.mcmc import metropolis_hastings  # noqa: E402
from mcmc_multiscale.observations import make_truth, restrict_pressure  # noqa: E402
from mcmc_multiscale.proposals import make_pcn_proposal  # noqa: E402
from mcmc_multiscale.sampler import red_black_conditioned_sampler  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "exp12"

# Regime that exhibits the global_field rel-k reversal (NOTES: 0.4629 -> 0.8855).
_REFRESH_ACCEPTANCE = "posterior"
_REFRESH_PRIOR_MODE = "global_field"
_REFRESH_THETA_P = "svd"
_DEFAULT_KS = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
def _reduced_config(seed: int) -> Config:
    """The reduced 16x16 profile (40 modes, 36 sensors, misfit floor 18).

    ``Nc=20`` keeps the local modes ``Nc + Mb`` within the smallest enlarged
    subdomain (36 cells at a corner under the 4x4 coarse / overlap-2 partition).
    """
    return Config(
        nx=16,
        ny=16,
        n_coarse_x=4,
        n_coarse_y=4,
        overlap_cells=2,
        n_global_modes=40,
        Nc=20,
        n_obs_x=6,
        n_obs_y=6,
        seed=seed,
    )


def _full_config(seed: int) -> Config:
    """The full 48x48 default profile."""
    return Config(seed=seed)


def _global_kle(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    C = exp_covariance(points, cfg.sigma, cfg.corr_length)
    return top_eigenpairs(C, cfg.n_global_modes)


# --------------------------------------------------------------------------- #
# (a) standardization primitives
# --------------------------------------------------------------------------- #
def _standardize_log_field(G: np.ndarray) -> np.ndarray:
    """Return (G - mean(G)) / std(G) in float64.

    DIAGNOSTIC ONLY. This is a nonlinear, non-measure-preserving remap of the
    log-perm field. It CHANGES THE TARGET (SPEC 12): it is scale-invariant in
    the KLE amplitude, so the resulting likelihood is blind to ||theta||. Never
    use it for a recovery claim.
    """
    G_arr = np.asarray(G, dtype=np.float64)
    std = float(G_arr.std())
    if std <= 1e-300:
        raise ValueError("cannot standardize a (near-)constant field: std is 0.")
    return (G_arr - float(G_arr.mean())) / std


def _field_misfit(
    theta: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    fwd: ForwardModel,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_obs: float,
    ny: int,
    nx: int,
    standardize: bool,
) -> float:
    """0.5/sigma_obs^2 ||R(p(theta)) - y||^2, optionally standardizing G first.

    With ``standardize=False`` this equals ``bayes.misfit``. With
    ``standardize=True`` the log-perm field is replaced by its z-score before the
    forward solve -- the M14(a) probe.
    """
    G_vec = field_from_theta(Phi, lam, theta)
    G = reshape_field(G_vec, ny, nx)
    if standardize:
        G = _standardize_log_field(G)
    k = permeability_from_log_field(G)
    p = fwd.solve(k)
    pred = restrict_pressure(p, sensor_idx)
    residual = pred - np.asarray(y_obs, dtype=np.float64)
    return float(0.5 / sigma_obs**2 * np.dot(residual, residual))


def _standardized_misfit(
    theta: np.ndarray,
    Phi: np.ndarray,
    lam: np.ndarray,
    fwd: ForwardModel,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_obs: float,
    ny: int,
    nx: int,
) -> float:
    """The standardized misfit (M14(a) probe). Scale-invariant in ``theta``."""
    return _field_misfit(
        theta, Phi, lam, fwd, y_obs, sensor_idx, sigma_obs, ny, nx, standardize=True
    )


def _run_field_chain(
    cfg: Config,
    truth,
    Phi: np.ndarray,
    lam: np.ndarray,
    fwd: ForwardModel,
    rng: np.random.Generator,
    theta0: np.ndarray,
    n_iter: int,
    beta: float,
    standardize: bool,
    burn_fraction: float = 0.5,
) -> dict:
    """Global pCN chain on the (optionally standardized) target; posterior mean."""

    def log_density(theta: np.ndarray) -> float:
        return log_prior(theta) - _field_misfit(
            theta,
            Phi,
            lam,
            fwd,
            truth.y_obs,
            truth.sensor_idx,
            cfg.sigma_obs,
            cfg.ny,
            cfg.nx,
            standardize,
        )

    proposal = make_pcn_proposal(beta)
    states = list(
        metropolis_hastings(
            log_density,
            proposal,
            theta0.astype(np.float64, copy=True),
            n_iter,
            rng,
            log_prior_fn=log_prior,
        )
    )
    thetas = np.stack([s.theta for s in states]).astype(np.float64)
    burn = int(burn_fraction * len(states))
    post = thetas[burn:]
    mean_theta = post.mean(axis=0)
    mean_field = reshape_field(field_from_theta(Phi, lam, mean_theta), cfg.ny, cfg.nx)
    mean_k = permeability_from_log_field(mean_field)
    return {
        "mean_field": mean_field,
        "rel_k": relative_error(mean_k, truth.k_true),
        "misfit_own_target": _field_misfit(
            mean_theta,
            Phi,
            lam,
            fwd,
            truth.y_obs,
            truth.sensor_idx,
            cfg.sigma_obs,
            cfg.ny,
            cfg.nx,
            standardize,
        ),
        "theta_norm": float(np.mean(np.linalg.norm(post, axis=1))),
        "acceptance": float(np.mean([s.accepted for s in states])),
    }


# --------------------------------------------------------------------------- #
# (b) refresh / reversal primitives
# --------------------------------------------------------------------------- #
def _sweep_relk(states: list) -> np.ndarray:
    """Return one accepted rel-k per sweep (the last state of each sweep)."""
    by_sweep: dict[int, float] = {}
    for state in states:
        by_sweep[state.sweep] = state.relative_k_error_accepted
    return np.asarray([by_sweep[s] for s in sorted(by_sweep)], dtype=np.float64)


def _reversal_onset(relk: np.ndarray, tol: float = 1e-3) -> int | None:
    """Return the 1-based sweep at which rel-k first reverses upward, else None.

    A reversal is a descend-then-rise: the running minimum occurs at an interior
    sweep, and a later sweep exceeds that minimum by more than ``tol``. A
    monotone or still-descending trajectory has no reversal (returns None).
    """
    series = np.asarray(relk, dtype=np.float64)
    if series.size < 3:
        return None
    i_min = int(np.argmin(series))
    if i_min == 0 or i_min == series.size - 1:
        return None
    floor = float(series[i_min])
    for j in range(i_min + 1, series.size):
        if series[j] > floor + tol:
            return j + 1
    return None


def _run_refresh(
    cfg: Config, Mb: int, K: int, n_sweeps: int, beta: float, seed: int
) -> np.ndarray:
    """Run the red-black global_field chain at cond_refresh_period=K; per-sweep rel-k."""
    states = list(
        red_black_conditioned_sampler(
            cfg,
            n_sweeps=n_sweeps,
            Mb=Mb,
            theta_p_method=_REFRESH_THETA_P,
            rng=np.random.default_rng(seed),
            beta=beta,
            acceptance=_REFRESH_ACCEPTANCE,
            prior_mode=_REFRESH_PRIOR_MODE,
            cond_refresh_period=K,
        )
    )
    return _sweep_relk(states)


# --------------------------------------------------------------------------- #
# Plotting / tables
# --------------------------------------------------------------------------- #
def _plot_standardize_scale(
    scales: np.ndarray, misfit_true: np.ndarray, misfit_std: np.ndarray, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(scales, misfit_true, marker="o", ms=3, label="true target  misfit(a*theta)")
    ax.plot(
        scales,
        misfit_std,
        marker="s",
        ms=3,
        label="standardized  misfit(a*theta)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("amplitude scale a  (theta -> a * theta)")
    ax.set_ylabel("data misfit")
    ax.set_title("Standardization blinds the likelihood to ||theta||")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path.name}")


def _plot_standardize_fields(
    field_true: np.ndarray,
    field_std: np.ndarray,
    truth_field: np.ndarray,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("true-target posterior mean", field_true),
        ("standardized-target posterior mean", field_std),
        ("truth G", truth_field),
    ]
    vmin = min(p[1].min() for p in panels)
    vmax = max(p[1].max() for p in panels)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    im = None
    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.suptitle("DIAGNOSTIC ONLY -- standardization changes the target", fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path.name}")


def _plot_refresh_relk(
    ks: list[int], relks: list[np.ndarray], onsets: list[int | None], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for K, relk, onset in zip(ks, relks, onsets):
        sweeps = np.arange(1, relk.size + 1)
        line = ax.plot(sweeps, relk, lw=1.2, label=f"K={K}")[0]
        if onset is not None:
            ax.plot(
                onset,
                relk[onset - 1],
                marker="v",
                ms=6,
                color=line.get_color(),
            )
    ax.set_xlabel("sweep")
    ax.set_ylabel("rel-k of accepted field")
    ax.set_title(
        "cond_refresh_period delays but does not cure the reversal\n"
        "(mitigation only -- SPEC 0 [V4] closure 1)"
    )
    ax.legend(fontsize=8, title="markers = reversal onset")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  figure -> {path.name}")


def _write_standardize_table(
    path: Path,
    true_chain: dict,
    std_chain: dict,
    scale_rel_residual: float,
    misfit_true_range: tuple[float, float],
) -> None:
    lo, hi = misfit_true_range
    lines = [
        "# exp12 (a) standardization probe",
        "",
        "**DIAGNOSTIC ONLY -- CHANGES THE TARGET, NOT A RECOVERY.**",
        "",
        "Standardization is scale-invariant in the KLE amplitude, so the",
        "standardized likelihood is blind to ||theta||: it does not target",
        "pi(G|Y). It does not fix the drift -- it makes the drift invisible to",
        "the likelihood. Proof (exact) is in tests/test_conditioning_diagnostics.py.",
        "",
        "Over the amplitude sweep a in [0.3, 3.0] (theta -> a*theta):",
        f"- true-target misfit sweeps {lo:.2f} .. {hi:.2f} (the data sees ||theta||).",
        f"- standardized misfit is flat: relative spread {scale_rel_residual:.2e}"
        f" (machine zero -- the data is blind to ||theta||).",
        "",
        "rel-k below is NOT comparable across the two rows: they are different",
        "posteriors (SPEC 12 pitfall 14). A lower standardized rel-k is meaningless",
        "-- the standardized target is scale-blind, not 'better'.",
        "",
        "| target        | rel-k  | misfit (own target) | mean \\|\\|theta\\|\\| | accept |",
        "|---------------|--------|---------------------|-------------------|--------|",
        "| pi(G\\|Y)       | {:.4f} | {:>19.3f} | {:>17.3f} | {:>6.3f} |".format(
            true_chain["rel_k"],
            true_chain["misfit_own_target"],
            true_chain["theta_norm"],
            true_chain["acceptance"],
        ),
        "| standardized  | {:.4f} | {:>19.3f} | {:>17.3f} | {:>6.3f} |".format(
            std_chain["rel_k"],
            std_chain["misfit_own_target"],
            std_chain["theta_norm"],
            std_chain["acceptance"],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  table  -> {path.name}")
    print("\n".join(lines))


def _write_refresh_table(
    path: Path,
    ks: list[int],
    n_sweeps: int,
    n_sub: int,
    relks: list[np.ndarray],
    onsets: list[int | None],
) -> None:
    header = [
        "# exp12 (b) cond_refresh_period reversal sweep",
        "",
        "Regime: acceptance=posterior, prior_mode=global_field, theta_p=svd",
        "(the rel-k reversal, NOT the likelihood-only runaway).",
        "",
        "**Mitigation, NOT a cure** -- SPEC 0 [V4] closure 1 proves the",
        "repeated-conditioning route cannot be repaired. A larger K only delays",
        "the reversal; it does not recover the field.",
        "",
        f"budget per K = {n_sweeps} sweeps x {n_sub} subdomains ="
        f" {n_sweeps * n_sub} local updates.",
        "",
        "| K  | onset sweep | min rel-k | final rel-k |",
        "|----|-------------|-----------|-------------|",
    ]
    rows = []
    for K, relk, onset in zip(ks, relks, onsets):
        onset_str = "none" if onset is None else str(onset)
        rows.append(
            f"| {K:>2} | {onset_str:>11} | {relk.min():.4f}    |"
            f" {relk[-1]:.4f}      |"
        )
    lines = header + rows + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  table  -> {path.name}")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def run_standardize(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = _full_config(args.seed) if args.full_grid else _reduced_config(args.seed)
    print(
        f"[standardize] grid {cfg.nx}x{cfg.ny}, {cfg.n_global_modes} modes, "
        f"n_iter={args.n_iter}, seed={args.seed}"
    )

    truth = make_truth(cfg, np.random.default_rng(args.seed + 1))
    Phi, lam = _global_kle(cfg)
    fwd = ForwardModel(cfg)
    beta = args.beta

    # Non-constant start (theta=0 gives a constant field the probe cannot z-score).
    theta0 = np.random.default_rng(args.seed + 7).standard_normal(cfg.n_global_modes)

    # Headline visual: scale-invariance of the standardized likelihood.
    scales = np.linspace(0.3, 3.0, 25, dtype=np.float64)
    fixed_theta = np.random.default_rng(args.seed + 11).standard_normal(
        cfg.n_global_modes
    )
    misfit_true = np.asarray(
        [
            _field_misfit(
                a * fixed_theta,
                Phi,
                lam,
                fwd,
                truth.y_obs,
                truth.sensor_idx,
                cfg.sigma_obs,
                cfg.ny,
                cfg.nx,
                standardize=False,
            )
            for a in scales
        ]
    )
    misfit_std = np.asarray(
        [
            _standardized_misfit(
                a * fixed_theta,
                Phi,
                lam,
                fwd,
                truth.y_obs,
                truth.sensor_idx,
                cfg.sigma_obs,
                cfg.ny,
                cfg.nx,
            )
            for a in scales
        ]
    )
    scale_rel_residual = float(
        (misfit_std.max() - misfit_std.min()) / max(1.0, float(misfit_std.mean()))
    )
    misfit_true_range = (float(misfit_true.min()), float(misfit_true.max()))

    true_chain = _run_field_chain(
        cfg,
        truth,
        Phi,
        lam,
        fwd,
        np.random.default_rng(args.seed + 100),
        theta0,
        args.n_iter,
        beta,
        standardize=False,
    )
    std_chain = _run_field_chain(
        cfg,
        truth,
        Phi,
        lam,
        fwd,
        np.random.default_rng(args.seed + 200),
        theta0,
        args.n_iter,
        beta,
        standardize=True,
    )

    _plot_standardize_scale(
        scales, misfit_true, misfit_std, outdir / "exp12_standardize_scale.png"
    )
    _plot_standardize_fields(
        true_chain["mean_field"],
        std_chain["mean_field"],
        truth.G_true,
        outdir / "exp12_standardize_fields.png",
    )
    _write_standardize_table(
        outdir / "exp12_standardize_table.md",
        true_chain,
        std_chain,
        scale_rel_residual,
        misfit_true_range,
    )


def run_refresh(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = _full_config(args.seed) if args.full_grid else _reduced_config(args.seed)
    ks = [int(k) for k in args.ks.split(",")]
    n_sub = cfg.n_coarse_x * cfg.n_coarse_y
    print(
        f"[refresh] grid {cfg.nx}x{cfg.ny}, {cfg.n_global_modes} modes, "
        f"n_sweeps={args.n_sweeps}, Mb={args.Mb}, K={ks}, seed={args.seed}"
    )

    relks: list[np.ndarray] = []
    onsets: list[int | None] = []
    for K in ks:
        relk = _run_refresh(cfg, args.Mb, K, args.n_sweeps, cfg.beta, args.seed)
        onset = _reversal_onset(relk)
        relks.append(relk)
        onsets.append(onset)
        onset_str = "none" if onset is None else str(onset)
        print(
            f"  K={K:>2}: min rel-k={relk.min():.4f}, "
            f"final={relk[-1]:.4f}, onset sweep={onset_str}"
        )

    _plot_refresh_relk(ks, relks, onsets, outdir / "exp12_refresh_relk.png")
    _write_refresh_table(
        outdir / "exp12_refresh_table.md",
        ks,
        args.n_sweeps,
        n_sub,
        relks,
        onsets,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="probe", required=True)

    a = sub.add_parser("standardize", help="M14(a) standardization probe")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--n-iter", type=int, default=4000)
    a.add_argument("--beta", type=float, default=0.06)
    a.add_argument("--full-grid", action="store_true")
    a.add_argument("--outdir", type=str, default=None)
    a.set_defaults(func=run_standardize)

    b = sub.add_parser("refresh", help="M14(b) cond_refresh_period sweep")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--ks", type=str, default=",".join(str(k) for k in _DEFAULT_KS))
    b.add_argument("--n-sweeps", type=int, default=200)
    b.add_argument("--Mb", type=int, default=8)
    b.add_argument("--full-grid", action="store_true")
    b.add_argument("--outdir", type=str, default=None)
    b.set_defaults(func=run_refresh)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
