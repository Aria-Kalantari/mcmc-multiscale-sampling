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
      exhibits the rel-k reversal, and plot the reversal trajectory per K. The
      metric is the rel-k of the POOLED posterior-mean field across N independent
      chains that share one truth: single chains only plateau (~0.75 on 48x48);
      pooling across chains cancels per-chain drift and exposes the documented
      descend-then-rise (0.4629 -> 0.8855). The reduced 16x16 grid does NOT host
      the reversal (best pooled floor ~0.88); the study runs on the full 48x48
      config by default. Hypothesis: a larger K DELAYS the reversal onset.
      Mitigation, not a cure (SPEC 0 [V4] closure 1).

Run (from the repository root):

  python -m experiments.exp12_conditioning_diagnostics standardize
  # Full 48x48 pooled sweep (the real study; long -- hours, run overnight):
  python -m experiments.exp12_conditioning_diagnostics refresh
  # Cheap smoke test on the reduced grid (does NOT host the reversal):
  python -m experiments.exp12_conditioning_diagnostics refresh --reduced \
      --n-sweeps 40 --n-chains 2 --ks 1,4
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
def _pooled_relk_trajectory(
    chain_fields: list[np.ndarray],
    truth_k: np.ndarray,
    burn_fraction: float = 1.0 / 3.0,
    n_checkpoints: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Running rel-k of the POOLED posterior-mean field, at growing checkpoints.

    ``chain_fields`` is one ``(n_sweeps, ny, nx)`` accepted-field array per chain
    (all chains share one truth). Following the exp08c convention the first
    ``burn_fraction`` of each chain is discarded, then at each checkpoint the
    post-burn samples from every chain up to that prefix are pooled, averaged into
    a single posterior-mean field, and scored by rel-k of ``exp(mean)``. Pooling
    across independent chains is what cancels per-chain drift and exposes the
    documented descend-then-rise -- a single chain only plateaus. Returns
    ``(sweeps_axis, relk)`` where ``sweeps_axis`` is the 1-based sweep of each
    checkpoint.
    """
    n_sweeps = chain_fields[0].shape[0]
    burn = int(burn_fraction * n_sweeps)
    if n_sweeps - burn < 2:
        raise ValueError("burn-in leaves too few retained samples.")
    retained = [fields[burn:] for fields in chain_fields]
    n_ret = n_sweeps - burn
    prefixes = sorted(
        {
            max(1, int(round(fr * n_ret)))
            for fr in np.linspace(1.0 / n_checkpoints, 1.0, n_checkpoints)
        }
    )
    sweeps_axis = np.asarray([burn + p for p in prefixes], dtype=np.int64)
    relk = np.empty(len(prefixes), dtype=np.float64)
    for i, p in enumerate(prefixes):
        pooled = np.concatenate([r[:p] for r in retained], axis=0)
        g_mean = pooled.mean(axis=0)
        relk[i] = relative_error(permeability_from_log_field(g_mean), truth_k)
    return sweeps_axis, relk


def _run_pooled_chains(
    cfg: Config,
    K: int,
    n_chains: int,
    n_sweeps: int,
    Mb: int,
    beta: float,
    seed: int,
    initial_scale: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run ``n_chains`` independent chains sharing one truth, at refresh period K.

    Uses the exp08 ``_TruthReplayGenerator`` so every chain targets the SAME
    posterior (shared truth / observations) while starting from an independent
    field -- the pooled posterior mean is only meaningful across chains that share
    the truth. Returns ``(chain_fields, truth_k)``.
    """
    # Lazy import so exp12's module import stays light (tests import exp12).
    from experiments.exp08_convergence_diagnostics import (
        _TruthReplayGenerator,
        _truth_replay_prefix,
    )

    n_sub = cfg.n_coarse_x * cfg.n_coarse_y
    truth, truth_draw, noise_draw = _truth_replay_prefix(cfg)
    chain_fields: list[np.ndarray] = []
    for i in range(n_chains):
        rng = _TruthReplayGenerator(
            theta_true_draw=truth_draw,
            noise_draw=noise_draw,
            chain_seed=seed + 100 + i,
            n_initial_draws=1 + n_sub,
            initial_scale=initial_scale,
        )
        fields = np.empty((n_sweeps, cfg.ny, cfg.nx), dtype=np.float64)
        for state in red_black_conditioned_sampler(
            cfg,
            n_sweeps=n_sweeps,
            Mb=Mb,
            theta_p_method=_REFRESH_THETA_P,
            rng=rng,  # type: ignore[arg-type]
            beta=beta,
            acceptance=_REFRESH_ACCEPTANCE,
            prior_mode=_REFRESH_PRIOR_MODE,
            cond_refresh_period=K,
        ):
            fields[state.sweep - 1] = state.G_accepted
        chain_fields.append(fields)
        print(f"    chain {i + 1}/{n_chains} done", flush=True)
    return chain_fields, truth.k_true


def _reversal_onset(
    relk: np.ndarray,
    burn_fraction: float = 1.0 / 3.0,
    tol: float = 1e-2,
    rise_window: int = 3,
    min_descent: float = 0.03,
) -> int | None:
    """Return the 1-based checkpoint of a genuine post-burn reversal, else None.

    A reversal is a genuine descend-then-rise. Following the exp08c convention,
    the first ``burn_fraction`` of the trajectory (the start-up transient) is
    excluded so a burn-in dip cannot masquerade as the reversal minimum. Then
    two conditions must both hold on the post-burn region:

    1. A genuine descent: the post-burn minimum lies at least ``min_descent``
       below the first post-burn value. Without this, a chain that merely drifts
       upward (never recovering) is not a reversal -- it never descended.
    2. A sustained rise: a run of ``rise_window`` consecutive later checkpoints
       all exceed the minimum by more than ``tol`` (a lone noisy uptick does not
       count).

    A monotone, still-descending, or drift-from-the-start trajectory returns
    None.
    """
    series = np.asarray(relk, dtype=np.float64)
    n = series.size
    start = int(burn_fraction * n)
    if n - start < 3 or rise_window < 1:
        return None
    i_min = start + int(np.argmin(series[start:]))
    if i_min > n - 1 - rise_window:
        return None
    floor = float(series[i_min])
    if float(series[start]) - floor < min_descent:
        return None
    for j in range(i_min + 1, n - rise_window + 1):
        if np.all(series[j : j + rise_window] > floor + tol):
            return j + 1
    return None


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
    ks: list[int],
    axes_relks: list[tuple[np.ndarray, np.ndarray]],
    onsets: list[int | None],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for K, (sweeps, relk), onset in zip(ks, axes_relks, onsets):
        line = ax.plot(sweeps, relk, lw=1.4, marker=".", ms=3, label=f"K={K}")[0]
        if onset is not None:
            # onset is a 1-based checkpoint index into this K's trajectory
            ax.plot(
                sweeps[onset - 1],
                relk[onset - 1],
                marker="v",
                ms=8,
                color=line.get_color(),
            )
    ax.set_xlabel("sweep (checkpoint)")
    ax.set_ylabel("pooled posterior-mean rel-k")
    ax.set_title(
        "cond_refresh_period vs the global_field reversal\n"
        "(mitigation only -- SPEC 0 [V4] closure 1; markers = reversal onset)"
    )
    ax.legend(fontsize=8)
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
    n_chains: int,
    grid: str,
    axes_relks: list[tuple[np.ndarray, np.ndarray]],
    onsets: list[int | None],
) -> None:
    header = [
        "# exp12 (b) cond_refresh_period vs the global_field reversal",
        "",
        f"Grid {grid}, regime acceptance=posterior / prior_mode=global_field /",
        "theta_p=svd (the documented rel-k reversal, NOT the likelihood-only",
        "runaway). Metric: rel-k of the POOLED posterior-mean field across",
        f"{n_chains} independent chains sharing one truth (single chains only",
        "plateau; pooling exposes the descend-then-rise).",
        "",
        "**Mitigation, NOT a cure** -- SPEC 0 [V4] closure 1 proves the",
        "repeated-conditioning route cannot be repaired. A larger K can only delay",
        "the reversal; it does not recover the field.",
        "",
        f"budget per K = {n_chains} chains x {n_sweeps} sweeps x {n_sub}"
        f" subdomains = {n_chains * n_sweeps * n_sub} local updates.",
        "",
        "| K  | onset sweep | min pooled rel-k | final pooled rel-k |",
        "|----|-------------|------------------|--------------------|",
    ]
    rows = []
    for K, (sweeps, relk), onset in zip(ks, axes_relks, onsets):
        onset_str = "none" if onset is None else str(int(sweeps[onset - 1]))
        rows.append(
            f"| {K:>2} | {onset_str:>11} | {relk.min():>16.4f} |"
            f" {relk[-1]:>18.4f} |"
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
    cfg = _reduced_config(args.seed) if args.reduced else _full_config(args.seed)
    grid = f"{cfg.nx}x{cfg.ny}"
    ks = [int(k) for k in args.ks.split(",")]
    n_sub = cfg.n_coarse_x * cfg.n_coarse_y
    total = args.n_chains * args.n_sweeps * n_sub
    print(
        f"[refresh] grid {grid}, {cfg.n_global_modes} modes, n_chains="
        f"{args.n_chains}, n_sweeps={args.n_sweeps}, Mb={args.Mb}, K={ks}, "
        f"seed={args.seed}  ({total} updates/K)"
    )

    axes_relks: list[tuple[np.ndarray, np.ndarray]] = []
    onsets: list[int | None] = []
    for K in ks:
        print(f"  K={K} ...", flush=True)
        chain_fields, truth_k = _run_pooled_chains(
            cfg,
            K,
            args.n_chains,
            args.n_sweeps,
            args.Mb,
            cfg.beta,
            args.seed,
            args.initial_scale,
        )
        sweeps, relk = _pooled_relk_trajectory(chain_fields, truth_k)
        onset = _reversal_onset(relk)
        axes_relks.append((sweeps, relk))
        onsets.append(onset)
        # Incremental save so a crash mid-run keeps completed K's.
        np.savez(
            outdir / f"exp12_refresh_K{K}.npz",
            sweeps=sweeps,
            relk=relk,
            onset=-1 if onset is None else onset,
        )
        onset_str = "none" if onset is None else str(int(sweeps[onset - 1]))
        print(
            f"  K={K:>2}: min pooled rel-k={relk.min():.4f}, "
            f"final={relk[-1]:.4f}, onset sweep={onset_str}",
            flush=True,
        )

    _plot_refresh_relk(ks, axes_relks, onsets, outdir / "exp12_refresh_relk.png")
    _write_refresh_table(
        outdir / "exp12_refresh_table.md",
        ks,
        args.n_sweeps,
        n_sub,
        args.n_chains,
        grid,
        axes_relks,
        onsets,
    )
    print("DONE")


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
    b.add_argument("--n-sweeps", type=int, default=2000)
    b.add_argument("--n-chains", type=int, default=4)
    b.add_argument("--Mb", type=int, default=8)
    b.add_argument("--initial-scale", type=float, default=1.0)
    b.add_argument(
        "--reduced",
        action="store_true",
        help="use the cheap 16x16 grid (does NOT host the reversal; for testing)",
    )
    b.add_argument("--outdir", type=str, default=None)
    b.set_defaults(func=run_refresh)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
