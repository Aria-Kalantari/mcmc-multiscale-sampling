"""Static local-conditioning experiment for Phase 1 / M1."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.conditioning.constraints import (  # noqa: E402
    build_A,
    build_c,
    select_conditioning_points,
)
from mcmc_multiscale.conditioning.nullspace import project_null  # noqa: E402
from mcmc_multiscale.conditioning.particular import svd_min_norm  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.field import field_from_theta, reshape_field  # noqa: E402
from mcmc_multiscale.grid import cell_centered_grid  # noqa: E402
from mcmc_multiscale.kle import top_eigenpairs  # noqa: E402
from mcmc_multiscale.subdomain import (  # noqa: E402
    Subdomain,
    interface_jump_rms,
    make_subdomain,
)


@dataclass(frozen=True)
class SummaryRow:
    Mb: int
    Next: int
    RankA: int
    NullDim: int
    CondA: float
    MinSingularValue: float
    MeanResidualConditioned: float
    MeanResidualUnconditioned: float
    MeanJumpConditioned: float
    MeanJumpUnconditioned: float


def _build_static_inputs(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Subdomain]:
    rng = np.random.default_rng(cfg.seed)
    _, _, _, _, global_pts = cell_centered_grid(cfg.nx, cfg.ny)

    C_global = exp_covariance(global_pts, cfg.sigma, cfg.corr_length)
    Phi_global, lambda_global = top_eigenpairs(C_global, cfg.n_global_modes)
    theta_global = rng.standard_normal(cfg.n_global_modes, dtype=np.float64)
    G_old_vec = field_from_theta(Phi_global, lambda_global, theta_global)

    sub = make_subdomain(cfg)
    local_pts = global_pts[sub.local_global_idx, :]
    C_local = exp_covariance(local_pts, cfg.sigma, cfg.corr_length)
    local_modes_max = cfg.Nc + max(cfg.Mb_list)
    Phi_local, lambda_local = top_eigenpairs(C_local, local_modes_max)
    return G_old_vec, global_pts, local_pts, Phi_local, lambda_local, sub


def run_experiment(cfg: Config | None = None) -> list[SummaryRow]:
    """Run the static-conditioning table from the MATLAB reference."""

    cfg = Config() if cfg is None else cfg
    rng = np.random.default_rng(cfg.seed)
    G_old_vec, _, local_pts, Phi_local, lambda_local, sub = _build_static_inputs(cfg)

    rows: list[SummaryRow] = []
    for Mb in cfg.Mb_list:
        Next = cfg.Nc + Mb
        cond_local_idx = select_conditioning_points(
            local_pts, sub.core_local_idx, sub.buffer_local_idx, Mb
        )
        Phi_ext = Phi_local[:, :Next]
        sqrt_lam = np.sqrt(lambda_local[:Next])
        A = build_A(Phi_ext, sqrt_lam, cond_local_idx)
        c = build_c(G_old_vec, sub.local_global_idx, cond_local_idx)
        theta_p, Z, info = svd_min_norm(A, c)

        conditioned_residuals = np.empty(cfg.n_samples, dtype=np.float64)
        unconstrained_residuals = np.empty(cfg.n_samples, dtype=np.float64)
        conditioned_jumps = np.empty(cfg.n_samples, dtype=np.float64)
        unconstrained_jumps = np.empty(cfg.n_samples, dtype=np.float64)

        for sample_idx in range(cfg.n_samples):
            eta = rng.standard_normal(Next, dtype=np.float64)
            theta_n = project_null(Z, eta)
            theta_cond = theta_p + theta_n
            G_local_cond = field_from_theta(Phi_ext, lambda_local[:Next], theta_cond)

            temp_vec = G_old_vec.copy()
            temp_vec[sub.core_global_idx] = G_local_cond[sub.core_local_idx]
            G_cond = reshape_field(temp_vec, cfg.ny, cfg.nx)
            conditioned_residuals[sample_idx] = np.linalg.norm(
                A @ theta_cond - c
            ) / max(1.0, np.linalg.norm(c))
            conditioned_jumps[sample_idx] = interface_jump_rms(G_cond, sub)

            theta_unc = rng.standard_normal(Next, dtype=np.float64)
            G_local_unc = field_from_theta(Phi_ext, lambda_local[:Next], theta_unc)
            temp_vec = G_old_vec.copy()
            temp_vec[sub.core_global_idx] = G_local_unc[sub.core_local_idx]
            G_unc = reshape_field(temp_vec, cfg.ny, cfg.nx)
            unconstrained_residuals[sample_idx] = np.linalg.norm(
                A @ theta_unc - c
            ) / max(1.0, np.linalg.norm(c))
            unconstrained_jumps[sample_idx] = interface_jump_rms(G_unc, sub)

        rows.append(
            SummaryRow(
                Mb=Mb,
                Next=Next,
                RankA=info.rankA,
                NullDim=Z.shape[1],
                CondA=info.cond_effective,
                MinSingularValue=info.min_nonzero_singular,
                MeanResidualConditioned=float(np.mean(conditioned_residuals)),
                MeanResidualUnconditioned=float(np.mean(unconstrained_residuals)),
                MeanJumpConditioned=float(np.mean(conditioned_jumps)),
                MeanJumpUnconditioned=float(np.mean(unconstrained_jumps)),
            )
        )
    return rows


def print_summary(rows: list[SummaryRow]) -> None:
    headers = (
        "Mb",
        "Next",
        "RankA",
        "NullDim",
        "CondA",
        "MinSingularValue",
        "MeanResidualConditioned",
        "MeanResidualUnconditioned",
        "MeanJumpConditioned",
        "MeanJumpUnconditioned",
    )
    widths = (4, 6, 7, 8, 12, 18, 27, 29, 23, 25)
    print("SUMMARY TABLE")
    print(" ".join(f"{header:>{width}}" for header, width in zip(headers, widths)))
    for row in rows:
        print(
            f"{row.Mb:4d} "
            f"{row.Next:6d} "
            f"{row.RankA:7d} "
            f"{row.NullDim:8d} "
            f"{row.CondA:12.4e} "
            f"{row.MinSingularValue:18.4e} "
            f"{row.MeanResidualConditioned:27.4e} "
            f"{row.MeanResidualUnconditioned:29.4e} "
            f"{row.MeanJumpConditioned:23.4e} "
            f"{row.MeanJumpUnconditioned:25.4e}"
        )


def main() -> None:
    cfg = Config()
    print("LOCAL CONDITIONING PROJECT")
    print(f"Global grid: {cfg.nx} x {cfg.ny}")
    print(f"Coarse partition: {cfg.n_coarse_x} x {cfg.n_coarse_y}")
    print(f"Target subdomain: row {cfg.target_row}, col {cfg.target_col}")
    print(f"Overlap width: {cfg.overlap_cells} fine-grid cells")
    print(f"Base local stochastic dimension Nc = {cfg.Nc}")
    print(f"Mb values tested: {' '.join(str(mb) for mb in cfg.Mb_list)}")
    print()
    rows = run_experiment(cfg)
    print_summary(rows)


if __name__ == "__main__":
    main()
