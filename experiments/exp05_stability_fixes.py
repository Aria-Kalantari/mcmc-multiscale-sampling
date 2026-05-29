"""M5 stability-fix comparison for repeated local conditioning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.app_core import (  # noqa: E402
    RunSummary,
    default_m5_methods,
    run_methods,
)
from mcmc_multiscale.config import Config  # noqa: E402


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4e}"


def _print_table(rows: Sequence[RunSummary]) -> None:
    print(
        "method                    rho      cand_max cand_final acc_max  acc_final "
        "cand/exp acc/exp accept resid_mean resid_max jump_acc relk_acc_final hidden_mean hidden_max"
    )
    for row in rows:
        rho_text = "n/a" if row.rho is None else f"{row.rho:.0e}"
        rel_k_final = (
            np.nan
            if row.final_relative_k_error_accepted is None
            else row.final_relative_k_error_accepted
        )
        print(
            f"{row.label:<24} {rho_text:>7} "
            f"{row.max_candidate_theta_norm:9.4f} "
            f"{row.final_candidate_theta_norm:10.4f} "
            f"{row.max_accepted_theta_norm:8.4f} "
            f"{row.final_accepted_theta_norm:10.4f} "
            f"{row.candidate_norm_over_expected:8.3f} "
            f"{row.accepted_norm_over_expected:7.3f} "
            f"{row.acceptance_rate:6.3f} "
            f"{row.mean_residual:10.3e} {row.max_residual:9.3e} "
            f"{row.mean_interface_jump_accepted:8.3e} {rel_k_final:14.3e} "
            f"{_fmt_optional(row.mean_hidden_null_norm):>11} "
            f"{_fmt_optional(row.max_hidden_null_norm):>10}"
        )


def _by_label(rows: Sequence[RunSummary]) -> dict[str, RunSummary]:
    return {row.label: row for row in rows}


def _print_comparisons(rows: Sequence[RunSummary]) -> None:
    keyed = _by_label(rows)
    lu = keyed["hard_data_lu"]
    svd = keyed["hard_data_svd"]
    stabilized = keyed["hard_data_lu_stabilized"]
    zero = keyed["hard_zero_svd"]
    tiny = np.finfo(np.float64).tiny
    zero_error = zero.final_relative_k_error_accepted
    svd_error = svd.final_relative_k_error_accepted

    print()
    print("Key comparisons:")
    print(
        "  LU/SVD max candidate norm ratio: "
        f"{lu.max_candidate_theta_norm / max(svd.max_candidate_theta_norm, tiny):.4f}"
    )
    print(
        "  LU/SVD max accepted norm ratio: "
        f"{lu.max_accepted_theta_norm / max(svd.max_accepted_theta_norm, tiny):.4f}"
    )
    print(
        "  LU-stabilized/SVD max candidate norm ratio: "
        f"{stabilized.max_candidate_theta_norm / max(svd.max_candidate_theta_norm, tiny):.4f}"
    )
    print(
        "  LU-stabilized/SVD max accepted norm ratio: "
        f"{stabilized.max_accepted_theta_norm / max(svd.max_accepted_theta_norm, tiny):.4f}"
    )
    print(
        "  c=0 final accepted relative k error vs SVD: "
        f"{zero_error:.4e} vs {svd_error:.4e}"
    )
    print()
    print("Soft rho sweep: residual vs norm vs error")
    for row in rows:
        if not row.label.startswith("soft_data"):
            continue
        rel_k_final = row.final_relative_k_error_accepted
        print(
            f"  rho={row.rho:.0e}: residual_mean={row.mean_residual:.4e}, "
            f"candidate_max={row.max_candidate_theta_norm:.4f}, "
            f"accepted_final={row.final_accepted_theta_norm:.4f}, "
            f"rel_k_final={rel_k_final:.4e}"
        )


def run_comparison(
    cfg: Config,
    n_iter: int,
    Mb: int,
    beta: float,
    seed: int,
    rho_values: Sequence[float],
    proposal: str = "pcn",
) -> list[RunSummary]:
    """Run the M5 comparison through the shared app-core helper."""

    results = run_methods(
        cfg=cfg,
        methods=default_m5_methods(rho_values),
        n_iter=n_iter,
        Mb=Mb,
        beta=beta,
        seed=seed,
        proposal=proposal,
    )
    return [summary for _, summary in results.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--rho-values",
        type=float,
        nargs="*",
        default=[1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4],
    )
    args = parser.parse_args()

    cfg = Config(seed=args.seed, beta=args.beta)
    rows = run_comparison(
        cfg=cfg,
        n_iter=args.n_iter,
        Mb=args.mb,
        beta=args.beta,
        seed=args.seed,
        rho_values=args.rho_values,
    )

    print("M5 STABILITY FIXES")
    print(
        f"n_iter={args.n_iter}, Mb={args.mb}, beta={args.beta}, seed={args.seed}, "
        f"expected_norm={rows[0].expected_norm:.4f}"
    )
    print()
    _print_table(rows)
    _print_comparisons(rows)


if __name__ == "__main__":
    main()
