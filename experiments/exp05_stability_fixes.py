"""M5 stability-fix comparison for repeated local conditioning."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.sampler import conditioned_sampler  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    label: str
    theta_p_method: str
    rhs_mode: str = "data"
    conditioning_mode: str = "hard"
    rho: float | None = None


@dataclass(frozen=True)
class RunSummary:
    label: str
    rho: float | None
    n_iter: int
    Mb: int
    beta: float
    expected_norm: float
    candidate_max: float
    candidate_final: float
    accepted_max: float
    accepted_final: float
    candidate_over_expected: float
    accepted_over_expected: float
    acceptance_rate: float
    residual_mean: float
    residual_max: float
    jump_candidate_mean: float
    jump_accepted_mean: float
    rel_k_candidate_mean: float
    rel_k_candidate_final: float
    rel_k_accepted_mean: float
    rel_k_accepted_final: float
    hidden_mean: float
    hidden_max: float


def _summarize(
    spec: MethodSpec, cfg: Config, n_iter: int, Mb: int, beta: float, seed: int
) -> RunSummary:
    candidate_norms: list[float] = []
    accepted_norms: list[float] = []
    accepted: list[bool] = []
    residuals: list[float] = []
    candidate_jumps: list[float] = []
    accepted_jumps: list[float] = []
    candidate_errors: list[float] = []
    accepted_errors: list[float] = []
    hidden_norms: list[float] = []
    expected_norm = np.nan

    for state in conditioned_sampler(
        cfg=cfg,
        n_iter=n_iter,
        Mb=Mb,
        theta_p_method=spec.theta_p_method,
        rng=np.random.default_rng(seed),
        beta=beta,
        update_scheme="single",
        proposal="pcn",
        rhs_mode=spec.rhs_mode,
        conditioning_mode=spec.conditioning_mode,
        rho=spec.rho,
    ):
        expected_norm = state.expected_norm
        candidate_norms.append(state.theta_norm_candidate)
        accepted_norms.append(state.theta_norm_accepted)
        accepted.append(state.accepted)
        residuals.append(state.constraint_residual_candidate)
        candidate_jumps.append(state.interface_jump_candidate)
        accepted_jumps.append(state.interface_jump_accepted)
        candidate_errors.append(float(state.relative_k_error_candidate))
        accepted_errors.append(float(state.relative_k_error_accepted))
        hidden_norms.append(state.hidden_null_norm)

    cand = np.asarray(candidate_norms, dtype=np.float64)
    acc = np.asarray(accepted_norms, dtype=np.float64)
    expected = float(expected_norm)
    hidden = np.asarray(hidden_norms, dtype=np.float64)

    return RunSummary(
        label=spec.label,
        rho=spec.rho,
        n_iter=n_iter,
        Mb=Mb,
        beta=beta,
        expected_norm=expected,
        candidate_max=float(np.max(cand)),
        candidate_final=float(cand[-1]),
        accepted_max=float(np.max(acc)),
        accepted_final=float(acc[-1]),
        candidate_over_expected=float(np.max(cand) / expected),
        accepted_over_expected=float(np.max(acc) / expected),
        acceptance_rate=float(np.mean(np.asarray(accepted, dtype=bool))),
        residual_mean=float(np.mean(np.asarray(residuals, dtype=np.float64))),
        residual_max=float(np.max(np.asarray(residuals, dtype=np.float64))),
        jump_candidate_mean=float(
            np.mean(np.asarray(candidate_jumps, dtype=np.float64))
        ),
        jump_accepted_mean=float(np.mean(np.asarray(accepted_jumps, dtype=np.float64))),
        rel_k_candidate_mean=float(
            np.mean(np.asarray(candidate_errors, dtype=np.float64))
        ),
        rel_k_candidate_final=float(candidate_errors[-1]),
        rel_k_accepted_mean=float(
            np.mean(np.asarray(accepted_errors, dtype=np.float64))
        ),
        rel_k_accepted_final=float(accepted_errors[-1]),
        hidden_mean=(
            float(np.nanmean(hidden)) if not np.all(np.isnan(hidden)) else np.nan
        ),
        hidden_max=float(np.nanmax(hidden)) if not np.all(np.isnan(hidden)) else np.nan,
    )


def _fmt(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.4e}"


def _print_table(rows: list[RunSummary]) -> None:
    print(
        "method                    rho      cand_max cand_final acc_max  acc_final "
        "cand/exp acc/exp accept resid_mean resid_max jump_acc relk_acc_final hidden_mean hidden_max"
    )
    for row in rows:
        rho_text = "n/a" if row.rho is None else f"{row.rho:.0e}"
        print(
            f"{row.label:<24} {rho_text:>7} "
            f"{row.candidate_max:9.4f} {row.candidate_final:10.4f} "
            f"{row.accepted_max:8.4f} "
            f"{row.accepted_final:10.4f} "
            f"{row.candidate_over_expected:8.3f} {row.accepted_over_expected:7.3f} "
            f"{row.acceptance_rate:6.3f} "
            f"{row.residual_mean:10.3e} {row.residual_max:9.3e} "
            f"{row.jump_accepted_mean:8.3e} {row.rel_k_accepted_final:14.3e} "
            f"{_fmt(row.hidden_mean):>11} {_fmt(row.hidden_max):>10}"
        )


def _by_label(rows: list[RunSummary]) -> dict[str, RunSummary]:
    return {row.label: row for row in rows}


def _print_comparisons(rows: list[RunSummary]) -> None:
    keyed = _by_label(rows)
    lu = keyed["hard_data_lu"]
    svd = keyed["hard_data_svd"]
    stabilized = keyed["hard_data_lu_stabilized"]
    zero = keyed["hard_zero_svd"]
    tiny = np.finfo(np.float64).tiny

    print()
    print("Key comparisons:")
    print(
        "  LU/SVD max candidate norm ratio: "
        f"{lu.candidate_max / max(svd.candidate_max, tiny):.4f}"
    )
    print(
        "  LU/SVD max accepted norm ratio: "
        f"{lu.accepted_max / max(svd.accepted_max, tiny):.4f}"
    )
    print(
        "  LU-stabilized/SVD max candidate norm ratio: "
        f"{stabilized.candidate_max / max(svd.candidate_max, tiny):.4f}"
    )
    print(
        "  LU-stabilized/SVD max accepted norm ratio: "
        f"{stabilized.accepted_max / max(svd.accepted_max, tiny):.4f}"
    )
    print(
        "  c=0 final accepted relative k error vs SVD: "
        f"{zero.rel_k_accepted_final:.4e} vs {svd.rel_k_accepted_final:.4e}"
    )
    print()
    print("Soft rho sweep: residual vs norm vs error")
    for row in rows:
        if not row.label.startswith("soft_data"):
            continue
        print(
            f"  rho={row.rho:.0e}: residual_mean={row.residual_mean:.4e}, "
            f"candidate_max={row.candidate_max:.4f}, "
            f"accepted_final={row.accepted_final:.4f}, "
            f"rel_k_final={row.rel_k_accepted_final:.4e}"
        )


def _method_specs(rho_values: list[float]) -> list[MethodSpec]:
    specs = [
        MethodSpec("hard_data_lu", "lu"),
        MethodSpec("hard_data_svd", "svd"),
        MethodSpec("hard_data_lu_stabilized", "lu_stabilized"),
        MethodSpec("hard_zero_svd", "svd", rhs_mode="zero"),
    ]
    specs.extend(
        MethodSpec(
            label=f"soft_data_rho_{rho:.0e}",
            theta_p_method="soft",
            conditioning_mode="soft",
            rho=rho,
        )
        for rho in rho_values
    )
    return specs


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
    rows = [
        _summarize(spec, cfg, args.n_iter, args.mb, args.beta, args.seed)
        for spec in _method_specs(args.rho_values)
    ]

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
