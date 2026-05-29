"""M4 LU/pivot repeated-conditioning instability reproduction."""

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
class RunSummary:
    method: str
    n_iter: int
    Mb: int
    beta: float
    expected_norm: float
    candidate_initial: float
    candidate_final: float
    candidate_max: float
    accepted_initial: float
    accepted_final: float
    accepted_max: float
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
    cond_A_mean: float
    cond_A_max: float
    cond_B_mean: float | None
    cond_B_max: float | None


def _summarize(
    method: str, cfg: Config, n_iter: int, Mb: int, beta: float, seed: int
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
    cond_A: list[float] = []
    cond_B: list[float] = []
    expected_norm = np.nan

    for state in conditioned_sampler(
        cfg=cfg,
        n_iter=n_iter,
        Mb=Mb,
        theta_p_method=method,
        rng=np.random.default_rng(seed),
        beta=beta,
        update_scheme="single",
        proposal="pcn",
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
        cond_A.append(state.cond_A)
        if state.cond_B is not None:
            cond_B.append(state.cond_B)

    cand = np.asarray(candidate_norms, dtype=np.float64)
    acc = np.asarray(accepted_norms, dtype=np.float64)
    accepted_arr = np.asarray(accepted, dtype=bool)
    residual_arr = np.asarray(residuals, dtype=np.float64)
    cand_jump = np.asarray(candidate_jumps, dtype=np.float64)
    acc_jump = np.asarray(accepted_jumps, dtype=np.float64)
    cand_err = np.asarray(candidate_errors, dtype=np.float64)
    acc_err = np.asarray(accepted_errors, dtype=np.float64)
    hidden = np.asarray(hidden_norms, dtype=np.float64)
    cond_a_arr = np.asarray(cond_A, dtype=np.float64)
    cond_b_arr = np.asarray(cond_B, dtype=np.float64)

    return RunSummary(
        method=method,
        n_iter=n_iter,
        Mb=Mb,
        beta=beta,
        expected_norm=float(expected_norm),
        candidate_initial=float(cand[0]),
        candidate_final=float(cand[-1]),
        candidate_max=float(np.max(cand)),
        accepted_initial=float(acc[0]),
        accepted_final=float(acc[-1]),
        accepted_max=float(np.max(acc)),
        acceptance_rate=float(np.mean(accepted_arr)),
        residual_mean=float(np.mean(residual_arr)),
        residual_max=float(np.max(residual_arr)),
        jump_candidate_mean=float(np.mean(cand_jump)),
        jump_accepted_mean=float(np.mean(acc_jump)),
        rel_k_candidate_mean=float(np.mean(cand_err)),
        rel_k_candidate_final=float(cand_err[-1]),
        rel_k_accepted_mean=float(np.mean(acc_err)),
        rel_k_accepted_final=float(acc_err[-1]),
        hidden_mean=float(np.mean(hidden)),
        hidden_max=float(np.max(hidden)),
        cond_A_mean=float(np.mean(cond_a_arr)),
        cond_A_max=float(np.max(cond_a_arr)),
        cond_B_mean=None if cond_b_arr.size == 0 else float(np.mean(cond_b_arr)),
        cond_B_max=None if cond_b_arr.size == 0 else float(np.max(cond_b_arr)),
    )


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4e}"


def _print_summary(lu: RunSummary, svd: RunSummary) -> None:
    tiny = np.finfo(np.float64).tiny
    candidate_ratio = lu.candidate_max / max(svd.candidate_max, tiny)
    accepted_ratio = lu.accepted_max / max(svd.accepted_max, tiny)
    candidate_triggered = candidate_ratio > 2.0

    print("M4 REPEATED-CONDITIONING INSTABILITY")
    print(f"n_iter={lu.n_iter}, Mb={lu.Mb}, beta={lu.beta}, seed paired by method")
    print(f"Expected Gaussian norm: {lu.expected_norm:.4f}")
    print()
    print(
        "method  cand_init  cand_final  cand_max  acc_init  acc_final  "
        "acc_max  accept  hidden_mean  hidden_max"
    )
    for row in (lu, svd):
        print(
            f"{row.method:>6} "
            f"{row.candidate_initial:10.4f} "
            f"{row.candidate_final:11.4f} "
            f"{row.candidate_max:9.4f} "
            f"{row.accepted_initial:9.4f} "
            f"{row.accepted_final:10.4f} "
            f"{row.accepted_max:8.4f} "
            f"{row.acceptance_rate:7.3f} "
            f"{row.hidden_mean:12.4e} "
            f"{row.hidden_max:11.4e}"
        )
    print()
    print(f"LU/SVD max candidate theta norm ratio: {candidate_ratio:.4f}")
    print(f"LU/SVD max accepted theta norm ratio: {accepted_ratio:.4f}")
    print(
        "Candidate norm / expected norm: "
        f"LU={lu.candidate_max / lu.expected_norm:.4f}, "
        f"SVD={svd.candidate_max / svd.expected_norm:.4f}"
    )
    print(
        "Accepted norm / expected norm: "
        f"LU={lu.accepted_max / lu.expected_norm:.4f}, "
        f"SVD={svd.accepted_max / svd.expected_norm:.4f}"
    )
    print()
    print("Constraint residuals:")
    print(
        f"  LU mean/max={lu.residual_mean:.4e}/{lu.residual_max:.4e}; "
        f"SVD mean/max={svd.residual_mean:.4e}/{svd.residual_max:.4e}"
    )
    print("Interface jumps:")
    print(
        f"  LU candidate/accepted mean={lu.jump_candidate_mean:.4e}/"
        f"{lu.jump_accepted_mean:.4e}; "
        f"SVD candidate/accepted mean={svd.jump_candidate_mean:.4e}/"
        f"{svd.jump_accepted_mean:.4e}"
    )
    print("Relative k errors:")
    print(
        f"  LU candidate mean/final={lu.rel_k_candidate_mean:.4e}/"
        f"{lu.rel_k_candidate_final:.4e}; accepted mean/final="
        f"{lu.rel_k_accepted_mean:.4e}/{lu.rel_k_accepted_final:.4e}"
    )
    print(
        f"  SVD candidate mean/final={svd.rel_k_candidate_mean:.4e}/"
        f"{svd.rel_k_candidate_final:.4e}; accepted mean/final="
        f"{svd.rel_k_accepted_mean:.4e}/{svd.rel_k_accepted_final:.4e}"
    )
    print("Conditioning matrix summaries:")
    print(
        f"  LU cond(A) mean/max={lu.cond_A_mean:.4e}/{lu.cond_A_max:.4e}; "
        f"cond(B) mean/max={_fmt_optional(lu.cond_B_mean)}/"
        f"{_fmt_optional(lu.cond_B_max)}"
    )
    print(
        f"  SVD cond(A) mean/max={svd.cond_A_mean:.4e}/{svd.cond_A_max:.4e}; "
        f"cond(B) mean/max={_fmt_optional(svd.cond_B_mean)}/"
        f"{_fmt_optional(svd.cond_B_max)}"
    )
    print()
    if candidate_triggered:
        print("Instability criterion triggered: LU max candidate norm / SVD > 2.0")
    else:
        print(
            "Instability criterion not triggered in this run. Increase n_iter or "
            "decrease beta to expose slower LU accumulation."
        )
    if candidate_triggered and accepted_ratio <= 2.0:
        print(
            "Generation-level instability observed; MH rejection suppresses accepted-chain drift."
        )
    elif accepted_ratio > 2.0:
        print("Accepted-chain LU drift is visible in this run.")
    else:
        print("Accepted-chain drift was not visible in this run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, beta=args.beta)
    lu = _summarize("lu", cfg, args.n_iter, args.mb, args.beta, args.seed)
    svd = _summarize("svd", cfg, args.n_iter, args.mb, args.beta, args.seed)
    _print_summary(lu, svd)


if __name__ == "__main__":
    main()
