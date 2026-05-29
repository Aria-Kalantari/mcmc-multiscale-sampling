"""Compare posterior-correct and likelihood-only conditioned sampler paths."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.diagnostics import relative_error  # noqa: E402
from mcmc_multiscale.observations import make_truth  # noqa: E402
from mcmc_multiscale.sampler import (  # noqa: E402
    ConditionedSamplerState,
    RedBlackSamplerState,
    conditioned_sampler,
    red_black_conditioned_sampler,
)


@dataclass(frozen=True)
class RecoverySummary:
    update_scheme: str
    acceptance: str
    n_updates: int
    relative_k_error_posterior_mean: float
    final_theta_norm: float
    max_theta_norm: float
    max_norm_over_expected: float
    misfit_initial: float
    misfit_final: float
    acceptance_rate: float


SamplerState = ConditionedSamplerState | RedBlackSamplerState


def _summarize(
    states: Sequence[SamplerState],
    truth_k: np.ndarray,
    update_scheme: str,
    acceptance: str,
    burn_fraction: float,
) -> RecoverySummary:
    burn = int(len(states) * burn_fraction)
    retained = states[burn:]
    k_posterior_mean = np.mean(
        np.stack([np.exp(state.G_accepted) for state in retained]), axis=0
    )
    candidate_norms = np.asarray(
        [state.theta_norm_candidate for state in states], dtype=np.float64
    )
    accepted = np.asarray([state.accepted for state in states], dtype=bool)
    expected_norm = float(states[-1].expected_norm)
    return RecoverySummary(
        update_scheme=update_scheme,
        acceptance=acceptance,
        n_updates=len(states),
        relative_k_error_posterior_mean=relative_error(k_posterior_mean, truth_k),
        final_theta_norm=float(states[-1].theta_norm_accepted),
        max_theta_norm=float(np.max(candidate_norms)),
        max_norm_over_expected=float(np.max(candidate_norms) / expected_norm),
        misfit_initial=float(-states[0].log_likelihood_accepted),
        misfit_final=float(-states[-1].log_likelihood_accepted),
        acceptance_rate=float(np.mean(accepted)),
    )


def _run_single(
    cfg: Config,
    n_iter: int,
    Mb: int,
    beta: float,
    acceptance: str,
    burn_fraction: float,
) -> RecoverySummary:
    truth = make_truth(cfg, np.random.default_rng(cfg.seed))
    states = list(
        conditioned_sampler(
            cfg=cfg,
            n_iter=n_iter,
            Mb=Mb,
            theta_p_method="svd",
            rng=np.random.default_rng(cfg.seed),
            beta=beta,
            acceptance=acceptance,
        )
    )
    return _summarize(states, truth.k_true, "single", acceptance, burn_fraction)


def _run_red_black(
    cfg: Config,
    n_sweeps: int,
    Mb: int,
    beta: float,
    acceptance: str,
    burn_fraction: float,
) -> RecoverySummary:
    truth = make_truth(cfg, np.random.default_rng(cfg.seed))
    states = list(
        red_black_conditioned_sampler(
            cfg=cfg,
            n_sweeps=n_sweeps,
            Mb=Mb,
            theta_p_method="svd",
            rng=np.random.default_rng(cfg.seed),
            beta=beta,
            acceptance=acceptance,
        )
    )
    return _summarize(states, truth.k_true, "red_black", acceptance, burn_fraction)


def _print_table(rows: Sequence[RecoverySummary], noise_floor: float) -> None:
    print(
        "scheme     acceptance       updates relk_mean final_norm max_norm "
        "max/expected misfit_start misfit_final accept"
    )
    for row in rows:
        print(
            f"{row.update_scheme:<10} {row.acceptance:<17} {row.n_updates:7d} "
            f"{row.relative_k_error_posterior_mean:9.4e} "
            f"{row.final_theta_norm:10.4f} {row.max_theta_norm:8.4f} "
            f"{row.max_norm_over_expected:12.4f} {row.misfit_initial:12.4f} "
            f"{row.misfit_final:12.4f} {row.acceptance_rate:6.3f}"
        )
    print(f"nominal observation-noise misfit floor: {noise_floor:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-iter", type=int, default=300)
    parser.add_argument("--red-black-sweeps", type=int, default=100)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--burn-fraction", type=float, default=1.0 / 3.0)
    args = parser.parse_args()

    if args.burn_fraction < 0.0 or args.burn_fraction >= 1.0:
        raise ValueError("burn-fraction must satisfy 0 <= burn-fraction < 1.")

    cfg = Config(seed=args.seed, beta=args.beta)
    rows = [
        _run_single(
            cfg,
            args.single_iter,
            args.mb,
            args.beta,
            acceptance,
            args.burn_fraction,
        )
        for acceptance in ("posterior", "likelihood_only")
    ]
    rows.extend(
        _run_red_black(
            cfg,
            args.red_black_sweeps,
            args.mb,
            args.beta,
            acceptance,
            args.burn_fraction,
        )
        for acceptance in ("posterior", "likelihood_only")
    )

    print("M8 POSTERIOR RECOVERY BASELINE")
    print(
        f"grid={cfg.ny} x {cfg.nx}; Mb={args.mb}; beta={args.beta}; "
        f"seed={args.seed}; burn_fraction={args.burn_fraction:.3f}"
    )
    print(
        f"single_iter={args.single_iter}; " f"red_black_sweeps={args.red_black_sweeps}"
    )
    print()
    _print_table(rows, noise_floor=0.5 * cfg.n_obs_x * cfg.n_obs_y)
    print()
    print(
        "Interpretation: posterior mode adds the projected global-KLE field "
        "prior and the hard-null pCN proposal correction. This route-(a) "
        "baseline is reported honestly: if the short chain does not recover "
        "well below the likelihood-only error, use longer chains and evaluate "
        "the constraint-manifold route described in SPEC section 3.8(b)."
    )


if __name__ == "__main__":
    main()
