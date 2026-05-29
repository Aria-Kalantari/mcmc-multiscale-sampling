"""M7 red-black checkerboard sweep demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.sampler import red_black_conditioned_sampler  # noqa: E402
from mcmc_multiscale.subdomain import red_black_order  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sweeps", type=int, default=3)
    parser.add_argument("--mb", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--theta-p-method", type=str, default="svd")
    parser.add_argument("--conditioning-mode", type=str, default="hard")
    parser.add_argument("--rhs-mode", type=str, default="data")
    parser.add_argument("--proposal", type=str, default="pcn")
    parser.add_argument("--rho", type=float, default=None)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, beta=args.beta)
    states = list(
        red_black_conditioned_sampler(
            cfg=cfg,
            n_sweeps=args.n_sweeps,
            Mb=args.mb,
            theta_p_method=args.theta_p_method,
            rng=np.random.default_rng(args.seed),
            beta=args.beta,
            proposal=args.proposal,
            rhs_mode=args.rhs_mode,
            conditioning_mode=args.conditioning_mode,
            rho=args.rho,
        )
    )

    order = red_black_order(cfg)
    colors_seen = sorted({state.color for state in states})
    subdomains_seen = {(state.subdomain_row, state.subdomain_col) for state in states}
    all_subdomains_seen = len(subdomains_seen) == cfg.n_coarse_x * cfg.n_coarse_y
    both_colors_seen = colors_seen == [0, 1]

    accepted = np.asarray([state.accepted for state in states], dtype=bool)
    residuals = np.asarray(
        [state.constraint_residual_candidate for state in states], dtype=np.float64
    )
    jumps = np.asarray(
        [state.interface_jump_accepted for state in states], dtype=np.float64
    )
    theta_norms = np.asarray(
        [state.theta_norm_candidate for state in states], dtype=np.float64
    )
    expected_norm = float(states[-1].expected_norm)
    rel_k_final = states[-1].relative_k_error_accepted

    print("M7 RED-BLACK CHECKERBOARD UPDATES")
    print(
        f"coarse partition: {cfg.n_coarse_y} x {cfg.n_coarse_x}; "
        f"subdomains={cfg.n_coarse_x * cfg.n_coarse_y}"
    )
    print(f"color 0 subdomains={len(order[0])}; color 1 subdomains={len(order[1])}")
    print(
        f"n_sweeps={args.n_sweeps}; total local updates={len(states)}; "
        f"Mb={args.mb}; beta={args.beta}; seed={args.seed}"
    )
    print(
        f"method={args.theta_p_method}; conditioning={args.conditioning_mode}; "
        f"rhs={args.rhs_mode}; proposal={args.proposal}"
    )
    print()
    print(f"acceptance rate: {float(np.mean(accepted)):.4f}")
    print(f"final relative k error: {rel_k_final:.4e}")
    print(
        "constraint residual mean/max: "
        f"{float(np.mean(residuals)):.4e} / {float(np.max(residuals)):.4e}"
    )
    print(f"accepted interface jump mean: {float(np.mean(jumps)):.4e}")
    print(
        "candidate theta norm mean/max: "
        f"{float(np.mean(theta_norms)):.4f} / {float(np.max(theta_norms)):.4f}"
    )
    print(
        "max candidate theta norm / expected norm: "
        f"{float(np.max(theta_norms)) / expected_norm:.4f}"
    )
    print(f"expected Gaussian norm: {expected_norm:.4f}")
    print(f"both colors updated: {both_colors_seen}")
    print(f"all subdomains updated: {all_subdomains_seen}")
    print()
    print(
        "Schedule note: this is a deterministic sequential frozen-snapshot "
        "schedule. During each color pass, each subdomain builds its "
        "conditioning RHS from the same frozen global field. With overlap, "
        "same-color diagonal regions can still be coupled, so 2-color "
        "checkerboarding is not an exact parallel-independence guarantee."
    )


if __name__ == "__main__":
    main()
