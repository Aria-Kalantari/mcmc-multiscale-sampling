"""M2 forward/Bayes sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcmc_multiscale.bayes import misfit  # noqa: E402
from mcmc_multiscale.config import Config  # noqa: E402
from mcmc_multiscale.covariance import exp_covariance  # noqa: E402
from mcmc_multiscale.field import reshape_field  # noqa: E402
from mcmc_multiscale.forward import ForwardModel  # noqa: E402
from mcmc_multiscale.forward.tpfa import _assemble_system  # noqa: E402
from mcmc_multiscale.grid import cell_centered_grid  # noqa: E402
from mcmc_multiscale.kle import top_eigenpairs  # noqa: E402
from mcmc_multiscale.observations import make_truth  # noqa: E402


def _manufactured_fields(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, X, Y, _ = cell_centered_grid(cfg.nx, cfg.ny)
    pi = np.pi
    amp_p = 0.05
    amp_k = 0.25

    p = 1.0 - X + amp_p * np.sin(pi * X) * np.cos(2.0 * pi * Y)
    k = np.exp(amp_k * np.sin(pi * X) * np.cos(pi * Y))

    p_x = -1.0 + amp_p * pi * np.cos(pi * X) * np.cos(2.0 * pi * Y)
    p_xx = -(amp_p * pi**2) * np.sin(pi * X) * np.cos(2.0 * pi * Y)
    p_y = -(2.0 * amp_p * pi) * np.sin(pi * X) * np.sin(2.0 * pi * Y)
    p_yy = -(4.0 * amp_p * pi**2) * np.sin(pi * X) * np.cos(2.0 * pi * Y)

    k_x = k * amp_k * pi * np.cos(pi * X) * np.cos(pi * Y)
    k_y = -k * amp_k * pi * np.sin(pi * X) * np.sin(pi * Y)
    source = -(k_x * p_x + k * p_xx + k_y * p_y + k * p_yy)
    return p.astype(np.float64), k.astype(np.float64), source.astype(np.float64)


def _constant_k_error() -> float:
    cfg = Config(nx=48, ny=48)
    p = ForwardModel(cfg).solve(np.ones((cfg.ny, cfg.nx), dtype=np.float64))
    _, _, X, _, _ = cell_centered_grid(cfg.nx, cfg.ny)
    return float(np.max(np.abs(p - (1.0 - X))))


def _manufactured_errors() -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for n in (16, 32, 64):
        cfg = Config(nx=n, ny=n)
        p_exact, k, source = _manufactured_fields(cfg)
        matrix, rhs = _assemble_system(k, cfg, source=source)
        p_num = reshape_field(spsolve(matrix, rhs), cfg.ny, cfg.nx)
        error = float(np.sqrt(np.mean((p_num - p_exact) ** 2)))
        rows.append((n, error))
    return rows


def _noiseless_truth_misfit() -> float:
    cfg = Config(nx=8, ny=8, n_global_modes=10, n_obs_x=3, n_obs_y=3, seed=11)
    truth = make_truth(cfg)
    _, _, _, _, points = cell_centered_grid(cfg.nx, cfg.ny)
    C = exp_covariance(points, cfg.sigma, cfg.corr_length)
    Phi, lam = top_eigenpairs(C, cfg.n_global_modes)
    return misfit(
        truth.theta_true,
        Phi,
        lam,
        ForwardModel(cfg),
        truth.y_clean,
        truth.sensor_idx,
        cfg.sigma_obs,
        cfg.ny,
        cfg.nx,
    )


def main() -> None:
    print("M2 FORWARD/BAYES SANITY")
    print(f"Constant-k max error: {_constant_k_error():.6e}")
    print("Manufactured L2 errors:")
    for n, error in _manufactured_errors():
        print(f"  n={n:2d}: {error:.6e}")
    print(f"Noiseless truth misfit: {_noiseless_truth_misfit():.6e}")


if __name__ == "__main__":
    main()
