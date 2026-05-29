from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from mcmc_multiscale.config import Config
from mcmc_multiscale.field import reshape_field
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.forward.tpfa import _assemble_system
from mcmc_multiscale.grid import cell_centered_grid


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


def _solve_with_source(cfg: Config, k: np.ndarray, source: np.ndarray) -> np.ndarray:
    matrix, rhs = _assemble_system(k, cfg, source=source)
    return reshape_field(spsolve(matrix, rhs), cfg.ny, cfg.nx)


def test_constant_permeability_matches_linear_profile() -> None:
    cfg = Config(nx=16, ny=12)
    p = ForwardModel(cfg).solve(np.ones((cfg.ny, cfg.nx), dtype=np.float64))
    _, _, X, _, _ = cell_centered_grid(cfg.nx, cfg.ny)

    np.testing.assert_allclose(p, 1.0 - X, atol=1e-12, rtol=0.0)


def test_pressure_shape_and_finiteness() -> None:
    cfg = Config(nx=10, ny=7)
    _, _, X, Y, _ = cell_centered_grid(cfg.nx, cfg.ny)
    k = np.exp(0.2 * np.sin(np.pi * X) * np.cos(np.pi * Y))

    p = ForwardModel(cfg).solve(k)

    assert p.shape == (cfg.ny, cfg.nx)
    assert np.all(np.isfinite(p))


def test_maximum_principle_for_positive_permeability() -> None:
    cfg = Config(nx=18, ny=14)
    _, _, X, Y, _ = cell_centered_grid(cfg.nx, cfg.ny)
    k = np.exp(0.5 * np.sin(2.0 * np.pi * X) * np.cos(np.pi * Y))

    p = ForwardModel(cfg).solve(k)

    assert np.min(p) >= -1e-12
    assert np.max(p) <= 1.0 + 1e-12


def test_sparse_matrix_structure() -> None:
    cfg = Config(nx=5, ny=4)
    k = np.ones((cfg.ny, cfg.nx), dtype=np.float64)

    matrix, rhs = _assemble_system(k, cfg)
    dense_delta = (matrix - matrix.T).toarray()
    coo = matrix.tocoo()
    offdiag = coo.data[coo.row != coo.col]

    assert matrix.shape == (cfg.nx * cfg.ny, cfg.nx * cfg.ny)
    assert rhs.shape == (cfg.nx * cfg.ny,)
    np.testing.assert_allclose(dense_delta, 0.0, atol=1e-14)
    assert np.all(matrix.diagonal() > 0.0)
    assert np.all(offdiag <= 0.0)


def test_manufactured_solution_refinement_convergence() -> None:
    errors: list[float] = []
    for n in (16, 32, 64):
        cfg = Config(nx=n, ny=n)
        p_exact, k, source = _manufactured_fields(cfg)
        p_num = _solve_with_source(cfg, k, source)
        errors.append(float(np.sqrt(np.mean((p_num - p_exact) ** 2))))

    assert errors[2] < errors[1] < errors[0]
    assert errors[1] / errors[0] < 0.35
    assert errors[2] / errors[1] < 0.35
