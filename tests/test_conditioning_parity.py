from __future__ import annotations

from functools import lru_cache

import numpy as np

from mcmc_multiscale.conditioning.constraints import (
    build_A,
    build_c,
    select_conditioning_points,
)
from mcmc_multiscale.conditioning.nullspace import null_basis, project_null
from mcmc_multiscale.conditioning.particular import svd_min_norm
from mcmc_multiscale.conditioning.project import affine_project, stabilize
from mcmc_multiscale.config import Config
from mcmc_multiscale.covariance import exp_covariance
from mcmc_multiscale.field import field_from_theta
from mcmc_multiscale.grid import cell_centered_grid
from mcmc_multiscale.kle import top_eigenpairs
from mcmc_multiscale.subdomain import Subdomain, make_subdomain


@lru_cache(maxsize=1)
def _static_setup() -> (
    tuple[Config, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Subdomain]
):
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    _, _, _, _, global_pts = cell_centered_grid(cfg.nx, cfg.ny)
    C_global = exp_covariance(global_pts, cfg.sigma, cfg.corr_length)
    Phi_global, lambda_global = top_eigenpairs(C_global, cfg.n_global_modes)
    theta_global = rng.standard_normal(cfg.n_global_modes, dtype=np.float64)
    G_old_vec = field_from_theta(Phi_global, lambda_global, theta_global)

    sub = make_subdomain(cfg)
    local_pts = global_pts[sub.local_global_idx, :]
    C_local = exp_covariance(local_pts, cfg.sigma, cfg.corr_length)
    Phi_local, lambda_local = top_eigenpairs(C_local, cfg.Nc + max(cfg.Mb_list))
    return cfg, G_old_vec, local_pts, Phi_local, lambda_local, sub


def test_conditioning_parity_all_Mb() -> None:
    cfg, G_old_vec, local_pts, Phi_local, lambda_local, sub = _static_setup()
    conds: list[float] = []

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

        assert A.shape == (Mb, Next)
        assert info.rankA == Mb
        assert Z.shape == (Next, 30)
        assert null_basis(A).shape == Z.shape

        relative_residual = np.linalg.norm(A @ theta_p - c) / max(
            1.0, np.linalg.norm(c)
        )
        assert relative_residual < 1e-12
        np.testing.assert_allclose(Z.T @ theta_p, np.zeros(Z.shape[1]), atol=1e-12)

        rng = np.random.default_rng(cfg.seed + Mb)
        for _ in range(5):
            eta = rng.standard_normal(Next, dtype=np.float64)
            theta_cond = theta_p + project_null(Z, eta)
            sampled_residual = np.linalg.norm(A @ theta_cond - c) / max(
                1.0, np.linalg.norm(c)
            )
            assert sampled_residual < 1e-12

        conds.append(info.cond_effective)

    assert np.all(np.diff(np.asarray(conds)) >= -1e-10)


def test_affine_projection_and_stabilize() -> None:
    cfg, G_old_vec, local_pts, Phi_local, lambda_local, sub = _static_setup()
    Mb = 8
    Next = cfg.Nc + Mb
    cond_local_idx = select_conditioning_points(
        local_pts, sub.core_local_idx, sub.buffer_local_idx, Mb
    )
    A = build_A(Phi_local[:, :Next], np.sqrt(lambda_local[:Next]), cond_local_idx)
    c = build_c(G_old_vec, sub.local_global_idx, cond_local_idx)
    theta_p, Z, _ = svd_min_norm(A, c)

    rng = np.random.default_rng(cfg.seed)
    theta = rng.standard_normal(Next, dtype=np.float64)
    projected = affine_project(theta, theta_p, Z)
    np.testing.assert_allclose(A @ projected, c, atol=1e-12)
    np.testing.assert_allclose(
        affine_project(projected, theta_p, Z), projected, atol=1e-12
    )

    theta_p_any = theta_p + project_null(Z, theta)
    stabilized = stabilize(theta_p_any, Z)
    np.testing.assert_allclose(stabilized, theta_p, atol=1e-12)
