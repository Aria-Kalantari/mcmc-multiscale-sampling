"""M14(a) tests + exp12 diagnostic-helper tests.

The headline gate is exact and deterministic: standardization is scale-invariant
in the KLE amplitude, so the standardized misfit is invariant under theta ->
a*theta. That proves the standardization probe does NOT target pi(G|Y) -- it
makes the likelihood blind to ||theta||, exactly the scale the drift lives in.
All of this is diagnostic-only (SPEC 12); no recovery is claimed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.forward import ForwardModel
from mcmc_multiscale.observations import make_truth

from experiments.exp12_conditioning_diagnostics import (
    _field_misfit,
    _global_kle,
    _reversal_onset,
    _standardize_log_field,
    _standardized_misfit,
    _sweep_relk,
)


def test_standardize_log_field_is_zero_mean_unit_std() -> None:
    """The probe returns a float64 z-score field (mean 0, std 1)."""
    rng = np.random.default_rng(3)
    G = rng.standard_normal((5, 7)) * 2.5 + 1.3
    standardized = _standardize_log_field(G)
    assert standardized.dtype == np.float64
    assert abs(float(standardized.mean())) < 1e-12
    assert abs(float(standardized.std()) - 1.0) < 1e-12


def test_standardize_log_field_rejects_constant_field() -> None:
    """A constant field has zero std and cannot be standardized."""
    with pytest.raises(ValueError, match="constant"):
        _standardize_log_field(np.full((4, 4), 2.0))


def test_standardization_makes_likelihood_scale_invariant_diagnostic_only() -> None:
    """DIAGNOSTIC ONLY: the standardized misfit is invariant under theta->a*theta.

    Since standardize(a*G)=standardize(G) for a>0 and G=Phi sqrt(lam) theta, the
    data cannot see the KLE amplitude. The standardized posterior reverts to the
    prior in the amplitude direction, so it does NOT target pi(G|Y). This is the
    exact gate; it needs no Monte Carlo and no tuned threshold.
    """
    cfg = Config(nx=10, ny=10, n_global_modes=12, n_obs_x=4, n_obs_y=4)
    truth = make_truth(cfg, np.random.default_rng(1))
    Phi, lam = _global_kle(cfg)
    fwd = ForwardModel(cfg)
    theta = np.random.default_rng(2).standard_normal(cfg.n_global_modes)

    base = _standardized_misfit(
        theta,
        Phi,
        lam,
        fwd,
        truth.y_obs,
        truth.sensor_idx,
        cfg.sigma_obs,
        cfg.ny,
        cfg.nx,
    )
    for a in (0.25, 0.5, 2.0, 4.0):
        scaled = _standardized_misfit(
            a * theta,
            Phi,
            lam,
            fwd,
            truth.y_obs,
            truth.sensor_idx,
            cfg.sigma_obs,
            cfg.ny,
            cfg.nx,
        )
        np.testing.assert_allclose(scaled, base, rtol=1e-8, atol=1e-6)

    # Sanity: WITHOUT standardization the misfit does depend on the amplitude,
    # so the invariance above is a real property of the probe, not of the setup.
    m1 = _field_misfit(
        theta,
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
    m2 = _field_misfit(
        2.0 * theta,
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
    assert abs(m1 - m2) > 1.0


def test_standardization_chain_mean_differs_from_analytic_posterior() -> None:
    """Secondary (not the gate): a chain on the standardized target does not

    recover the analytic linear-Gaussian posterior. Standardization is
    scale-blind, so the amplitude direction reverts to the prior and the
    posterior mean lands far from mu_post.
    """
    rng = np.random.default_rng(0)
    N, n_cell, n_obs = 4, 20, 12
    M = rng.standard_normal((n_cell, N))  # field = M @ theta (linear)
    L = rng.standard_normal((n_obs, n_cell))  # linear observation of the field
    sig = 0.3
    theta_true = rng.standard_normal(N)
    a_eff = L @ M  # obs = a_eff @ theta
    y = a_eff @ theta_true + sig * rng.standard_normal(n_obs)

    sigma_post = np.linalg.inv(np.eye(N) + a_eff.T @ a_eff / sig**2)
    mu_post = sigma_post @ (a_eff.T @ y) / sig**2

    def log_target_std(theta: np.ndarray) -> float:
        G = M @ theta
        std = G.std()
        field = (G - G.mean()) / std
        resid = L @ field - y
        return float(-0.5 * theta @ theta - 0.5 / sig**2 * resid @ resid)

    mh_rng = np.random.default_rng(1)
    theta = np.array([1.0, 0.0, 0.0, 0.0])  # non-constant field start
    logp = log_target_std(theta)
    step = 0.3
    n_iter = 12000
    samples = np.empty((n_iter, N))
    for i in range(n_iter):
        prop = theta + step * mh_rng.standard_normal(N)
        logp_prop = log_target_std(prop)
        if np.log(mh_rng.uniform()) < logp_prop - logp:
            theta, logp = prop, logp_prop
        samples[i] = theta
    mean_std = samples[n_iter // 2 :].mean(axis=0)

    assert not np.allclose(mean_std, mu_post, atol=0.05)
    assert np.max(np.abs(mean_std - mu_post)) > 0.1


def test_reversal_onset_detects_rise_after_minimum() -> None:
    """Onset is the first sweep that rises past an interior running minimum."""
    assert _reversal_onset(np.array([0.9, 0.6, 0.46, 0.5, 0.7, 0.88])) == 4
    # monotone descending -> minimum at the end -> no reversal
    assert _reversal_onset(np.array([0.9, 0.7, 0.5, 0.4, 0.3])) is None
    # monotone rising -> minimum at the start -> no reversal
    assert _reversal_onset(np.array([0.3, 0.5, 0.7])) is None
    # too short
    assert _reversal_onset(np.array([0.5, 0.4])) is None


def test_sweep_relk_extracts_one_value_per_sweep() -> None:
    """_sweep_relk returns the last accepted rel-k of each sweep, in order."""
    states = [
        SimpleNamespace(sweep=s, relative_k_error_accepted=float(v))
        for s, v in [(1, 0.9), (1, 0.8), (2, 0.7), (2, 0.6), (3, 0.55)]
    ]
    np.testing.assert_array_equal(_sweep_relk(states), np.array([0.8, 0.6, 0.55]))
