"""M14(b) regression tests for red_black_conditioned_sampler cond_refresh_period.

The ``cond_refresh_period`` parameter rebuilds the conditioning RHS ``c`` and
particular solution ``theta_p`` only every ``K`` sweeps. Its default ``K=1`` must
reproduce the pre-change sampler bit-for-bit; the golden fixture
``tests/data/m14_red_black_golden.npz`` was captured from the pristine sampler
before the parameter was added. A ``K>1`` run must genuinely differ once the
field has evolved, and non-positive ``K`` must be rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcmc_multiscale.config import Config
from mcmc_multiscale.sampler import red_black_conditioned_sampler

_FIXTURE = Path(__file__).parent / "data" / "m14_red_black_golden.npz"

# MUST match the config used to generate the golden fixture.
_N_SWEEPS = 3
_MB = 2


def _golden_config() -> Config:
    return Config(
        nx=12,
        ny=12,
        n_coarse_x=4,
        n_coarse_y=4,
        overlap_cells=1,
        n_global_modes=8,
        Nc=4,
        n_obs_x=3,
        n_obs_y=3,
        sigma_obs=1.0e6,
        beta=0.1,
        seed=13,
    )


def _trajectory(acceptance: str, prior_mode: str | None, **kwargs) -> dict:
    cfg = _golden_config()
    states = list(
        red_black_conditioned_sampler(
            cfg,
            n_sweeps=_N_SWEEPS,
            Mb=_MB,
            theta_p_method="svd",
            rng=np.random.default_rng(cfg.seed),
            beta=cfg.beta,
            acceptance=acceptance,
            prior_mode=prior_mode,
            **kwargs,
        )
    )
    return {
        "G": np.stack([s.G_accepted for s in states]).astype(np.float64),
        "accepted": np.asarray([s.accepted for s in states], dtype=bool),
        "relk": np.asarray(
            [s.relative_k_error_accepted for s in states], dtype=np.float64
        ),
        "theta_norm": np.asarray(
            [s.theta_norm_accepted for s in states], dtype=np.float64
        ),
    }


def _assert_matches_fixture(traj: dict, fx, prefix: str) -> None:
    np.testing.assert_array_equal(traj["G"], fx[f"{prefix}_G"])
    np.testing.assert_array_equal(traj["accepted"], fx[f"{prefix}_accepted"])
    np.testing.assert_array_equal(traj["relk"], fx[f"{prefix}_relk"])
    np.testing.assert_array_equal(traj["theta_norm"], fx[f"{prefix}_theta_norm"])


def test_cond_refresh_period_default_reproduces_golden_fixture() -> None:
    """Default (and explicit K=1) reproduce the pristine sampler bit-for-bit."""
    fx = np.load(_FIXTURE)

    # Default call omits cond_refresh_period entirely.
    _assert_matches_fixture(_trajectory("likelihood_only", None), fx, "ll")
    _assert_matches_fixture(_trajectory("posterior", "global_field"), fx, "post")

    # Explicit K=1 is the same code path and must also match exactly.
    _assert_matches_fixture(
        _trajectory("likelihood_only", None, cond_refresh_period=1), fx, "ll"
    )
    _assert_matches_fixture(
        _trajectory("posterior", "global_field", cond_refresh_period=1), fx, "post"
    )


def test_cond_refresh_period_changes_conditioning_when_greater_than_one() -> None:
    """K>1 freezes c/theta_p across sweeps, so the trajectory diverges from K=1.

    Sweep 1 is identical (both refresh on the first sweep); once the accepted
    field has evolved, the frozen RHS differs from the rebuilt one.
    """
    cfg = _golden_config()
    n_sweeps = 4
    n_sub = cfg.n_coarse_x * cfg.n_coarse_y
    common = dict(
        n_sweeps=n_sweeps,
        Mb=_MB,
        theta_p_method="svd",
        beta=cfg.beta,
    )

    states_k1 = list(
        red_black_conditioned_sampler(
            cfg, rng=np.random.default_rng(cfg.seed), cond_refresh_period=1, **common
        )
    )
    states_k4 = list(
        red_black_conditioned_sampler(
            cfg, rng=np.random.default_rng(cfg.seed), cond_refresh_period=4, **common
        )
    )

    g_k1 = np.stack([s.G_accepted for s in states_k1])
    g_k4 = np.stack([s.G_accepted for s in states_k4])

    # First sweep identical.
    np.testing.assert_array_equal(g_k1[:n_sub], g_k4[:n_sub])
    # Later sweeps differ (the parameter actually changes conditioning).
    assert not np.array_equal(g_k1[n_sub:], g_k4[n_sub:])


def test_cond_refresh_period_rejects_nonpositive() -> None:
    """cond_refresh_period < 1 raises before any sampling work."""
    cfg = _golden_config()
    with pytest.raises(ValueError, match="cond_refresh_period"):
        list(
            red_black_conditioned_sampler(
                cfg,
                n_sweeps=1,
                Mb=_MB,
                theta_p_method="svd",
                rng=np.random.default_rng(cfg.seed),
                cond_refresh_period=0,
            )
        )
