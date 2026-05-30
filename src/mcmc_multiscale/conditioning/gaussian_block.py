"""Precision-based Gaussian block-conditioning primitives.

Closed-form machinery for the corrected multiscale update derived in
``docs/conditioned_posterior_derivation.md``. The proper, full-rank field prior
is ``G ~ N(0, C_tau)`` with ``C_tau = Phi diag(lam) Phi.T + tau2 I`` (truncated
KLE covariance plus a small nugget ``tau2``). Its precision has a Woodbury
closed form, so we never form an ``n x n`` inverse:

    Q = C_tau^{-1} = (1 / tau2) [ I - Phi diag(d) Phi.T ],   d = lam / (lam + tau2)

with ``Phi`` carrying Euclidean-orthonormal columns (``Phi.T Phi = I``). For a
core block ``S`` (core cells of one coarse subdomain; the cores tile the grid)
the prior full conditional is ``G_S | G_R ~ N(m_S, Q_SS^{-1})`` with the
covariance-weighted regression mean ``m_S = g_S - Q_SS^{-1} (Q g)_S`` and the
Schur-complement covariance ``Q_SS^{-1}``. A pCN proposal around ``m_S`` makes
the block Metropolis acceptance exactly likelihood-only, so a sweep over the
core blocks targets ``pi(G | Y) ~ exp(-Phi_mis) N(0, C_tau)``.

Pure, float64, explicit ``numpy.random.Generator``; no global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


@dataclass(frozen=True)
class PrecisionOperator:
    """Closed-form precision ``Q = C_tau^{-1}`` via Woodbury.

    ``Q = (1 / tau2) [ I - Phi diag(d) Phi.T ]`` with ``d = lam / (lam + tau2)``.
    Both the matvec and the principal-submatrix extraction cost ``O(n N)`` and
    never materialise the dense ``n x n`` operator.
    """

    Phi: np.ndarray  # (n, N) orthonormal KLE eigenvectors
    d: np.ndarray  # (N,) = lam / (lam + tau2)
    tau2: float

    @property
    def n(self) -> int:
        """Number of grid cells (rows of ``Phi``)."""

        return int(self.Phi.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of retained KLE modes (columns of ``Phi``)."""

        return int(self.Phi.shape[1])

    def matvec(self, g: np.ndarray) -> np.ndarray:
        """Return ``Q @ g`` for a vector ``(n,)`` or column stack ``(n, k)``."""

        g_arr = np.asarray(g, dtype=np.float64)
        if g_arr.shape[0] != self.n:
            raise ValueError("g must have one row per grid cell.")
        proj = self.Phi.T @ g_arr
        if proj.ndim == 1:
            proj = self.d * proj
        else:
            proj = self.d[:, np.newaxis] * proj
        return (g_arr - self.Phi @ proj) / self.tau2

    def submatrix(self, idx: np.ndarray) -> np.ndarray:
        """Return the principal submatrix ``Q_SS`` for global indices ``idx``.

        ``Q_SS = (1 / tau2) [ I - Phi_S diag(d) Phi_S.T ]`` is SPD.
        """

        rows = np.asarray(idx, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError("idx must be one-dimensional.")
        Phi_S = self.Phi[rows, :]
        gram = (Phi_S * self.d[np.newaxis, :]) @ Phi_S.T
        return (np.eye(rows.size, dtype=np.float64) - gram) / self.tau2

    def dense(self) -> np.ndarray:
        """Return the dense ``Q`` (diagnostics/tests only)."""

        return self.matvec(np.eye(self.n, dtype=np.float64))


def build_precision(Phi: np.ndarray, lam: np.ndarray, tau2: float) -> PrecisionOperator:
    """Build the closed-form precision of ``C_tau = Phi diag(lam) Phi.T + tau2 I``."""

    Phi_arr = np.asarray(Phi, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    if Phi_arr.ndim != 2:
        raise ValueError("Phi must be two-dimensional.")
    if lam_arr.shape != (Phi_arr.shape[1],):
        raise ValueError("lam must have one entry per Phi column.")
    if np.any(lam_arr < 0.0):
        raise ValueError("KLE eigenvalues must be non-negative.")
    tau2_value = float(tau2)
    if not np.isfinite(tau2_value) or tau2_value <= 0.0:
        raise ValueError("tau2 (block nugget) must be positive.")
    d = lam_arr / (lam_arr + tau2_value)
    return PrecisionOperator(Phi=Phi_arr, d=d, tau2=tau2_value)


@dataclass(frozen=True)
class BlockConditioner:
    """A core block's global cell indices and the Cholesky factor of ``Q_SS``."""

    global_idx: np.ndarray  # (|S|,) global cell indices of the core block
    chol_lower: np.ndarray  # lower-triangular L with Q_SS = L L.T

    @property
    def size(self) -> int:
        """Number of cells in the core block."""

        return int(self.global_idx.size)


def make_block_conditioner(
    precision: PrecisionOperator, global_idx: np.ndarray
) -> BlockConditioner:
    """Build a block conditioner: indices plus the Cholesky factor of ``Q_SS``."""

    idx = np.asarray(global_idx, dtype=np.int64)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("global_idx must be a non-empty one-dimensional array.")
    Q_SS = precision.submatrix(idx)
    chol_lower = np.linalg.cholesky(Q_SS)
    return BlockConditioner(global_idx=idx, chol_lower=chol_lower)


def conditional_mean(
    precision: PrecisionOperator, block: BlockConditioner, g: np.ndarray
) -> np.ndarray:
    """Return the prior full-conditional mean ``m_S = g_S - Q_SS^{-1} (Q g)_S``.

    This equals the covariance-weighted regression ``-Q_SS^{-1} Q_{S,R} g_R`` of
    the core block on the frozen exterior, without forming ``Q_{S,R}``.
    """

    g_arr = np.asarray(g, dtype=np.float64)
    q_g = precision.matvec(g_arr)
    rhs = q_g[block.global_idx]
    correction = cho_solve((block.chol_lower, True), rhs)
    return g_arr[block.global_idx] - correction


def propose_block(
    precision: PrecisionOperator,
    block: BlockConditioner,
    g: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a pCN-around-conditional proposal for the core block values.

    ``g_S' = m_S + sqrt(1 - beta^2) (g_S - m_S) + beta w`` with the conditional
    draw ``w ~ N(0, Q_SS^{-1})`` formed as ``w = L^{-T} zeta``, ``zeta ~ N(0, I)``.
    The exterior enters only through ``m_S``; holding it fixed makes the move
    reversible w.r.t. the base Gaussian ``N(m_S, Q_SS^{-1})``, so the block MH
    acceptance is exactly likelihood-only.
    """

    beta_value = float(beta)
    if beta_value <= 0.0 or beta_value > 1.0:
        raise ValueError("block pCN beta must satisfy 0 < beta <= 1.")
    g_arr = np.asarray(g, dtype=np.float64)
    m_s = conditional_mean(precision, block, g_arr)
    g_s = g_arr[block.global_idx]
    zeta = rng.standard_normal(block.size, dtype=np.float64)
    w = solve_triangular(block.chol_lower, zeta, lower=True, trans="T")
    return m_s + np.sqrt(1.0 - beta_value**2) * (g_s - m_s) + beta_value * w
