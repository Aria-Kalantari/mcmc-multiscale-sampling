"""Hard conditioning constraint construction."""

from __future__ import annotations

import numpy as np


def _matlab_round_positive(values: np.ndarray) -> np.ndarray:
    """Round positive values the way MATLAB `round` does."""

    return np.floor(values + 0.5).astype(np.int64)


def _unique_stable(values: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    out: list[int] = []
    for value in values.tolist():
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return np.asarray(out, dtype=np.int64)


def select_conditioning_points(
    local_pts: np.ndarray,
    core_local_idx: np.ndarray,
    buffer_local_idx: np.ndarray,
    Mb: int,
) -> np.ndarray:
    """Select `Mb` buffer points by angular order around the core centroid.

    This is a direct port of MATLAB `selectConditioningPoints`: order buffer
    cells by polar angle about the core centroid, spread positions with
    `round(linspace(...))`, keep unique positions stably, then top up from the
    remaining ordered buffer cells if rounding produced duplicates.
    """

    pts = np.asarray(local_pts, dtype=np.float64)
    core_idx = np.asarray(core_local_idx, dtype=np.int64)
    buffer_idx = np.asarray(buffer_local_idx, dtype=np.int64)
    if Mb < 0:
        raise ValueError("Mb must be nonnegative.")
    if Mb > buffer_idx.size:
        raise ValueError("Mb is larger than the number of available buffer cells.")
    if Mb == 0:
        return np.empty(0, dtype=np.int64)

    center = np.mean(pts[core_idx, :], axis=0)
    buffer_pts = pts[buffer_idx, :]
    angles = np.arctan2(buffer_pts[:, 1] - center[1], buffer_pts[:, 0] - center[0])
    order = np.argsort(angles, kind="mergesort")
    ordered_buffer = buffer_idx[order]

    raw_positions = _matlab_round_positive(
        np.linspace(1, ordered_buffer.size, Mb + 2, dtype=np.float64)
    )[1:-1]
    raw_positions = _unique_stable(raw_positions)

    while raw_positions.size < Mb:
        used = set(raw_positions.tolist())
        missing = [idx for idx in range(1, ordered_buffer.size + 1) if idx not in used]
        raw_positions = np.append(raw_positions, missing[0])

    return ordered_buffer[raw_positions[:Mb] - 1].astype(np.int64)


def build_A(
    PhiExt: np.ndarray, sqrt_lam: np.ndarray, cond_local_idx: np.ndarray
) -> np.ndarray:
    """Build `A[j, ell] = sqrt(lambda_ell) * phi_ell(y_j)`."""

    Phi_arr = np.asarray(PhiExt, dtype=np.float64)
    sqrt_lam_arr = np.asarray(sqrt_lam, dtype=np.float64)
    cond_idx = np.asarray(cond_local_idx, dtype=np.int64)
    if Phi_arr.ndim != 2:
        raise ValueError("PhiExt must be two-dimensional.")
    if sqrt_lam_arr.shape != (Phi_arr.shape[1],):
        raise ValueError("sqrt_lam must have one entry per PhiExt column.")
    return Phi_arr[cond_idx, :] * sqrt_lam_arr[np.newaxis, :]


def build_c(
    G_old_vec: np.ndarray, local_global_idx: np.ndarray, cond_local_idx: np.ndarray
) -> np.ndarray:
    """Build the frozen-neighbor values `c` at conditioning points."""

    G_vec = np.asarray(G_old_vec, dtype=np.float64)
    local_idx = np.asarray(local_global_idx, dtype=np.int64)
    cond_idx = np.asarray(cond_local_idx, dtype=np.int64)
    return G_vec[local_idx[cond_idx]]
