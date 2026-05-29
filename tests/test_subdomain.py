from __future__ import annotations

from mcmc_multiscale.config import Config
from mcmc_multiscale.subdomain import make_subdomain


def test_default_subdomain_cell_counts() -> None:
    sub = make_subdomain(Config())

    assert sub.core_local_idx.size == 144
    assert sub.local_global_idx.size == 256
    assert sub.buffer_local_idx.size == 112


def test_default_subdomain_zero_based_bounds() -> None:
    sub = make_subdomain(Config())

    assert sub.core_rows[0] == 12
    assert sub.core_rows[-1] == 23
    assert sub.core_cols[0] == 12
    assert sub.core_cols[-1] == 23
    assert sub.hat_rows[0] == 10
    assert sub.hat_rows[-1] == 25
    assert sub.hat_cols[0] == 10
    assert sub.hat_cols[-1] == 25
