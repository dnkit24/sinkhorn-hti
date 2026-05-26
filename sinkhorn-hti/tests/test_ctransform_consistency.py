"""Cross-check entropic soft c-transform vs L-BFGS hard c-transform.

Gaussian-to-Gaussian toy with quadratic cost and ``g == 0``: both variants
should match the analytical hard c-transform over the empirical batch.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sinkhorn_hti.config import LBFGSInnerConfig
from sinkhorn_hti.ctransform import (
    entropic_ctransform,
    lbfgs_ctransform,
    soft_argmin,
)


def test_entropic_matches_lbfgs_in_limit():
    B, d = 32, 2
    rng = np.random.default_rng(0)
    y0 = jnp.asarray(rng.standard_normal((B, d)), dtype=jnp.float32)
    y1_pool = jnp.asarray(rng.standard_normal((B, d)), dtype=jnp.float32)
    g_vals = jnp.zeros(B)

    C = 0.5 * jnp.sum((y0[:, None, :] - y1_pool[None, :, :]) ** 2, axis=-1)

    hard = jnp.min(C - g_vals[None, :], axis=-1)

    g_ct_soft = entropic_ctransform(y0, y1_pool, g_vals, C, 1e-5)
    err_soft = float(jnp.max(jnp.abs(g_ct_soft - hard)))
    assert err_soft < 1e-3, f"soft vs hard: max err {err_soft:.3e}"

    def g_fn(y):
        return jnp.zeros(())

    def c_fn(a, b):
        return 0.5 * jnp.sum((b - a) ** 2)

    cfg = LBFGSInnerConfig(maxiter=50, y1_bound=100.0)
    _, ct_vals, _ = lbfgs_ctransform(y0, y0, g_fn, c_fn, cfg)
    # Continuum c-transform value is 0 at these y0 (min at y1 = y0).
    err_lbfgs = float(jnp.max(jnp.abs(ct_vals)))
    assert err_lbfgs < 1e-3, f"lbfgs ct_vals vs 0: max err {err_lbfgs:.3e}"

    T_soft = soft_argmin(y1_pool, g_vals, C, 1e-5)
    argmin_idx = jnp.argmin(C, axis=-1)
    T_hard = y1_pool[argmin_idx]
    err_T = float(jnp.max(jnp.linalg.norm(T_soft - T_hard, axis=-1)))
    assert err_T < 1e-2, f"soft_argmin vs hard: max err {err_T:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
