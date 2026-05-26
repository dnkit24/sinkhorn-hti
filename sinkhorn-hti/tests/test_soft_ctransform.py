"""Correctness tests for the soft c-transform (convergence, argmin, stability)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sinkhorn_hti.ctransform import entropic_ctransform, soft_argmin


def _hard_ctransform_from_matrix(
    g_values: jnp.ndarray,     # (B1,)
    cost_matrix: jnp.ndarray,  # (B0, B1)
) -> jnp.ndarray:
    return jnp.min(cost_matrix - g_values[None, :], axis=-1)


def test_soft_recovers_hard_in_eps_limit():
    """``soft <= hard`` (Jensen) and ``soft -> hard`` as ``eps -> 0``."""
    rng = np.random.default_rng(0)
    B0, B1 = 16, 32
    y0 = jnp.asarray(rng.standard_normal((B0, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((B1, 2)), dtype=jnp.float32)
    g = jnp.asarray(rng.standard_normal((B1,)), dtype=jnp.float32)

    C = jnp.sum((y0[:, None, :] - y1[None, :, :]) ** 2, axis=-1)

    hard = _hard_ctransform_from_matrix(g, C)
    for eps in [1.0, 0.1, 0.01, 1e-3, 1e-4]:
        soft = entropic_ctransform(y0, y1, g, C, eps)
        err = jnp.max(jnp.abs(soft - hard))
        assert jnp.all(hard - soft >= -1e-5), f"eps={eps}: soft exceeded hard."
        if eps <= 1e-3:
            assert err < 1e-2, f"eps={eps}: max-err={err:.4e} too large."


def test_soft_convergence_rate():
    """Monotone decrease of the hard-soft gap as eps decreases."""
    rng = np.random.default_rng(1)
    B0, B1 = 8, 16
    y0 = jnp.asarray(rng.standard_normal((B0, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((B1, 2)), dtype=jnp.float32)
    g = jnp.asarray(rng.standard_normal((B1,)), dtype=jnp.float32)
    C = jnp.sum((y0[:, None, :] - y1[None, :, :]) ** 2, axis=-1)
    hard = _hard_ctransform_from_matrix(g, C)

    eps_grid = [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3]
    gaps = []
    for eps in eps_grid:
        soft = entropic_ctransform(y0, y1, g, C, eps)
        gaps.append(float(jnp.mean(jnp.abs(hard - soft))))
    for a, b in zip(gaps[:-1], gaps[1:], strict=True):
        assert b <= a + 1e-6, f"gap not monotone: {a:.4e} -> {b:.4e}"
    assert gaps[-1] < gaps[0] / 10.0


def test_soft_argmin_matches_argmin_at_small_eps():
    rng = np.random.default_rng(2)
    B0, B1 = 4, 24
    y0 = jnp.asarray(rng.standard_normal((B0, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((B1, 2)), dtype=jnp.float32)
    g = jnp.asarray(rng.standard_normal((B1,)), dtype=jnp.float32)
    C = jnp.sum((y0[:, None, :] - y1[None, :, :]) ** 2, axis=-1)

    hard_argmin_idx = jnp.argmin(C - g[None, :], axis=-1)
    hard_argmin_pt = y1[hard_argmin_idx]

    soft_map = soft_argmin(y1, g, C, epsilon=1e-4)
    err = float(jnp.max(jnp.linalg.norm(soft_map - hard_argmin_pt, axis=-1)))
    assert err < 1e-3, f"soft_argmin did not concentrate: max err {err:.4e}"


def test_soft_ctransform_is_numerically_stable_small_eps():
    rng = np.random.default_rng(3)
    B0, B1 = 64, 64
    y0 = jnp.asarray(rng.standard_normal((B0, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((B1, 2)), dtype=jnp.float32)
    g = jnp.asarray(rng.standard_normal((B1,)) * 5.0, dtype=jnp.float32)
    C = jnp.sum((y0[:, None, :] - y1[None, :, :]) ** 2, axis=-1)

    for eps in [1e-5, 1e-7]:
        soft = entropic_ctransform(y0, y1, g, C, eps)
        assert jnp.all(jnp.isfinite(soft)), f"eps={eps}: non-finite values."


if __name__ == "__main__":
    test_soft_recovers_hard_in_eps_limit()
    test_soft_convergence_rate()
    test_soft_argmin_matches_argmin_at_small_eps()
    test_soft_ctransform_is_numerically_stable_small_eps()
    print("All soft c-transform tests passed.")
