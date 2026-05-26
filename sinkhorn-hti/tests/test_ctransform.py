"""Tests for the c-transform module (soft and L-BFGS variants)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sinkhorn_hti.config import LBFGSInnerConfig
from sinkhorn_hti.ctransform import (
    entropic_ctransform,
    lbfgs_ctransform,
    soft_argmin,
)


def test_lbfgs_decreases_objective():
    """On ``c(y_0, y_1) = 0.5 ||y_1 - y_0 - 0.3||^2``, recovers ``y_1* = y_0 + 0.3``."""
    key = jax.random.PRNGKey(0)
    B = 8
    y0 = jax.random.normal(key, (B, 2))
    y0_init = jax.random.normal(jax.random.PRNGKey(1), (B, 2))

    def g_fn(y):
        return jnp.zeros(())

    def c_fn(a, b):
        return 0.5 * jnp.sum((b - a - 0.3) ** 2)

    y_star, _, _ = lbfgs_ctransform(y0, y0_init, g_fn, c_fn, LBFGSInnerConfig(maxiter=20))
    err = float(jnp.max(jnp.abs(y_star - (y0 + 0.3))))
    assert err < 1e-3, f"max err {err:.3e}"


def test_lbfgs_warm_start_used():
    """Starting from the optimum, L-BFGS terminates with near-zero change."""
    B = 4
    y0 = jnp.zeros((B, 2))
    y0_init = y0 + 0.3

    def g_fn(y):
        return jnp.zeros(())

    def c_fn(a, b):
        return 0.5 * jnp.sum((b - a - 0.3) ** 2)

    y_star, _, _ = lbfgs_ctransform(y0, y0_init, g_fn, c_fn, LBFGSInnerConfig(maxiter=10))
    assert float(jnp.max(jnp.abs(y_star - y0_init))) < 1e-5


def test_lbfgs_no_trust_region():
    """Optimum at ``y_0 + 5`` with warm start at ``y_0``: must converge."""
    B = 4
    y0 = jnp.zeros((B, 2))
    y0_init = y0.copy()

    def g_fn(y):
        return jnp.zeros(())

    def c_fn(a, b):
        return 0.5 * jnp.sum((b - a - 5.0) ** 2)

    y_star, _, _ = lbfgs_ctransform(y0, y0_init, g_fn, c_fn, LBFGSInnerConfig(maxiter=20))
    err = float(jnp.max(jnp.abs(y_star - (y0 + 5.0))))
    assert err < 1e-2, f"max err {err:.3e} (trust region regressed?)"


def test_lbfgs_logs_grad_norm():
    def g_fn(y):
        return jnp.zeros(())

    def c_fn(y0, y1):
        return 0.5 * jnp.sum((y1 - y0 - 5.0) ** 2)

    y0 = jnp.zeros((4, 2))
    cfg = LBFGSInnerConfig(maxiter=20, y1_bound=100.0)
    y_star, ct, gn = lbfgs_ctransform(y0, y0, g_fn, c_fn, cfg)
    assert gn.shape == (4,), f"expected per-sample grad norms, got {gn.shape}"
    assert float(gn.max()) < 1e-4, f"expected converged grad-norm, got {float(gn.max()):.3e}"
    assert float(jnp.max(jnp.abs(y_star - 5.0))) < 1e-3


def test_lbfgs_config_has_no_dead_fields():
    """``LBFGSInnerConfig`` must not have a ``tol`` field (dead, never read)."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(LBFGSInnerConfig)}
    assert "tol" not in field_names, "tol field must not exist (dead code)"
    assert "maxiter" in field_names
    assert "memory_size" in field_names
    assert "y1_bound" in field_names


def test_lbfgs_grad_norm_unconverged_at_maxiter_2():
    """Rosenbrock-like non-convex objective: at ``maxiter=2`` the grad-norm
    must stay above 1e-2. Plain quadratics L-BFGS-trivialise in one step.
    """
    def g_fn(y):
        return jnp.zeros(())

    def c_fn(y0, y1):
        return 100.0 * (y1[1] - y1[0] ** 2) ** 2 + (1.0 - y1[0]) ** 2

    y0 = jnp.zeros((4, 2))
    y_init = -jnp.ones((4, 2))
    cfg = LBFGSInnerConfig(maxiter=2, y1_bound=100.0)
    _, _, gn = lbfgs_ctransform(y0, y_init, g_fn, c_fn, cfg)
    assert float(gn.max()) > 1e-2, (
        f"expected unconverged grad-norm > 1e-2 at maxiter=2, got {float(gn.max()):.3e}"
    )


def _cost_matrix_quadratic(y0, y1):
    return jnp.sum((y0[:, None, :] - y1[None, :, :]) ** 2, axis=-1)


def test_entropic_recovers_hard_in_limit():
    """Jensen bound ``soft <= hard``; gap monotone in eps."""
    rng = np.random.default_rng(42)
    B0, B1 = 8, 64
    y0 = jnp.asarray(rng.standard_normal((B0, 2)), dtype=jnp.float32)
    y1 = y0[:, None, :] + 0.3 + 0.01 * jnp.asarray(
        rng.standard_normal((B0, B1, 2)), dtype=jnp.float32,
    )
    y1_flat = y1.reshape(-1, 2)
    C = _cost_matrix_quadratic(y0, y1_flat)
    g = jnp.zeros(y1_flat.shape[0])

    hard = jnp.min(C - g[None, :], axis=-1)
    gaps = []
    for eps in [1.0, 0.1, 0.01]:
        soft = entropic_ctransform(y0, y1_flat, g, C, eps)
        gaps.append(float(jnp.mean(hard - soft)))
        assert jnp.all(hard - soft >= -1e-4)
    for a, b in zip(gaps[:-1], gaps[1:], strict=True):
        assert b <= a + 1e-6


def test_barycenter_in_convex_hull():
    """``T_hat(y_0)`` is a softmax-weighted average, hence coordinate-wise in ``[min, max]``."""
    rng = np.random.default_rng(7)
    y0 = jnp.asarray(rng.standard_normal((8, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((32, 2)), dtype=jnp.float32)
    g = jnp.asarray(rng.standard_normal((32,)), dtype=jnp.float32)
    C = _cost_matrix_quadratic(y0, y1)

    T_hat = soft_argmin(y1, g, C, epsilon=0.1)
    lo = jnp.min(y1, axis=0)
    hi = jnp.max(y1, axis=0)
    assert jnp.all(T_hat >= lo - 1e-5)
    assert jnp.all(T_hat <= hi + 1e-5)


def test_logsumexp_numerical_stability():
    """At eps = 1e-3 with ``|g/eps| ~ 1e3``, no inf/nan in value or gradient."""
    rng = np.random.default_rng(11)
    y0 = jnp.asarray(rng.standard_normal((16, 2)), dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((32, 2)), dtype=jnp.float32)
    C = _cost_matrix_quadratic(y0, y1)
    g = jnp.asarray(rng.standard_normal((32,)) * 1.0, dtype=jnp.float32)

    eps = 1e-3
    val = entropic_ctransform(y0, y1, g, C, eps)
    assert jnp.all(jnp.isfinite(val))

    grad = jax.grad(lambda g_: jnp.sum(entropic_ctransform(y0, y1, g_, C, eps)))(g)
    assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
