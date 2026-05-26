"""Tests for Lagrangian pieces (KDE, spline action, frozen U invariance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sinkhorn_hti.data.semicircles import sample_anchors
from sinkhorn_hti.lagrangian import (
    action_along_spline,
    build_obs_by_condition,
    cubic_spline_eval,
    kde_density,
    potential_U,
    stack_obs_per_condition,
)


def test_kde_density_normalisation():
    """Riemann sum of ``p_hat(q|x_c)`` over a fine grid is close to 1."""
    key = jax.random.PRNGKey(0)
    data = sample_anchors(n_per_condition=200, anchor_times=(0.0, 0.5, 1.0), key=key)
    by_c = build_obs_by_condition(data)
    ys_c = by_c[1]

    lo, hi, n = -3.0, 3.0, 200
    ax = jnp.linspace(lo, hi, n)
    XX, YY = jnp.meshgrid(ax, ax)
    Q = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    p = kde_density(Q, ys_c, h_y=0.1)
    cell_area = (hi - lo) ** 2 / (n ** 2)
    Z = float(jnp.sum(p) * cell_area)
    assert abs(Z - 1.0) < 0.05, f"integral {Z:.4f} not close to 1"


def test_kde_uses_full_data_not_subsample():
    """Calling KDE twice with the same source set gives bytewise-identical output."""
    key = jax.random.PRNGKey(1)
    data = sample_anchors(n_per_condition=50, anchor_times=(0.0, 1.0), key=key)
    by_c = build_obs_by_condition(data)
    ys_c = by_c[2]

    q = jnp.asarray([[-1.0, 0.5]])
    p1 = kde_density(q, ys_c, h_y=0.1)
    p2 = kde_density(q, ys_c, h_y=0.1)
    assert jnp.array_equal(p1, p2)


def test_stack_obs_per_condition_shape():
    key = jax.random.PRNGKey(2)
    data = sample_anchors(n_per_condition=30, anchor_times=(0.0, 0.5, 1.0), key=key)
    stacked = stack_obs_per_condition(data, conditions=(1, 2, 3, 4))
    assert stacked.shape == (4, 90, 2), stacked.shape


def test_spline_velocity_finite():
    key = jax.random.PRNGKey(3)
    ctrl = jax.random.normal(key, (8, 2))
    _, qdot = cubic_spline_eval(ctrl, jnp.asarray([0.0, 1.0]))
    assert jnp.all(jnp.isfinite(qdot))


def test_action_straight_line_zero_potential():
    """``G = I``, ``U = 0``, straight line ``(0,0) -> (1,0)`` has action ``0.5``."""
    K = 5
    ctrl = jnp.stack([
        jnp.linspace(0.0, 1.0, K),
        jnp.zeros(K),
    ], axis=-1)

    def metric_fn(q, x):
        return jnp.eye(2)

    def potential_fn(q, x):
        return jnp.zeros(())

    S = float(action_along_spline(ctrl, jnp.asarray([1.0]), metric_fn, potential_fn, n_quad=9))
    assert abs(S - 0.5) < 1e-5, f"action {S:.5f} != 0.5"


def test_potential_U_frozen_under_repeated_calls():
    key = jax.random.PRNGKey(4)
    data = sample_anchors(n_per_condition=40, anchor_times=(0.0, 0.5, 1.0), key=key)
    by_c = build_obs_by_condition(data)
    ys_c = by_c[1]

    q = jnp.asarray([-1.0, 0.5])
    u1 = potential_U(q, ys_c, h_y=0.1, alpha=1.0)
    u2 = potential_U(q, ys_c, h_y=0.1, alpha=1.0)
    assert jnp.array_equal(u1, u2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
