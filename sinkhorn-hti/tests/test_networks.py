"""Tests for the network modules (zero-init, FiLM modes, SPD metric)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from sinkhorn_hti.networks import (
    FiLMMLP,
    HTIComponents,
    MapNet,
    MetricNet,
    PotentialNet,
    SplineNet,
)


@pytest.fixture
def comps():
    key = jax.random.PRNGKey(0)
    return HTIComponents.init(
        key=key,
        n_intervals=2,
        dy=2,
        dx=1,
        film_mode="full",
    )


def test_forward_shapes(comps):
    y = jnp.asarray([0.3, -0.2])
    x = jnp.asarray([2.0])
    assert comps.potentials[0](y, x).shape == ()
    assert comps.maps[0](y, x).shape == (2,)
    ctrl = comps.spline(y, y + 1.0, x)
    assert ctrl.shape == (15, 2)
    assert comps.metric(y, x).shape == (2, 2)


def test_residual_map_init_is_identity(comps):
    key = jax.random.PRNGKey(7)
    ys = jax.random.normal(key, (100, 2))
    xs = jax.random.uniform(jax.random.PRNGKey(8), (100, 1), minval=1.0, maxval=4.0)

    def one(y, x):
        return comps.maps[0](y, x) - y

    diffs = jax.vmap(one)(ys, xs)
    assert float(jnp.max(jnp.abs(diffs))) < 1e-6


def test_spline_init_is_linear_interpolation(comps):
    key = jax.random.PRNGKey(9)
    k1, k2, k3 = jax.random.split(key, 3)
    y0 = jax.random.normal(k1, (2,))
    y1 = jax.random.normal(k2, (2,))
    x = jax.random.uniform(k3, (1,), minval=1.0, maxval=4.0)
    ctrl = comps.spline(y0, y1, x)
    K = ctrl.shape[0]
    for k in range(K):
        s = k / (K - 1)
        expected = (1 - s) * y0 + s * y1
        err = float(jnp.linalg.norm(ctrl[k] - expected))
        assert err < 1e-6, f"k={k}: err={err:.3e}"


def test_spline_endpoints(comps):
    """``spline[0] == y0`` and ``spline[-1] == y1`` exactly."""
    y0 = jnp.asarray([0.3, -0.2])
    y1 = jnp.asarray([1.1, 0.4])
    x = jnp.asarray([2.0])
    ctrl = comps.spline(y0, y1, x)
    assert jnp.array_equal(ctrl[0], y0)
    assert jnp.array_equal(ctrl[-1], y1)


def test_film_full_vs_partial():
    """``full``: x-swap changes all channels. ``partial``: only first ``film_size``."""
    key = jax.random.PRNGKey(10)
    hidden = 8
    film = 3

    mlp_full = FiLMMLP(
        in_size=2, out_size=1, hidden_sizes=(hidden, hidden),
        x_size=1, film_size=film, key=key, film_mode="full",
    )
    mlp_partial = FiLMMLP(
        in_size=2, out_size=1, hidden_sizes=(hidden, hidden),
        x_size=1, film_size=film, key=key, film_mode="partial",
    )

    z = jnp.asarray([0.5, -0.5])
    x1 = jnp.asarray([1.0])
    x2 = jnp.asarray([3.0])

    def hidden_0(mlp, x):
        h = mlp.first(z)
        gamma = mlp.film_scale(x); beta = mlp.film_shift(x)
        if mlp.film_mode == "full":
            return gamma * h + beta
        return h.at[: mlp.film_size].set(gamma * h[: mlp.film_size] + beta)

    d_full = hidden_0(mlp_full, x1) - hidden_0(mlp_full, x2)
    d_part = hidden_0(mlp_partial, x1) - hidden_0(mlp_partial, x2)

    assert jnp.all(jnp.abs(d_full) > 1e-6), "'full' mode should change all channels"
    assert jnp.all(jnp.abs(d_part[:film]) > 1e-6), "'partial' should change first film_size"
    assert jnp.all(jnp.abs(d_part[film:]) < 1e-6), \
        "'partial' should NOT change channels after film_size"


def test_metric_spd():
    key = jax.random.PRNGKey(11)
    metric = MetricNet(
        dy=2, dx=1, hidden_sizes=(32, 32), film_size=16,
        eigenvalue_budget=2.0, key=key,
    )
    for seed in range(5):
        k = jax.random.PRNGKey(seed + 100)
        q = jax.random.normal(k, (2,))
        x = jax.random.uniform(k, (1,), minval=1.0, maxval=4.0)
        G = metric(q, x)
        assert float(jnp.max(jnp.abs(G - G.T))) < 1e-5
        eigs = jnp.linalg.eigvalsh(G)
        assert float(jnp.min(eigs)) > 0.0


def test_metric_eigenvalue_budget():
    """``tr(G) == budget`` up to float tol."""
    key = jax.random.PRNGKey(12)
    budget = 2.0
    metric = MetricNet(
        dy=2, dx=1, hidden_sizes=(32, 32), film_size=16,
        eigenvalue_budget=budget, key=key, eigenvalue_floor=0.1,
    )
    q = jnp.asarray([0.3, -0.2])
    x = jnp.asarray([2.0])
    G = metric(q, x)
    assert abs(float(jnp.trace(G)) - budget) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
