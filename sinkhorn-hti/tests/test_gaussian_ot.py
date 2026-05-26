"""Closed-form Bures/W_2 sanity check for the entropic dual.

For centred isotropic Gaussians ``N(0, s_i^2 I)`` in R^d:

    W_2^2(N(0, s0^2 I), N(0, s1^2 I)) = d (s0 - s1)^2.

We optimise the entropic dual over an affine-quadratic g-family and compare
to the closed-form value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sinkhorn_hti.ctransform import entropic_ctransform


def _quadratic_cost_matrix(y0: jnp.ndarray, y1: jnp.ndarray) -> jnp.ndarray:
    diff = y0[:, None, :] - y1[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)


def test_entropic_dual_matches_closed_form_gaussian():
    d = 2
    s0, s1 = 1.0, 2.0
    N = 400
    rng = np.random.default_rng(0)
    y0 = jnp.asarray(rng.standard_normal((N, d)) * s0, dtype=jnp.float32)
    y1 = jnp.asarray(rng.standard_normal((N, d)) * s1, dtype=jnp.float32)
    C = _quadratic_cost_matrix(y0, y1)
    true_w2_sq = d * (s0 - s1) ** 2

    def g_fn(y, params):
        a, b, c = params[0], params[1 : 1 + d], params[1 + d]
        return a + jnp.dot(b, y) + c * jnp.sum(y ** 2)

    def dual_value(params, eps):
        g_vals = jax.vmap(lambda y: g_fn(y, params))(y1)
        g_ct = entropic_ctransform(y0, y1, g_vals, C, eps)
        return jnp.mean(g_ct) + jnp.mean(g_vals)

    params = jnp.zeros(2 + d)
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, eps):
        loss, grad = jax.value_and_grad(lambda p: -dual_value(p, eps))(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, -loss

    eps = 0.05
    for _ in range(1000):
        params, opt_state, dual = step(params, opt_state, eps)

    dual_val = float(dual)
    err = abs(dual_val - true_w2_sq)
    assert err < 1.0, f"dual {dual_val:.3f} vs true W_2^2 {true_w2_sq:.3f}, err {err:.3f}"


if __name__ == "__main__":
    test_entropic_dual_matches_closed_form_gaussian()
    print("Gaussian closed-form OT test passed.")
