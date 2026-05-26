import jax
import jax.numpy as jnp

from .ctransform import (
    LBFGSInnerConfig,
    entropic_ctransform,
    lbfgs_ctransform,
    soft_argmin,
)
from .lagrangian import action_along_spline, potential_U
from .networks import HTIComponents


def action_cost(
    comps: HTIComponents,
    y0: jnp.ndarray,
    y1: jnp.ndarray,
    x: jnp.ndarray,
    ys_c: jnp.ndarray,
    h_y: float,
    alpha: float,
    n_quad: int = 9,
) -> jnp.ndarray:
    ctrl = comps.spline(y0, y1, x)
    return action_along_spline(
        ctrl, x, comps.metric,
        lambda q, _x: potential_U(q, ys_c, h_y, alpha),
        n_quad,
    )


def _cost_matrix(comps, y0, y1, x, ys_c, h_y, alpha, n_quad):
    cost = lambda a, b: action_cost(comps, a, b, x, ys_c, h_y, alpha, n_quad)
    return jax.vmap(lambda a: jax.vmap(lambda b: cost(a, b))(y1))(y0)



def dual_and_map_losses_entropic(
    comps, k, y_k, y_kp1, x, ys_c,
    h_y: float, alpha: float, epsilon: float, n_quad: int = 9,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    A = _cost_matrix(comps, y_k, y_kp1, x, ys_c, h_y, alpha, n_quad)
    g_vals = jax.vmap(lambda y: comps.potentials[k](y, x))(y_kp1)
    g_ct = entropic_ctransform(y_k, y_kp1, g_vals, A, epsilon)
    dual = jnp.mean(g_ct) + jnp.mean(g_vals)

    T_hat = jax.lax.stop_gradient(soft_argmin(y_kp1, g_vals, A, epsilon))
    T_pred = jax.vmap(lambda y: comps.maps[k](y, x))(y_k)
    map_mse = jnp.mean(jnp.sum((T_pred - T_hat) ** 2, axis=-1))

    aux = {
        "grad_max": jnp.asarray(0.0, dtype=jnp.float32),
        "grad_mean": jnp.asarray(0.0, dtype=jnp.float32),
    }
    return -dual, map_mse, aux


def dual_and_map_losses_lbfgs(
    comps, k, y_k, y_kp1, x, ys_c,
    h_y: float, alpha: float,
    lbfgs_cfg: LBFGSInnerConfig = LBFGSInnerConfig(), n_quad: int = 9,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    T_init = jax.vmap(lambda y: comps.maps[k](y, x))(y_k)
    g_fn = lambda y: comps.potentials[k](y, x)
    c_fn = lambda y0, y1: action_cost(comps, y0, y1, x, ys_c, h_y, alpha, n_quad)

    y_star, _, grad_norms = lbfgs_ctransform(y_k, T_init, g_fn, c_fn, lbfgs_cfg)
    # detach gradient on y_star: outer gradient ignores the response of inner argmin
    y_star = jax.lax.stop_gradient(y_star)
    grad_norms = jax.lax.stop_gradient(grad_norms)

    ct_vals = jax.vmap(c_fn)(y_k, y_star) - jax.vmap(g_fn)(y_star)
    g_vals = jax.vmap(g_fn)(y_kp1)
    dual = jnp.mean(ct_vals) + jnp.mean(g_vals)

    T_pred = jax.vmap(lambda y: comps.maps[k](y, x))(y_k)
    map_mse = jnp.mean(jnp.sum((T_pred - y_star) ** 2, axis=-1))

    aux = {"grad_max": jnp.max(grad_norms), "grad_mean": jnp.mean(grad_norms)}
    return -dual, map_mse, aux


def path_loss(
    comps, k, y_k, x, ys_c,
    h_y: float, alpha: float, n_quad: int = 9,
) -> jnp.ndarray:
    T_vals = jax.lax.stop_gradient(jax.vmap(lambda y: comps.maps[k](y, x))(y_k))
    return jnp.mean(jax.vmap(
        lambda a, b: action_cost(comps, a, b, x, ys_c, h_y, alpha, n_quad)
    )(y_k, T_vals))
