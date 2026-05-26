import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import logsumexp

from .config import LBFGSInnerConfig


def entropic_ctransform(y0, y1_samples, g_values, cost_matrix, epsilon):
    # sum term dropped as no gradient contribution
    z = (g_values[None, :] - cost_matrix) / epsilon
    return -epsilon * logsumexp(z, axis=-1)


def soft_argmin(y1_samples, g_values, cost_matrix, epsilon):
    weights = jax.nn.softmax((g_values[None, :] - cost_matrix) / epsilon, axis=-1)
    return weights @ y1_samples


def _lbfgs_minimise(y1_init, f_fn, cfg):
    solver = optax.lbfgs(
        memory_size=cfg.memory_size,
        scale_init_precond=True,
        linesearch=optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=15,
            slope_rtol=1e-4,
            decrease_factor=0.5,
            store_grad=True,
        ),
    )
    state = solver.init(y1_init)

    def step(carry, _):
        y1, state = carry
        val, grad = jax.value_and_grad(f_fn)(y1)
        updates, state = solver.update(
            grad, state, y1, value=val, grad=grad, value_fn=f_fn,
        )
        y1_new = jnp.clip(optax.apply_updates(y1, updates), -cfg.y1_bound, cfg.y1_bound)
        return (y1_new, state), None

    (y1_final, _), _ = jax.lax.scan(step, (y1_init, state), None, length=cfg.maxiter)
    final_grad = jax.grad(f_fn)(y1_final)
    return y1_final, jnp.max(jnp.abs(final_grad))


def lbfgs_ctransform(y0, y0_init, g_fn, cost_fn, cfg=LBFGSInnerConfig()):
    def _solve_one(y0_i, y1_init_i):
        return _lbfgs_minimise(y1_init_i, lambda y1: cost_fn(y0_i, y1) - g_fn(y1), cfg)

    y1_star, grad_norms = jax.vmap(_solve_one)(y0, y0_init)
    ct_vals = jax.vmap(lambda a, b: cost_fn(a, b) - g_fn(b))(y0, y1_star)
    return y1_star, ct_vals, grad_norms
