import bisect

import jax
import jax.numpy as jnp

from .lagrangian import cubic_spline_eval
from .networks import HTIComponents


def locate_interval(t_star, anchor_times):
    ts = list(anchor_times)
    if t_star <= ts[0]:
        return 0, 0.0
    if t_star >= ts[-1]:
        return len(ts) - 2, 1.0
    k = bisect.bisect_right(ts, t_star) - 1
    return k, float((t_star - ts[k]) / (ts[k + 1] - ts[k]))


def sample(
    comps: HTIComponents,
    anchor_times,
    anchor_samples,
    x: jnp.ndarray,
    t_star: float,
    n_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    k, s_star = locate_interval(t_star, anchor_times)
    pool = anchor_samples[anchor_times[k]]
    y_k = pool[jax.random.randint(key, (n_samples,), 0, pool.shape[0])]

    def _one(y):
        # geodesic from y to T(y); evaluated at s_star to land at time t_star
        ctrl = comps.spline(y, comps.maps[k](y, x), x)
        q, _ = cubic_spline_eval(ctrl, jnp.asarray([s_star]))
        return q[0]

    return jax.vmap(_one)(y_k)
