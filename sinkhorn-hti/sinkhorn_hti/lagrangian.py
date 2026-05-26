import jax
import jax.numpy as jnp


def gaussian_kernel(u, h):
    d = u.shape[-1]
    log_norm = -0.5 * d * jnp.log(2.0 * jnp.pi * h ** 2)
    return jnp.exp(log_norm - jnp.sum(u ** 2, axis=-1) / (2.0 * h ** 2))


def kde_density(q, ys_c, h_y):
    q_arr = jnp.atleast_2d(q)
    p = jnp.mean(gaussian_kernel(q_arr[:, None, :] - ys_c[None, :, :], h_y), axis=-1)
    return p[0] if q.ndim == 1 else p


def potential_U(q, ys_c, h_y, alpha, density_eps=1e-8):
    return alpha * jnp.log(kde_density(q, ys_c, h_y) + density_eps)


def build_obs_by_condition(data):
    times = sorted(data.keys())
    conds = sorted(data[times[0]].keys())
    return {c: jnp.concatenate([data[t][c] for t in times], axis=0) for c in conds}


def stack_obs_per_condition(data, conditions):
    by_c = build_obs_by_condition(data)
    return jnp.stack([by_c[c] for c in conditions], axis=0)


def _givens_rotation(d, i, j, theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.eye(d).at[i, i].set(c).at[j, j].set(c).at[i, j].set(-s).at[j, i].set(s)


def assemble_metric(angles, eigenvalues, d):
    R = jnp.eye(d)
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            R = R @ _givens_rotation(d, i, j, angles[idx])
            idx += 1
    return R @ jnp.diag(eigenvalues) @ R.T


def cubic_spline_eval(control_points, s):
    K, _ = control_points.shape
    n_seg = K - 1

    p = control_points
    v = jnp.zeros_like(p)
    # tangent at each knot from neighbouring points; one-sided at the ends
    v = v.at[0].set(p[1] - p[0]).at[-1].set(p[-1] - p[-2])
    if K > 2:
        v = v.at[1:-1].set((p[2:] - p[:-2]) / 2.0)

    s = jnp.atleast_1d(s)
    seg = jnp.clip(jnp.floor(s * n_seg).astype(jnp.int32), 0, n_seg - 1)
    u = s * n_seg - seg

    p0, p1 = p[seg], p[seg + 1]
    v0, v1 = v[seg], v[seg + 1]

    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    q = (h00[..., None] * p0 + h10[..., None] * v0
         + h01[..., None] * p1 + h11[..., None] * v1)

    dh00 = 6 * u ** 2 - 6 * u
    dh10 = 3 * u ** 2 - 4 * u + 1
    dh01 = -6 * u ** 2 + 6 * u
    dh11 = 3 * u ** 2 - 2 * u
    qdot = n_seg * (dh00[..., None] * p0 + dh10[..., None] * v0
                    + dh01[..., None] * p1 + dh11[..., None] * v1)
    return q, qdot


def action_along_spline(control_points, x, metric_fn, potential_fn, n_quad=9):
    assert n_quad % 2 == 1, "Simpson's rule needs odd n_quad."
    s = jnp.linspace(0.0, 1.0, n_quad)
    w = jnp.ones(n_quad).at[1:-1:2].set(4.0).at[2:-1:2].set(2.0) / (3.0 * (n_quad - 1))

    q, qdot = cubic_spline_eval(control_points, s)

    def _integrand(qi, qdoti):
        return 0.5 * qdoti @ metric_fn(qi, x) @ qdoti - potential_fn(qi, x)

    return jnp.sum(w * jax.vmap(_integrand)(q, qdot))
