import jax
import jax.numpy as jnp
import numpy as np

R_NOM = 1.0
SIGMA_RAD = 0.05
KAPPA_ANG = 5.0

SEMICIRCLE_TIMES = (0.0, 0.5, 1.0)
SEMICIRCLE_CONDITIONS = (1, 2, 3, 4)
SEMICIRCLE_EVAL_TIMES = (0.25, 0.75)


def offset_x(c):
    return -1.0 if c in (1, 2) else 1.0


def _mean_angle(c, t):
    # one half-circle per condition; t parametrises angular motion from 0 to +/- pi
    if c == 1:
        return t * jnp.pi
    if c == 2:
        return -t * jnp.pi
    if c == 3:
        return (1.0 - t) * jnp.pi
    if c == 4:
        return (t - 1.0) * jnp.pi
    raise ValueError(f"Unknown condition c={c}")


def _sample_vonmises(key, mu, kappa, shape):
    # JAX has no native Von Mises; fall back to numpy.
    seed = int(jax.random.bits(key, (1,))[0])
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.vonmises(mu=float(mu), kappa=float(kappa), size=shape), dtype=jnp.float32)


def _sample_one(key, n, c, t):
    k1, k2 = jax.random.split(key)
    r = jnp.exp(jax.random.normal(k1, (n,)) * SIGMA_RAD + jnp.log(R_NOM))
    phi = _sample_vonmises(k2, _mean_angle(c, t), KAPPA_ANG, (n,))
    offset = jnp.asarray([offset_x(c), 0.0])
    return offset + jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi)], axis=-1)


def sample_semicircles(key, n_per_condition, t, conditions=SEMICIRCLE_CONDITIONS):
    keys = jax.random.split(key, len(conditions))
    ys = [_sample_one(ck, n_per_condition, c, t) for ck, c in zip(keys, conditions)]
    xs = [jnp.full((n_per_condition,), c, dtype=jnp.int32) for c in conditions]
    return jnp.concatenate(ys, axis=0), jnp.concatenate(xs, axis=0)


def build_training_set(
    key,
    n_per_condition=100,
    times=SEMICIRCLE_TIMES,
    conditions=SEMICIRCLE_CONDITIONS,
):
    keys = jax.random.split(key, len(times) * len(conditions))
    out = {}
    idx = 0
    for t in times:
        out[t] = {}
        for c in conditions:
            out[t][c] = _sample_one(keys[idx], n_per_condition, c, t)
            idx += 1
    return out


def sample_anchors(n_per_condition, anchor_times, key, conditions=SEMICIRCLE_CONDITIONS):
    return build_training_set(key, n_per_condition, anchor_times, conditions)


def eval_grid(n_per_condition, t_eval, key, conditions=SEMICIRCLE_CONDITIONS):
    return build_training_set(key, n_per_condition, (t_eval,), conditions)[t_eval]
