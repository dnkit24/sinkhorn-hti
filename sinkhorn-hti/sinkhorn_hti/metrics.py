import jax
import jax.numpy as jnp
import numpy as np

from .data.semicircles import (
    SEMICIRCLE_CONDITIONS,
    SEMICIRCLE_EVAL_TIMES,
    offset_x,
    sample_semicircles,
)
from .sampling import sample


def nll_kde(samples, eval_points, bandwidth):
    n, d = samples.shape
    sq = jnp.sum((eval_points[:, None, :] - samples[None, :, :]) ** 2, axis=-1)
    log_norm = -0.5 * d * jnp.log(2.0 * jnp.pi * bandwidth ** 2)
    # log-density of each eval_point under the Gaussian KDE built from samples
    log_p = jax.scipy.special.logsumexp(log_norm - sq / (2.0 * bandwidth ** 2), axis=-1) - jnp.log(n)
    return float(-jnp.mean(log_p))


def circle_distance(samples, offset_x, radius=1.0):
    r = jnp.linalg.norm(samples - jnp.asarray([offset_x, 0.0]), axis=-1)
    return float(jnp.mean(jnp.abs(r - radius)))


def evaluate_semicircles(
    comps,
    data,
    eval_times=SEMICIRCLE_EVAL_TIMES,
    conditions=SEMICIRCLE_CONDITIONS,
    n_eval=500,
    bandwidth=0.05,
):
    anchors = sorted(data.keys())
    out = {}
    for t_star in eval_times:
        nlls, cds = [], []
        for c in conditions:
            per_c = {t: data[t][c] for t in anchors}
            gen = sample(
                comps, anchors, per_c,
                jnp.asarray([float(c)]), float(t_star),
                n_eval, jax.random.PRNGKey(999 * c),
            )
            gt, _ = sample_semicircles(
                jax.random.PRNGKey(100 + c),
                n_per_condition=n_eval, t=t_star, conditions=(c,),
            )
            nlls.append(nll_kde(gen, gt, bandwidth=bandwidth))
            cds.append(circle_distance(gen, offset_x=offset_x(c)))
        out[t_star] = (float(np.mean(nlls)), float(np.mean(cds)))
    return out
