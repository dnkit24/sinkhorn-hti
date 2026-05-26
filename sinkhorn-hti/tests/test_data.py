"""Tests for the semicircles dataset."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sinkhorn_hti.data.semicircles import (
    KAPPA_ANG,
    R_NOM,
    SEMICIRCLE_CONDITIONS,
    SEMICIRCLE_TIMES,
    SIGMA_RAD,
    eval_grid,
    offset_x,
    sample_anchors,
    sample_semicircles,
)


def _mean_angle_ref(c: int, t: float) -> float:
    """Paper §C.1.1 reference formula; duplicated so tests catch drift."""
    if c == 1:
        return float(t * np.pi)
    if c == 2:
        return float(-t * np.pi)
    if c == 3:
        return float((1.0 - t) * np.pi)
    if c == 4:
        return float((t - 1.0) * np.pi)
    raise ValueError(f"unknown c={c}")


def test_semicircle_geometry():
    key = jax.random.PRNGKey(0)
    data = sample_anchors(n_per_condition=500, anchor_times=(0.5,), key=key)[0.5]
    for c, ys in data.items():
        centre = jnp.asarray([offset_x(c), 0.0])
        r = jnp.linalg.norm(ys - centre, axis=-1)
        lo = R_NOM * float(np.exp(-5.0 * SIGMA_RAD))
        hi = R_NOM * float(np.exp(+5.0 * SIGMA_RAD))
        frac_inside = float(jnp.mean((r >= lo) & (r <= hi)))
        assert frac_inside > 0.99, f"c={c}: only {frac_inside:.3f} inside 5-sigma band"


def test_anchor_dict_shape():
    key = jax.random.PRNGKey(1)
    times = SEMICIRCLE_TIMES
    conds = SEMICIRCLE_CONDITIONS
    n = 37
    data = sample_anchors(n_per_condition=n, anchor_times=times, key=key)
    assert set(data.keys()) == set(times)
    for t, by_c in data.items():
        assert set(by_c.keys()) == set(conds)
        for c, ys in by_c.items():
            assert ys.shape == (n, 2), f"t={t}, c={c}: got {ys.shape}"
            assert ys.dtype == jnp.float32


def test_eval_grid_shape_and_condition_coverage():
    key = jax.random.PRNGKey(2)
    grid = eval_grid(n_per_condition=50, t_eval=0.25, key=key)
    assert set(grid.keys()) == set(SEMICIRCLE_CONDITIONS)
    for c, ys in grid.items():
        assert ys.shape == (50, 2)


def test_paper_c11_spec():
    """Regression on per-(c, t) radial and angular marginals (paper §C.1.1)."""
    n = 100_000
    for t in (0.0, 0.5, 1.0):
        for c in (1, 2, 3, 4):
            key = jax.random.PRNGKey(int(1000 * t + c))
            ys, _ = sample_semicircles(
                key, n_per_condition=n, t=t, conditions=(c,)
            )
            centre = jnp.asarray([offset_x(c), 0.0])

            r = jnp.linalg.norm(ys - centre, axis=-1)
            mean_r = float(jnp.mean(r))
            assert abs(mean_r - 1.0) < 0.005, (
                f"c={c}, t={t}: mean radius {mean_r:.4f} outside 1.0 ± 0.005"
            )

            dx = ys - centre
            theta = jnp.arctan2(dx[:, 1], dx[:, 0])
            mu = _mean_angle_ref(c, t)
            dtheta = jnp.mod(theta - mu + jnp.pi, 2 * jnp.pi) - jnp.pi
            cdf_at_mu = float(jnp.mean(dtheta < 0))
            assert abs(cdf_at_mu - 0.5) < 0.01, (
                f"c={c}, t={t}: angular CDF at μ = {cdf_at_mu:.4f} outside 0.5 ± 0.01"
            )

    assert offset_x(1) == -1.0
    assert offset_x(2) == -1.0
    assert offset_x(3) == +1.0
    assert offset_x(4) == +1.0


def test_conditional_means_at_endpoints():
    key = jax.random.PRNGKey(3)
    data = sample_anchors(n_per_condition=500, anchor_times=(0.0, 1.0), key=key)
    d0 = data[0.0]; d1 = data[1.0]
    assert float(jnp.mean(d0[1][:, 0])) > -1.0   # c=1, t=0 -> near (0, 0)
    assert float(jnp.mean(d1[1][:, 0])) < -1.0   # c=1, t=1 -> near (-2, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
