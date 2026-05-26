"""Smoke tests for the training loop (tiny dataset, finite losses)."""

from __future__ import annotations

import math

import jax
import pytest

from sinkhorn_hti.config import (
    EntropicConfig,
    LagrangianConfig,
    LBFGSInnerConfig,
    OptimConfig,
    TrainConfig,
)
from sinkhorn_hti.data.semicircles import sample_anchors
from sinkhorn_hti.lagrangian import potential_U, stack_obs_per_condition
from sinkhorn_hti.networks import HTIComponents
from sinkhorn_hti.train import train


def _tiny_data():
    return sample_anchors(
        n_per_condition=5,
        anchor_times=(0.0, 0.5, 1.0),
        key=jax.random.PRNGKey(0),
    )


def _tiny_comps():
    return HTIComponents.init(
        key=jax.random.PRNGKey(1),
        n_intervals=2,
        dy=2, dx=1,
        potential_hidden=(16, 16),
        map_hidden=(16, 16),
        spline_hidden=(32, 32),
        metric_hidden=(16, 16),
        spline_knots=5,
    )


def test_one_outer_iter():
    cfg = TrainConfig(n_outer=1, n_inner=1, batch_size=4, method="entropic", seed=0, log_every=1)
    data = _tiny_data()
    comps = _tiny_comps()
    _, logs = train(comps, data, cfg)
    assert len(logs) == 1
    for k, v in logs[0].items():
        if k == "outer":
            continue
        assert math.isfinite(float(v)), f"{k} = {v} non-finite"


def test_loss_finite_for_50_iter_entropic():
    cfg = TrainConfig(n_outer=50, n_inner=1, batch_size=8, method="entropic", seed=0, log_every=25)
    data = _tiny_data()
    comps = _tiny_comps()
    _, logs = train(comps, data, cfg)
    for entry in logs:
        for k in ("dual", "map", "path", "metric"):
            assert math.isfinite(entry[k]), f"outer={entry['outer']} {k}={entry[k]}"
        assert abs(entry["dual"]) < 10.0, f"dual drifted to {entry['dual']}"


def test_loss_finite_for_50_iter_lbfgs():
    cfg = TrainConfig(n_outer=50, n_inner=1, batch_size=8, method="lbfgs", seed=0, log_every=25)
    lbfgs = LBFGSInnerConfig(maxiter=3)
    data = _tiny_data()
    comps = _tiny_comps()
    _, logs = train(comps, data, cfg, lbfgs_cfg=lbfgs)
    for entry in logs:
        for k in ("dual", "map", "path", "metric"):
            assert math.isfinite(entry[k]), f"outer={entry['outer']} {k}={entry[k]}"


def test_kde_potential_is_pure():
    import jax.numpy as jnp

    data = _tiny_data()
    ys_per_c = stack_obs_per_condition(data, conditions=(1, 2, 3, 4))

    q = jnp.asarray([-1.0, 0.2])
    u1 = potential_U(q, ys_per_c[0], h_y=0.1, alpha=1.0)
    u2 = potential_U(q, ys_per_c[0], h_y=0.1, alpha=1.0)
    assert jnp.array_equal(u1, u2)


def test_train_does_not_mutate_data():
    """``train()`` must not mutate its ``data`` argument (``ys_per_c`` is
    captured once, not re-derived from a mutated source).
    """
    import jax.numpy as jnp

    data = _tiny_data()
    snapshot = {t: {c: v.copy() for c, v in by_c.items()}
                for t, by_c in data.items()}
    comps = _tiny_comps()
    cfg = TrainConfig(n_outer=2, n_inner=1, batch_size=4, method="entropic",
                      seed=0, log_every=1)
    train(comps, data, cfg)
    for t, by_c in data.items():
        for c, arr in by_c.items():
            assert jnp.array_equal(arr, snapshot[t][c]), f"data[{t}][{c}] mutated"


def test_evaluate_semicircles_returns_finite():
    from sinkhorn_hti.metrics import evaluate_semicircles

    data = _tiny_data()
    comps = _tiny_comps()
    res = evaluate_semicircles(comps, data, n_eval=8)
    assert set(res.keys()) == {0.25, 0.75}
    for t_star, (nll, cd) in res.items():
        assert math.isfinite(nll), f"NLL at t*={t_star} is {nll}"
        assert math.isfinite(cd), f"CD at t*={t_star} is {cd}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
