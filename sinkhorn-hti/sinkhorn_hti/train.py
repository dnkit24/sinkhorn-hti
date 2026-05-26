import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import (
    EntropicConfig,
    LagrangianConfig,
    LBFGSInnerConfig,
    OptimConfig,
    TrainConfig,
)
from .lagrangian import stack_obs_per_condition
from .losses import (
    dual_and_map_losses_entropic,
    dual_and_map_losses_lbfgs,
    path_loss,
)
from .networks import HTIComponents


def train(
    comps: HTIComponents,
    data: dict[float, dict[int, jnp.ndarray]],
    train_cfg: TrainConfig,
    lag_cfg: LagrangianConfig = LagrangianConfig(),
    optim_cfg: OptimConfig = OptimConfig(),
    ent_cfg: EntropicConfig = EntropicConfig(),
    lbfgs_cfg: LBFGSInnerConfig = LBFGSInnerConfig(),
) -> tuple[HTIComponents, list[dict[str, float]]]:
    rng = np.random.default_rng(train_cfg.seed)
    times = sorted(data.keys())
    conds = tuple(sorted(data[times[0]].keys()))
    n_int = len(times) - 1

    ys_per_c = stack_obs_per_condition(data, conditions=conds)
    xs_per_c = jnp.stack([jnp.asarray([float(c)]) for c in conds])

    kc = [(k, c_idx) for k in range(n_int) for c_idx in range(len(conds))]
    xs_kc = jnp.stack([xs_per_c[c_idx] for _, c_idx in kc])
    ys_per_kc = jnp.stack([ys_per_c[c_idx] for _, c_idx in kc])
    ks_static = tuple(k for k, _ in kc)

    if train_cfg.method == "entropic":
        def dual_map(comps, k, y0, y1, x, ys_c, eps):
            return dual_and_map_losses_entropic(
                comps, k, y0, y1, x, ys_c,
                h_y=lag_cfg.h_y, alpha=lag_cfg.alpha,
                epsilon=eps, n_quad=lag_cfg.n_quad,
            )
    else:
        def dual_map(comps, k, y0, y1, x, ys_c, _eps):
            return dual_and_map_losses_lbfgs(
                comps, k, y0, y1, x, ys_c,
                h_y=lag_cfg.h_y, alpha=lag_cfg.alpha,
                lbfgs_cfg=lbfgs_cfg, n_quad=lag_cfg.n_quad,
            )

    clip = optax.clip_by_global_norm(optim_cfg.grad_clip)
    opts = {
        "g": optax.chain(clip, optax.adam(optim_cfg.lr_g)),
        "T": optax.chain(clip, optax.adam(optim_cfg.lr_T)),
        "S": optax.chain(clip, optax.adam(optim_cfg.lr_S)),
        "G": optax.chain(clip, optax.adam(optim_cfg.lr_G)),
    }
    st_g = opts["g"].init(eqx.filter(comps.potentials, eqx.is_array))
    st_T = opts["T"].init(eqx.filter(comps.maps, eqx.is_array))
    st_S = opts["S"].init(eqx.filter(comps.spline, eqx.is_array))
    st_G = opts["G"].init(eqx.filter(comps.metric, eqx.is_array))

    @eqx.filter_jit
    def step_gT(comps, st_g, st_T, k, y0, y1, xs, ys_per_c, eps):
        def per_c(a, b, x, ys_c):
            d, mp, aux = dual_map(comps, k, a, b, x, ys_c, eps)
            return d + mp, (d, mp, aux)

        def scalar(m):
            scs, (ds, mps, aux) = jax.vmap(
                lambda a, b, x, ys: per_c(a, b, x, ys)
            )(y0, y1, xs, ys_per_c)
            return jnp.mean(scs), (
                jnp.mean(ds), jnp.mean(mps),
                {k: jnp.mean(v) for k, v in aux.items()},
            )

        (_, (d_val, m_val, aux_mean)), grad = eqx.filter_value_and_grad(
            scalar, has_aux=True,
        )(comps)
        g_upd, st_g = opts["g"].update(
            eqx.filter(grad.potentials, eqx.is_array),
            st_g, eqx.filter(comps.potentials, eqx.is_array),
        )
        T_upd, st_T = opts["T"].update(
            eqx.filter(grad.maps, eqx.is_array),
            st_T, eqx.filter(comps.maps, eqx.is_array),
        )
        comps = eqx.tree_at(
            lambda t: (t.potentials, t.maps),
            comps,
            (eqx.apply_updates(comps.potentials, g_upd),
             eqx.apply_updates(comps.maps, T_upd)),
        )
        return comps, st_g, st_T, d_val, m_val, aux_mean

    @eqx.filter_jit
    def step_S(comps, st_S, ks, y0_kc, xs_kc, ys_per_kc):
        def scalar(m):
            return jnp.mean(jnp.stack([
                path_loss(m, k, y0_kc[i], xs_kc[i], ys_per_kc[i],
                          h_y=lag_cfg.h_y, alpha=lag_cfg.alpha,
                          n_quad=lag_cfg.n_quad)
                for i, k in enumerate(ks)
            ]))

        loss, grad = eqx.filter_value_and_grad(scalar)(comps)
        upd, st_S = opts["S"].update(
            eqx.filter(grad.spline, eqx.is_array),
            st_S, eqx.filter(comps.spline, eqx.is_array),
        )
        return (
            eqx.tree_at(lambda t: t.spline, comps,
                        eqx.apply_updates(comps.spline, upd)),
            st_S, loss,
        )

    @eqx.filter_jit
    def step_G(comps, st_G, ks, y0_kc, y1_kc, xs_kc, ys_per_kc, eps):
        def scalar(m):
            # update the outer metric: minimise the negative dual at fixed g, T, S, y_1^*
            return jnp.mean(jnp.stack([
                -dual_map(m, k, y0_kc[i], y1_kc[i], xs_kc[i], ys_per_kc[i], eps)[0]
                for i, k in enumerate(ks)
            ]))

        loss, grad = eqx.filter_value_and_grad(scalar)(comps)
        upd, st_G = opts["G"].update(
            eqx.filter(grad.metric, eqx.is_array),
            st_G, eqx.filter(comps.metric, eqx.is_array),
        )
        return (
            eqx.tree_at(lambda t: t.metric, comps,
                        eqx.apply_updates(comps.metric, upd)),
            st_G, loss,
        )

    B = train_cfg.batch_size

    def draw(k, c):
        p0, p1 = data[times[k]][c], data[times[k + 1]][c]
        return (p0[rng.integers(0, p0.shape[0], size=B)],
                p1[rng.integers(0, p1.shape[0], size=B)])

    def sample_interval(k):
        pairs = [draw(k, c) for c in conds]
        return jnp.stack([y0 for y0, _ in pairs]), jnp.stack([y1 for _, y1 in pairs])

    def sample_kc():
        pairs = [draw(k, conds[c_idx]) for k, c_idx in kc]
        return jnp.stack([y0 for y0, _ in pairs]), jnp.stack([y1 for _, y1 in pairs])

    eps_t = jnp.asarray(ent_cfg.epsilon, dtype=jnp.float32)
    logs: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for outer in range(train_cfg.n_outer):
        d_acc = m_acc = s_acc = gmax_acc = 0.0
        n_steps = 0

        for _ in range(train_cfg.n_inner):
            for k in range(n_int):
                y0, y1 = sample_interval(k)
                comps, st_g, st_T, d, mp, aux = step_gT(
                    comps, st_g, st_T, k, y0, y1, xs_per_c, ys_per_c, eps_t,
                )
                d_acc += float(d); m_acc += float(mp); n_steps += 1
                gmax_acc += float(aux["grad_max"])

            y0_kc, _ = sample_kc()
            comps, st_S, s_loss = step_S(
                comps, st_S, ks_static, y0_kc, xs_kc, ys_per_kc,
            )
            s_acc += float(s_loss)

        y0_kc, y1_kc = sample_kc()
        comps, st_G, g_loss = step_G(
            comps, st_G, ks_static, y0_kc, y1_kc, xs_kc, ys_per_kc, eps_t,
        )

        if outer % train_cfg.log_every == 0 or outer == train_cfg.n_outer - 1:
            entry = {
                "outer": outer,
                "dual": -d_acc / max(n_steps, 1),
                "map": m_acc / max(n_steps, 1),
                "path": s_acc / max(train_cfg.n_inner, 1),
                "metric": float(g_loss),
                "grad_max_avg": gmax_acc / max(n_steps, 1),
                "wall_s": time.perf_counter() - t0,
            }
            logs.append(entry)
            print(
                f"{train_cfg.method:>8} {outer:>4d}: "
                f"dual={entry['dual']:+.3f} map={entry['map']:+.4f} "
                f"path={entry['path']:+.3f} metric={entry['metric']:+.3f} "
                f"t={entry['wall_s']:.0f}s",
                flush=True,
            )

    return comps, logs
