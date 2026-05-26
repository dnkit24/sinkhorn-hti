"""Forward-pass-only analyses on cached headline runs.

  1. eps-sweep:       empirical soft-dual on cached comps at varying eps
  2. per-eval-time:   NLL/CD breakdown at t in {0.25, 0.75}
  3. Sinkhorn div:    numerical OT^eps and S_eps on cached cost network

    python scripts/extra_analyses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

FOLDER = Path(__file__).resolve().parent.parent
PROJECT = FOLDER.parent
sys.path.insert(0, str(FOLDER))

from sinkhorn_hti.caching import load_run
from sinkhorn_hti.data.semicircles import (
    SEMICIRCLE_CONDITIONS, SEMICIRCLE_TIMES,
)
from sinkhorn_hti.losses import _cost_matrix as cost_matrix

RUNS = FOLDER / "runs"
REPORT = PROJECT / "report"
FIGS = REPORT / "figs"
SEED = 5


def load_pair():
    out = {}
    for m in ("lbfgs", "entropic"):
        payload, _ = load_run(RUNS / f"{m}_seed{SEED}.pkl")
        out[m] = payload
    return out


# ---------- 1. eps-sweep -------------------------------------------------

def empirical_soft_dual(comps, k, y0, y1, x, ys_c, h_y, alpha, eps):
    """Empirical soft-dual int g^{c,eps} dmu_hat + int g dnu_hat for one batch."""
    C = cost_matrix(comps, y0, y1, x, ys_c, h_y, alpha, n_quad=9)
    g_vals = jax.vmap(lambda y: comps.potentials[k](y, x))(y1)
    z = (g_vals[None, :] - C) / eps
    soft_ct = -eps * logsumexp(z, axis=-1)  # drop +eps log B (theta_g-indep)
    return float(jnp.mean(soft_ct) + jnp.mean(g_vals))


def eps_sweep(payloads):
    eps_grid = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    rng = np.random.default_rng(0)
    n_repeats = 8  # mini-batches per eps value for variance estimate
    B = 64
    h_y, alpha = 0.05, 0.05

    # collect data per condition / interval
    results = {m: np.zeros((len(eps_grid), n_repeats)) for m in payloads}
    for m, p in payloads.items():
        comps = p["comps"]
        data = p["data"]
        ts = sorted(data.keys())
        # stack obs per condition for U
        from sinkhorn_hti.lagrangian import stack_obs_per_condition
        ys_per_c = stack_obs_per_condition(data, conditions=SEMICIRCLE_CONDITIONS)
        for r in range(n_repeats):
            # one mini-batch per (k, c), average over k and c
            vals = []
            for k in range(len(ts) - 1):
                for c_idx, c in enumerate(SEMICIRCLE_CONDITIONS):
                    p0 = data[ts[k]][c]; p1 = data[ts[k + 1]][c]
                    i0 = rng.integers(0, p0.shape[0], size=B)
                    i1 = rng.integers(0, p1.shape[0], size=B)
                    y0, y1 = p0[i0], p1[i1]
                    x = jnp.asarray([float(c)])
                    for ie, eps in enumerate(eps_grid):
                        d = empirical_soft_dual(
                            comps, k, y0, y1, x, ys_per_c[c_idx],
                            h_y, alpha, float(eps),
                        )
                        vals.append((ie, d))
            arr = np.zeros((len(eps_grid), 0))
            from collections import defaultdict
            buckets = defaultdict(list)
            for ie, d in vals:
                buckets[ie].append(d)
            for ie in range(len(eps_grid)):
                results[m][ie, r] = np.mean(buckets[ie])

    return eps_grid, results


def plot_eps_sweep(eps_grid, results, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    mean = results["entropic"].mean(axis=1)
    std = results["entropic"].std(axis=1)
    ax.errorbar(eps_grid, mean, yerr=std, color="C1",
                label=r"Sinkhorn-HTI components ($\varepsilon_\text{train}=3{\cdot}10^{-3}$)",
                marker="o", lw=1.4, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"evaluation $\varepsilon$")
    ax.set_ylabel(r"empirical soft semi-dual")
    ax.axvline(3e-3, color="gray", ls="--", lw=0.8, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(3e-3, ymax, r" $\varepsilon_\text{train}$",
            ha="left", va="top", fontsize=8, color="gray")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------- 2. per-eval-time -----------------------------------------------

def per_eval_time(payloads):
    rows = {}
    for m, p in payloads.items():
        rows[m] = {t: p["per_t"][t] for t in (0.25, 0.75)}
    return rows


# ---------- 3. Sinkhorn divergence ---------------------------------------

def sinkhorn_log(C, eps, max_iter=2000, tol=1e-7):
    """Log-domain Sinkhorn with uniform empirical marginals; returns OT^eps."""
    B0, B1 = C.shape
    f = jnp.zeros(B0); g = jnp.zeros(B1)
    log_a = -jnp.log(B0); log_b = -jnp.log(B1)

    def body(carry, _):
        f, g, _, _ = carry
        # alternating soft c-transform updates (row-wise on f, then column-wise on g)
        f_new = -eps * (logsumexp(((g - C) / eps) + log_b, axis=-1))
        g_new = -eps * (logsumexp(((f_new[:, None] - C) / eps) + log_a, axis=0))
        df = jnp.max(jnp.abs(f_new - f)); dg = jnp.max(jnp.abs(g_new - g))
        return (f_new, g_new, df, dg), None

    (f, g, df, dg), _ = jax.lax.scan(
        body, (f, g, jnp.asarray(jnp.inf), jnp.asarray(jnp.inf)),
        None, length=max_iter,
    )
    ot_val = jnp.mean(f) + jnp.mean(g)
    return ot_val, df, dg


def sinkhorn_divergence_table(payloads, eps=3e-3):
    """Numerical OT^eps and S_eps on cached cost networks, per (method, k, c)."""
    rows = []
    h_y, alpha = 0.05, 0.05
    from sinkhorn_hti.lagrangian import stack_obs_per_condition
    for m, p in payloads.items():
        comps = p["comps"]; data = p["data"]
        ts = sorted(data.keys())
        ys_per_c = stack_obs_per_condition(data, conditions=SEMICIRCLE_CONDITIONS)
        for k in range(len(ts) - 1):
            for c_idx, c in enumerate(SEMICIRCLE_CONDITIONS):
                p0 = jnp.asarray(data[ts[k]][c]); p1 = jnp.asarray(data[ts[k + 1]][c])
                x = jnp.asarray([float(c)])
                ys_c = ys_per_c[c_idx]
                Cab = cost_matrix(comps, p0, p1, x, ys_c, h_y, alpha, n_quad=9)
                Caa = cost_matrix(comps, p0, p0, x, ys_c, h_y, alpha, n_quad=9)
                Cbb = cost_matrix(comps, p1, p1, x, ys_c, h_y, alpha, n_quad=9)
                ot_ab, _, _ = sinkhorn_log(Cab, eps)
                ot_aa, _, _ = sinkhorn_log(Caa, eps)
                ot_bb, _, _ = sinkhorn_log(Cbb, eps)
                S = float(ot_ab) - 0.5 * (float(ot_aa) + float(ot_bb))
                rows.append({
                    "method": m, "k": k, "c": c,
                    "OT_ab": float(ot_ab),
                    "OT_aa": float(ot_aa),
                    "OT_bb": float(ot_bb),
                    "S_eps": S,
                    "bias_abs": float(ot_ab) - S,
                })
    return rows


# ---------- main ---------------------------------------------------------

def main():
    payloads = load_pair()

    print("\n[1] eps-sweep")
    eps_grid, results = eps_sweep(payloads)
    out1 = FIGS / "eps_sweep.pdf"
    plot_eps_sweep(eps_grid, results, out1)
    print(f"    wrote {out1.relative_to(PROJECT)}")
    for m in payloads:
        means = results[m].mean(axis=1)
        stds = results[m].std(axis=1)
        print(f"    {m:>9s}  " + "  ".join(
            f"eps={eps:.0e}: {mean:+.3f}+-{std:.3f}"
            for eps, mean, std in zip(eps_grid, means, stds)
        ))

    print("\n[2] per-eval-time")
    rows3 = per_eval_time(payloads)
    for m, byt in rows3.items():
        for t, (nll, cd) in byt.items():
            print(f"    {m:>9s}  t={t:.2f}  NLL={nll:7.3f}  CD={cd:.4f}")

    print("\n[3] Sinkhorn divergence (eps=3e-3)")
    rows4 = sinkhorn_divergence_table(payloads, eps=3e-3)
    # aggregate
    for m in ("lbfgs", "entropic"):
        sub = [r for r in rows4 if r["method"] == m]
        ot_ab = np.mean([r["OT_ab"] for r in sub])
        ot_aa = np.mean([r["OT_aa"] for r in sub])
        ot_bb = np.mean([r["OT_bb"] for r in sub])
        s_eps = np.mean([r["S_eps"] for r in sub])
        bias = np.mean([r["bias_abs"] for r in sub])
        rel_bias = bias / abs(ot_ab) if abs(ot_ab) > 1e-12 else float("nan")
        print(f"    {m:>9s}  OT_eps(mu_k,mu_kp1)={ot_ab:+.4f}  "
              f"S_eps={s_eps:+.4f}  bias={bias:+.4f}  rel={rel_bias:+.3f}")

    return payloads, eps_grid, results, rows3, rows4


if __name__ == "__main__":
    main()
