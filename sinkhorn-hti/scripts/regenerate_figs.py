"""Regenerate report figures, numbers.tex and table_results.tex from the
cached headline payloads in runs/. No retraining.

    python scripts/regenerate_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = Path(__file__).resolve().parent.parent
PROJECT = FOLDER.parent
sys.path.insert(0, str(FOLDER))

from sinkhorn_hti.caching import load_run
from sinkhorn_hti.data.semicircles import (
    SEMICIRCLE_CONDITIONS, SEMICIRCLE_TIMES, sample_anchors,
)
from sinkhorn_hti.lagrangian import cubic_spline_eval
from sinkhorn_hti.metrics import evaluate_semicircles

RUNS = FOLDER / "runs"
REPORT = PROJECT / "report"
FIGS = REPORT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

SEED = 5
ENT_LABEL = r"Sinkhorn-HTI, $\varepsilon=3\times 10^{-3}$"


def load_headline(method: str):
    pkl = RUNS / f"{method}_seed{SEED}.pkl"
    if not pkl.exists():
        raise RuntimeError(f"missing {pkl}; run run_headline.py")
    payload, _ = load_run(pkl)
    if "data" not in payload:
        payload["data"] = sample_anchors(
            n_per_condition=100, anchor_times=SEMICIRCLE_TIMES,
            key=jax.random.PRNGKey(SEED),
        )
    return payload


def training_curves(payloads):
    keys = ["dual", "map", "path", "metric"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharex=True)
    for method, color, label in (("lbfgs", "C0", "L-BFGS"),
                                  ("entropic", "C1", ENT_LABEL)):
        logs = payloads[method]["logs"]
        outer = [e["outer"] for e in logs]
        for ax, k in zip(axes, keys):
            ax.plot(outer, [e[k] for e in logs], color=color, lw=1.4, label=label)
    for ax, k in zip(axes, keys):
        ax.set_xlabel("outer iteration"); ax.set_title(k); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    out = FIGS / "training_curves.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.relative_to(PROJECT)}")


def _trajectory(comps, y0, x, anchors, ts):
    n_int = len(anchors) - 1
    endpoints = [y0]
    for k in range(n_int):
        endpoints.append(comps.maps[k](endpoints[-1], x))
    pts = []
    for t in ts:
        k = next((j for j in range(n_int - 1) if t <= anchors[j + 1]), n_int - 1)
        s = (t - anchors[k]) / (anchors[k + 1] - anchors[k])
        ctrl = comps.spline(endpoints[k], endpoints[k + 1], x)
        q, _ = cubic_spline_eval(ctrl, jnp.asarray([float(s)]))
        pts.append(np.asarray(q[0]))
    return np.stack(pts, axis=0)


def trajectories(payloads, n_traces=12, n_t=25):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    for row, (method, color, label) in enumerate((
        ("lbfgs", "C0", "L-BFGS"),
        ("entropic", "C1", ENT_LABEL),
    )):
        p = payloads[method]
        data = p["data"]
        anchors = sorted(data.keys())
        ts = np.linspace(0.0, 1.0, n_t)
        rng = np.random.default_rng(0)
        for ax, c in zip(axes[row], SEMICIRCLE_CONDITIONS):
            pool = np.asarray(data[anchors[0]][c])
            x = jnp.asarray([float(c)])
            for y0 in pool[rng.choice(pool.shape[0], n_traces, replace=False)]:
                traj = _trajectory(p["comps"], jnp.asarray(y0), x, anchors, ts)
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.55, lw=0.9)
            for t, a in zip(SEMICIRCLE_TIMES, [0.3, 0.6, 1.0]):
                ys = np.asarray(data[t][c])
                ax.scatter(ys[:, 0], ys[:, 1], s=4, alpha=a, color="k")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            if row == 0: ax.set_title(f"c={c}")
        axes[row, 0].set_ylabel(label)
    fig.tight_layout()
    out = FIGS / "trajectories_compare.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.relative_to(PROJECT)}")


def write_numbers(payloads):
    rows = {}
    per_t_rows = {}
    for method, p in payloads.items():
        per_t = evaluate_semicircles(p["comps"], p["data"])
        per_t_rows[method] = per_t
        rows[method] = {
            "NLL": sum(v[0] for v in per_t.values()) / len(per_t),
            "CD": sum(v[1] for v in per_t.values()) / len(per_t),
            "wall": p["logs"][-1]["wall_s"],
        }
    ratio = rows["lbfgs"]["wall"] / rows["entropic"]["wall"]
    n_outer = payloads["lbfgs"]["cfg"]["n_outer"]
    maxiter = payloads["lbfgs"]["cfg"]["lbfgs_maxiter"]

    (REPORT / "numbers.tex").write_text(
        f"\\newcommand{{\\nOuter}}{{{n_outer}}}\n"
        f"\\newcommand{{\\theSeed}}{{{SEED}}}\n"
        f"\\newcommand{{\\maxiterLBFGS}}{{{maxiter}}}\n"
        f"\\newcommand{{\\nllLbfgs}}{{{rows['lbfgs']['NLL']:.2f}}}\n"
        f"\\newcommand{{\\nllEnt}}{{{rows['entropic']['NLL']:.2f}}}\n"
        f"\\newcommand{{\\cdLbfgs}}{{{rows['lbfgs']['CD']:.4f}}}\n"
        f"\\newcommand{{\\cdEnt}}{{{rows['entropic']['CD']:.4f}}}\n"
        f"\\newcommand{{\\wallLbfgs}}{{{rows['lbfgs']['wall']:.0f}}}\n"
        f"\\newcommand{{\\wallEnt}}{{{rows['entropic']['wall']:.0f}}}\n"
        f"\\newcommand{{\\wallRatio}}{{{ratio:.1f}}}\n"
    )
    pt = per_t_rows
    (REPORT / "table_results.tex").write_text(
        "\\begin{tabular}{l c cc cc cc}\n"
        "\\toprule\n"
        "& Time (s) & \\multicolumn{2}{c}{Mean} & \\multicolumn{2}{c}{$t=0.25$} & \\multicolumn{2}{c}{$t=0.75$} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        "Method & $\\downarrow$ & NLL $\\downarrow$ & CD $\\downarrow$ & NLL $\\downarrow$ & CD $\\downarrow$ & NLL $\\downarrow$ & CD $\\downarrow$ \\\\\n"
        "\\midrule\n"
        f"L-BFGS inner solver & {rows['lbfgs']['wall']:.0f} "
        f"& {rows['lbfgs']['NLL']:.2f} & {rows['lbfgs']['CD']:.4f} "
        f"& {pt['lbfgs'][0.25][0]:.2f} & {pt['lbfgs'][0.25][1]:.4f} "
        f"& {pt['lbfgs'][0.75][0]:.2f} & {pt['lbfgs'][0.75][1]:.4f} \\\\\n"
        f"Sinkhorn-HTI, $\\varepsilon=3\\times 10^{{-3}}$ & {rows['entropic']['wall']:.0f} "
        f"& {rows['entropic']['NLL']:.2f} & {rows['entropic']['CD']:.4f} "
        f"& {pt['entropic'][0.25][0]:.2f} & {pt['entropic'][0.25][1]:.4f} "
        f"& {pt['entropic'][0.75][0]:.2f} & {pt['entropic'][0.75][1]:.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    for method, r in rows.items():
        print(f"  {method:>9s}  NLL={r['NLL']:7.2f}  CD={r['CD']:.4f}  wall={r['wall']:.0f}s")
    print(
        f"wrote report/numbers.tex (ratio {ratio:.1f}x, maxiter={maxiter}), "
        f"report/table_results.tex"
    )


def main():
    payloads = {m: load_headline(m) for m in ("lbfgs", "entropic")}
    training_curves(payloads)
    trajectories(payloads)
    write_numbers(payloads)


if __name__ == "__main__":
    main()
