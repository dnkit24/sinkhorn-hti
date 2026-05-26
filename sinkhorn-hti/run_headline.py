"""Train both headline variants at paper budget on semicircles.

    python run_headline.py             # both, ~7.5h
    python run_headline.py lbfgs       # L-BFGS only, ~6.5h
    python run_headline.py entropic    # entropic only, ~1h
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))

import jax
import numpy as np

from sinkhorn_hti.caching import save_run
from sinkhorn_hti.config import (
    EntropicConfig, LagrangianConfig, LBFGSInnerConfig, OptimConfig, TrainConfig,
)
from sinkhorn_hti.data.semicircles import SEMICIRCLE_TIMES, build_training_set
from sinkhorn_hti.metrics import evaluate_semicircles
from sinkhorn_hti.networks import HTIComponents
from sinkhorn_hti.train import train

RUN_DIR = FOLDER / "runs"

SEED = 5
N_OUTER = 2001
N_INNER = 10
BATCH_SIZE = 64
LBFGS_MAXITER = 10
ALPHA = 0.05
H_Y = 0.05


def run_one(method: str, epsilon: float) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    data = build_training_set(jax.random.PRNGKey(SEED), n_per_condition=100)
    comps = HTIComponents.init(
        jax.random.PRNGKey(SEED + 10_000),
        n_intervals=len(SEMICIRCLE_TIMES) - 1, dy=2, dx=1,
    )
    optim_cfg = OptimConfig()

    print(f"=== {method} α={ALPHA} h_y={H_Y} N={N_OUTER} B={BATCH_SIZE} maxiter={LBFGS_MAXITER} eps={epsilon} ===", flush=True)

    t0 = time.time()
    comps, logs = train(
        comps, data,
        train_cfg=TrainConfig(
            n_outer=N_OUTER, n_inner=N_INNER, batch_size=BATCH_SIZE,
            method=method, seed=SEED, log_every=50,
        ),
        lag_cfg=LagrangianConfig(h_y=H_Y, alpha=ALPHA),
        optim_cfg=optim_cfg,
        ent_cfg=EntropicConfig(epsilon=epsilon),
        lbfgs_cfg=LBFGSInnerConfig(maxiter=LBFGS_MAXITER),
    )
    wall = time.time() - t0

    per_t = evaluate_semicircles(comps, data)
    nll = float(np.mean([v[0] for v in per_t.values()]))
    cd = float(np.mean([v[1] for v in per_t.values()]))
    print(f"{method}: NLL={nll:.3f} CD={cd:.4f} wall={wall:.0f}s", flush=True)

    cfg = {
        "method": method, "n_outer": N_OUTER, "n_inner": N_INNER,
        "lbfgs_maxiter": LBFGS_MAXITER, "alpha": ALPHA, "h_y": H_Y,
        "epsilon": epsilon, "seed": SEED, "batch_size": BATCH_SIZE,
        "lr_G": optim_cfg.lr_G, "grad_clip": optim_cfg.grad_clip,
    }
    payload = {"comps": comps, "logs": logs, "data": data, "cfg": cfg,
               "NLL": nll, "CD": cd, "train_time_s": wall, "per_t": per_t}
    pkl = RUN_DIR / f"{method}_seed{SEED}.pkl"
    save_run(pkl, cfg, payload)
    print(f"saved -> {pkl}", flush=True)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "lbfgs"):
        run_one("lbfgs", epsilon=0.01)
    if which in ("both", "entropic"):
        run_one("entropic", epsilon=3e-3)


if __name__ == "__main__":
    main()
