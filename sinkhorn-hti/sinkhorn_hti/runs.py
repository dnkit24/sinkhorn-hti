from pathlib import Path

import jax

from .caching import load_run
from .data.semicircles import SEMICIRCLE_TIMES, sample_anchors


def load_headline(method: str, seed: int = 5):
    pkl = Path(__file__).resolve().parent.parent / "runs" / f"{method}_seed{seed}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} missing; run run_headline.py")
    payload, _ = load_run(pkl)
    if "data" not in payload:
        payload["data"] = sample_anchors(
            n_per_condition=100,
            anchor_times=SEMICIRCLE_TIMES,
            key=jax.random.PRNGKey(seed),
        )
    return payload
