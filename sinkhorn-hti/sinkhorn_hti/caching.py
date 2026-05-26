import json
from pathlib import Path

import cloudpickle


def save_run(pkl_path, cfg, payload):
    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    pkl_path.write_bytes(cloudpickle.dumps(payload))
    pkl_path.with_suffix(".json").write_text(json.dumps(cfg, indent=2, default=list))


def load_run(pkl_path):
    pkl_path = Path(pkl_path)
    sidecar = pkl_path.with_suffix(".json")
    cfg = json.loads(sidecar.read_text()) if sidecar.exists() else None
    return cloudpickle.loads(pkl_path.read_bytes()), cfg
