from __future__ import annotations

from sinkhorn_hti.caching import load_run, save_run


def test_save_load_roundtrip(tmp_path):
    cfg = {"method": "lbfgs", "seed": 7, "alpha": 0.25}
    payload = {"logs": [1, 2, 3], "x": "abc"}
    pkl = tmp_path / "run.pkl"
    save_run(pkl, cfg, payload)

    assert pkl.exists()
    assert pkl.with_suffix(".json").exists()
    loaded, loaded_cfg = load_run(pkl)
    assert loaded == payload
    assert loaded_cfg == cfg


def test_load_run_no_sidecar_returns_none_cfg(tmp_path):
    import cloudpickle
    pkl = tmp_path / "run.pkl"
    pkl.write_bytes(cloudpickle.dumps({"toy": 42}))

    payload, cfg = load_run(pkl)
    assert payload == {"toy": 42}
    assert cfg is None
