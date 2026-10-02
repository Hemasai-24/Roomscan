import json

import numpy as np

from roomscan.video_scale import load_bias, scale_from_depths


def _frames(n=8, seed=0):
    rng = np.random.default_rng(seed)
    true = [rng.uniform(1.0, 4.0, (20, 30)) for _ in range(n)]
    rel = [t / 3.0 * rng.normal(1, 0.01, t.shape) for t in true]       # VGGT: right shape, size /3
    metric = [1.4 * t * rng.normal(1, 0.03, t.shape) for t in true]     # metric model reads 40% far
    conf = [np.full(t.shape, 5.0) for t in true]
    return rel, metric, conf


def test_scale_recovered_after_bias():
    rel, metric, conf = _frames()
    s, spread = scale_from_depths(rel, metric, conf, bias=1.4)
    assert abs(s - 3.0) / 3.0 < 0.01
    assert spread < 0.02


def test_low_confidence_pixels_ignored():
    rel, metric, conf = _frames()
    for r, c in zip(rel, conf):
        r[:10] = 50.0          # garbage
        c[:10] = 0.5           # ... marked low confidence
    s, _ = scale_from_depths(rel, metric, conf, bias=1.4)
    assert abs(s - 3.0) / 3.0 < 0.01


def test_load_bias_reads_calibration(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"depth_anything_v2_metric_indoor_large": {"bias": 1.39}}))
    assert load_bias(p) == 1.39
    assert load_bias(tmp_path / "missing.json") == 1.0
