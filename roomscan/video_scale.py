"""Video tier, step V3: real-world size. VGGT's reconstruction has the right shape but unknown size; a
metric depth model (Depth Anything V2, metric indoor) gives metres. On the sample phone it reads ~39% too
far, so its bias is calibrated on LiDAR sample captures (scripts/calibrate_metric_depth.py)."""
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

CALIBRATION = Path(__file__).with_name("calibration.json")
KEY = "depth_anything_v2_metric_indoor_large"


def load_bias(path=CALIBRATION):
    try:
        return float(json.loads(Path(path).read_text())[KEY]["bias"])
    except (OSError, KeyError, ValueError):
        warnings.warn(f"no metric-depth calibration at {path}; using bias 1.0 (scale may be off)")
        return 1.0


def scale_from_depths(rel, metric, conf, bias):
    """Metres per VGGT unit = median over frames of median(metric/bias / rel) on that frame's confident half.
    Returns (scale, spread) with spread the robust relative std of the per-frame ratios."""
    per = []
    for r, m, c in zip(rel, metric, conf):
        if m.shape != r.shape:
            m = cv2.resize(m, (r.shape[1], r.shape[0]))
        ok = (r > 1e-6) & (m > 0.1) & (c >= np.median(c))
        if ok.sum() > 50:
            per.append(float(np.median(m[ok] / bias / r[ok])))
    per = np.array(per)
    s = float(np.median(per))
    spread = float(1.4826 * np.median(np.abs(per - s)) / s)
    return s, spread


class MetricDepth:
    """Depth Anything V2 Metric-Indoor-Large (transformers), fp16 on GPU."""

    def __init__(self, root=Path(__file__).resolve().parents[1]):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        w = root / "weights" / "da2-metric-indoor-large"
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(w)
        self.model = AutoModelForDepthEstimation.from_pretrained(w).eval().cuda().half()

    def predict(self, images):
        out = []
        for img in images:
            inp = self.proc(images=img, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                d = self.model(pixel_values=inp["pixel_values"].half()).predicted_depth[0].float().cpu().numpy()
            out.append(cv2.resize(d, (img.shape[1], img.shape[0])))
        return out
