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


def intrinsics(fx, shape):
    """Pinhole K for an image of shape (h, w) with focal fx (px) and the principal point at the centre."""
    h, w = shape[:2]
    return np.array([[fx, 0.0, w / 2], [0.0, fx, h / 2], [0.0, 0.0, 1.0]])


class MetricDepth:
    """Depth Anything V2 Metric-Indoor-Large (transformers), fp16 on GPU. Ignores the focal length."""

    def __init__(self, root=Path(__file__).resolve().parents[1]):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        w = root / "weights" / "da2-metric-indoor-large"
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(w)
        self.model = AutoModelForDepthEstimation.from_pretrained(w).eval().cuda().half()

    def predict(self, images, fx=None):
        out = []
        for img in images:
            inp = self.proc(images=img, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                d = self.model(pixel_values=inp["pixel_values"].half()).predicted_depth[0].float().cpu().numpy()
            out.append(cv2.resize(d, (img.shape[1], img.shape[0])))
        return out


class UniDepthMetric:
    """UniDepth V2 (ViT-S): metric depth that uses the camera intrinsics when given (EXIF focal), else its own
    estimate. Measured on the LiDAR sample with no calibration: depth ratio 0.998 (per-frame std 0.11) vs
    Depth Anything 1.425 (std 0.30). Code: lpiccinelli-eth/UniDepth (CC BY-NC 4.0), fetched to third_party/."""

    def __init__(self, root=Path(__file__).resolve().parents[1]):
        import sys
        import types
        import torch
        for code in (root / "third_party" / "UniDepth", root / "exp_third" / "UniDepth-main"):
            if code.exists():
                sys.path.insert(0, str(code))
                break
        if "wandb" not in sys.modules:                 # imported by UniDepth for training logs only
            import importlib.machinery
            stub = types.ModuleType("wandb")
            stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
            sys.modules["wandb"] = stub
        from unidepth.models import UniDepthV2
        w = root / "weights" / "unidepth-v2-vits14"
        self.torch = torch
        self.model = UniDepthV2.from_pretrained(str(w) if w.exists() else "lpiccinelli/unidepth-v2-vits14")
        self.model = self.model.cuda().eval()

    def predict(self, images, fx=None):
        out = []
        for img in images:
            t = self.torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).cuda()
            K = None if fx is None else self.torch.tensor(intrinsics(fx, img.shape), dtype=self.torch.float32).cuda()
            with self.torch.no_grad():
                d = (self.model.infer(t, K) if K is not None else self.model.infer(t))["depth"][0, 0]
            out.append(d.float().cpu().numpy())
        return out


METRIC_MODELS = {"depth_anything": (MetricDepth, KEY), "unidepth": (UniDepthMetric, None)}


def metric_model(root=Path(__file__).resolve().parents[1]):
    """(model, bias) chosen by ROOMSCAN_METRIC_DEPTH (default depth_anything). A model without a
    calibration key is used as is (bias 1.0)."""
    import os
    name = os.environ.get("ROOMSCAN_METRIC_DEPTH", "depth_anything")
    if name not in METRIC_MODELS:
        raise ValueError(f"ROOMSCAN_METRIC_DEPTH={name!r}: choose one of {sorted(METRIC_MODELS)}")
    make, key = METRIC_MODELS[name]
    return make(root), (load_bias() if key else 1.0)
