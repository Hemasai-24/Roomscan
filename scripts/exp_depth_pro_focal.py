"""Experiment (candidate fix 1): focal-aware metric depth. Depth Pro's metric depth is raw * width / focal;
its own focal guess was +30 % off on the sample phone and its depth error followed it. With the known focal
(EXIF on real photos; Stray intrinsics here) that error should cancel. Compared per frame against LiDAR depth,
alongside Depth Anything V2 Metric-Indoor with our calibrated bias.

usage: python scripts/exp_depth_pro_focal.py <out_json> <stray_capture_dir> [...]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roomscan.load_capture import load_stray                     # noqa: E402
from roomscan.oracle import upright_frame                        # noqa: E402
from roomscan.video_frames import read_frames, sharpness         # noqa: E402
from roomscan.video_scale import MetricDepth, load_bias          # noqa: E402

W, H = 720, 960
N_FRAMES = 10


def _ratio(pred, lidar):
    p = cv2.resize(pred.astype(np.float32), lidar.shape[::-1], interpolation=cv2.INTER_AREA)
    ok = (lidar > 0.1) & (p > 0.05)
    return float(np.median(p[ok] / lidar[ok]))


def main():
    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
    dev = "cpu"                          # keep the GPU for the VGGT queue (8 GB)
    _ = free
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = DepthProImageProcessorFast.from_pretrained(ROOT / "weights" / "depth-pro")
    model = DepthProForDepthEstimation.from_pretrained(ROOT / "weights" / "depth-pro", torch_dtype=dtype).to(dev).eval()
    da, bias = MetricDepth(ROOT), load_bias()
    rows = []
    for cap_dir in map(Path, sys.argv[2:]):
        cap = load_stray(cap_dir)
        n = len(cap.frames)
        want = set(np.linspace(n * 0.05, n * 0.95, N_FRAMES).astype(int).tolist())
        for i, img in read_frames(cap_dir / "rgb.mp4", (W, H), rotate=90):
            if i not in want or i >= n:
                continue
            f = cap.frames[i]
            ld, K2, _ = upright_frame(cap.load_depth(f.index), f.K, f.T_wc)
            f_true = K2[0, 0] * W / ld.shape[1]                       # focal at the 720-px-wide image
            inp = proc(images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**{k: v.to(dtype) if v.is_floating_point() else v for k, v in inp.items()})
            raw = out.predicted_depth[0].float().cpu().numpy()
            fov = float(out.field_of_view[0].float().cpu())
            f_est = 0.5 * W / np.tan(np.radians(fov) / 2)
            raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
            dp_est = raw * W / f_est
            dp_true = raw * W / f_true
            d_any = da.predict(img[None])[0] / bias
            rows.append({"capture": cap_dir.parent.name, "frame": i, "sharp": round(sharpness(img), 1),
                         "f_true": round(f_true, 1), "f_est": round(f_est, 1),
                         "depth_pro_est_focal": round(_ratio(dp_est, ld), 4),
                         "depth_pro_true_focal": round(_ratio(dp_true, ld), 4),
                         "depth_anything_calibrated": round(_ratio(d_any, ld), 4)})
            print(rows[-1], flush=True)
    summ = {}
    for k in ("depth_pro_est_focal", "depth_pro_true_focal", "depth_anything_calibrated"):
        r = np.array([x[k] for x in rows])
        summ[k] = {"median_ratio": round(float(np.median(r)), 3),
                   "median_abs_err_pct": round(float(np.median(np.abs(r - 1))) * 100, 1),
                   "robust_spread_pct": round(float(1.4826 * np.median(np.abs(r - np.median(r))) / np.median(r)) * 100, 1)}
    summ["device"] = dev
    print(json.dumps(summ, indent=1))
    Path(sys.argv[1]).write_text(json.dumps({"summary": summ, "frames": rows}, indent=2))


if __name__ == "__main__":
    main()
