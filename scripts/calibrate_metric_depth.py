"""Fit the metric-depth model's bias on LiDAR captures (where true depth is known).

Usage: python scripts/calibrate_metric_depth.py <stray_capture_dir> [<stray_capture_dir> ...]
Writes roomscan/calibration.json with the bias (median over captures), each capture's own ratio, and the
leave-one-capture-out error (fit on the others, test on this one) - the honest number to report.
Frames are stored sideways by Stray Scanner, so they are rotated upright exactly as the video tier does."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.load_capture import load_stray        # noqa: E402
from roomscan.video_frames import pick_sharpest     # noqa: E402
from roomscan.video_scale import CALIBRATION, KEY, MetricDepth   # noqa: E402

SIZE = (392, 518)


def capture_ratio(model, cap_dir, per_sec=0.5):
    cap = load_stray(cap_dir)
    idx, imgs = pick_sharpest(Path(cap_dir) / "rgb.mp4", per_sec=per_sec, size=SIZE, rotate=90)
    ratios = []
    for i, d in zip(idx, model.predict(imgs)):
        if i >= len(cap.frames):
            continue
        li = cv2.rotate(cap.load_depth(i), cv2.ROTATE_90_CLOCKWISE)
        li = cv2.resize(li, SIZE, interpolation=cv2.INTER_NEAREST)
        m = li > 0.2
        if m.sum() > 500:
            ratios.append(float(np.median(d[m] / li[m])))
    return float(np.median(ratios)), float(np.std(ratios)), len(ratios)


def main():
    model = MetricDepth()
    per = {}
    for c in sys.argv[1:]:
        r, sd, n = capture_ratio(model, c)
        per[Path(c).parent.name] = {"ratio": round(r, 4), "per_frame_std": round(sd, 4), "frames": n}
        print(f"{Path(c).parent.name}: metric/LiDAR = {r:.3f} (per-frame std {sd:.3f}, {n} frames)")
    names = list(per)
    loo = {}
    for k in names:
        others = [per[o]["ratio"] for o in names if o != k]
        if others:
            loo[k] = round(per[k]["ratio"] / float(np.median(others)) - 1, 4)
    bias = float(np.median([per[k]["ratio"] for k in names]))
    print(f"bias {bias:.3f}; leave-one-capture-out relative scale error: {loo}")
    data = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
    data[KEY] = {"bias": round(bias, 4), "per_capture": per, "leave_one_out_error": loo,
                 "note": "Depth Anything V2 Metric-Indoor-Large vs Stray Scanner LiDAR, upright 392x518 frames"}
    CALIBRATION.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
