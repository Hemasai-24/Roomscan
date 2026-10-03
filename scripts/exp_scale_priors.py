"""Experiment: is "the camera is held ~1.4 m above the floor" a better metric-scale cue than Depth Anything?

For each reconstructed room/segment written with the TRUE scale (oracle modes scale / pose_scale), measure the
camera height above the detected floor. A prior-based scale would be off by 1.4 / h_true - 1; Depth Anything's
error for the same room is logged by the oracle runs (da_scale_err_pct).

usage: python scripts/exp_scale_priors.py <oracle_dir> [<oracle_dir> ...]
       (oracle_dir = .../oracle_scale or .../oracle_pose_scale; reads its eval.json and capture folders)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.find_surfaces import CaptureError, classify_planes, extract_planes   # noqa: E402
from roomscan.load_capture import load_stray                                      # noqa: E402
from roomscan.point_cloud import estimate_normals, fuse_points                     # noqa: E402

PRIOR = 1.4


def camera_height(cap_dir):
    cap = load_stray(cap_dir)
    cams = np.array([f.T_wc[:3, 3] for f in cap.frames])
    pts = fuse_points(cap, stride=1, max_depth=6.0)
    try:
        c = classify_planes(extract_planes(pts, estimate_normals(pts), min_inliers=800))
    except CaptureError:
        return None
    h = float(np.median(cams[:, 1]) - c["floor"].height)
    return h if h > 0.5 else None


def main():
    rows = []
    for d in map(Path, sys.argv[1:]):
        ev = json.loads((d / "eval.json").read_text())
        log = ev["summary"].get("per_room_scale") or {}
        for rid, info in log.items():
            cap_dir = d / "photo_rooms" / rid
            if not cap_dir.exists() or "da_scale_err_pct" not in info:
                continue
            h = camera_height(cap_dir)
            rows.append({"set": d.parent.name, "room": rid, "true_cam_height_m": None if h is None else round(h, 3),
                         "prior_scale_err_pct": None if h is None else round(100 * (PRIOR / h - 1), 1),
                         "da_scale_err_pct": info["da_scale_err_pct"]})
    for r in rows:
        print(r)
    ok = [r for r in rows if r["prior_scale_err_pct"] is not None]
    if ok:
        pe = np.abs([r["prior_scale_err_pct"] for r in ok])
        de = np.abs([r["da_scale_err_pct"] for r in ok])
        print(f"rooms with a floor: {len(ok)}/{len(rows)}; median |err| prior {np.median(pe):.1f}% vs "
              f"Depth Anything {np.median(de):.1f}%; prior better in {int((pe < de).sum())}/{len(ok)}")
    Path(sys.argv[1]).parent.joinpath("scale_priors.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
