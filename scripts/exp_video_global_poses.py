"""Experiment (candidate fix 3): chained VGGT chunks vs ONE VGGT pass on keyframes spread over the whole walk.
Pose quality vs LiDAR odometry after a similarity (scale + rotation + shift) alignment: absolute trajectory
error (metres, RMS) and median rotation error, plus how much of the walk each method keeps.

usage: python scripts/exp_video_global_poses.py <stray_capture_dir> <out_json> [n_keyframes ...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roomscan.load_capture import load_stray                         # noqa: E402
from roomscan.oracle import upright_frame                            # noqa: E402
from roomscan.video_frames import model_size, pick_sharpest          # noqa: E402
from roomscan.video_pipeline import FRAMES_PER_SEC                   # noqa: E402
from roomscan.video_poses import VGGTRunner, rot_angle_deg, run_chunks   # noqa: E402


def umeyama(src, dst):
    """s, R, t minimising |dst - (s R src + t)|."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def score(T_est, T_true):
    c_e = np.array([T[:3, 3] for T in T_est])
    c_t = np.array([T[:3, 3] for T in T_true])
    s, R, t = umeyama(c_e, c_t)
    ate = float(np.sqrt(np.mean(np.sum((c_e @ (s * R).T + t - c_t) ** 2, 1))))
    rot = [rot_angle_deg((R @ Te[:3, :3]).T @ Tt[:3, :3]) for Te, Tt in zip(T_est, T_true)]
    extent = float(np.ptp(c_t[:, [0, 2]], axis=0).max())
    return {"ate_m": round(ate, 3), "ate_pct_of_extent": round(100 * ate / extent, 1),
            "median_rot_err_deg": round(float(np.median(rot)), 2), "n_frames": len(T_est)}


def main():
    cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    ns = [int(x) for x in sys.argv[3:]] or [20]
    cap = load_stray(cap_dir)
    by = {f.index: f for f in cap.frames}
    video = cap_dir / "rgb.mp4"
    idx, imgs = pick_sharpest(video, per_sec=FRAMES_PER_SEC, size=model_size(video, 90), rotate=90)
    true = [upright_frame(cap.load_depth(i), by[i].K, by[i].T_wc)[2] for i in idx]
    runner = VGGTRunner(ROOT)
    res = {}
    t0 = time.time()
    merged = run_chunks(runner, imgs)
    seg = np.asarray(merged["seg"])
    order = np.asarray(merged["idx"])
    best = int(np.argmax(np.bincount(seg)))
    keep = np.where(seg == best)[0]
    res["chained_largest_segment"] = {**score([merged["T"][k] for k in keep], [true[order[k]] for k in keep]),
                                      "share_of_walk": round(len(keep) / len(idx), 3), "segments": int(merged["segments"]),
                                      "seconds": round(time.time() - t0, 1)}
    print("chained", res["chained_largest_segment"], flush=True)
    for n in ns:
        t0 = time.time()
        sel = np.linspace(0, len(idx) - 1, n).round().astype(int)
        r = runner.run(imgs[sel])
        res[f"global_{n}_keyframes"] = {**score(r["T"], [true[k] for k in sel]), "share_of_walk": 1.0,
                                        "seconds": round(time.time() - t0, 1)}
        print(f"global {n}", res[f"global_{n}_keyframes"], flush=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
