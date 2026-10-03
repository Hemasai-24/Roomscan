"""Experiment: where does the photo tier's error come from? Re-run it with LiDAR "oracles" substituted.

modes:  baseline   - the shipped photo tier (VGGT poses+depth, Depth Anything scale)
        scale      - VGGT poses+depth, TRUE scale (median LiDAR/VGGT depth ratio)
        pose       - LiDAR camera poses, VGGT depth x Depth Anything scale
        pose_scale - LiDAR camera poses, VGGT depth x TRUE scale
        all        - LiDAR poses + LiDAR depth of the same photos (only the room-shape method is left)
        walls      - baseline reconstruction, room extent from fitted wall planes (candidate fix, see --extent)

usage: python scripts/exp_photo_oracle.py <make_photo_folders out_dir> <stray_capture_dir> <mode> [<mode> ...]
writes <out_dir>/oracle_<mode>/plan.json and prints one summary row per mode."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_photo_vs_lidar import evaluate                                   # noqa: E402
from roomscan.load_capture import load_stray                              # noqa: E402
from roomscan.oracle import frame_index, oracle_scale, upright_frame      # noqa: E402
from roomscan.photo_folders import _images                                # noqa: E402
from roomscan.photo_pipeline import run_photos                            # noqa: E402
from roomscan.photo_room import (_confidence_maps, measure_photo_room,    # noqa: E402
                                 photos_to_capture)
from roomscan.point_cloud import estimate_normals, fuse_points            # noqa: E402
from roomscan.save_plan import validate                                   # noqa: E402
from roomscan.video_capture import estimate_up, level, refine_up, write_capture   # noqa: E402
from roomscan.video_scale import scale_from_depths                       # noqa: E402

KEYS = ["rooms_photo", "rooms_lidar", "footprint_err_pct", "footprint_iou_after_alignment",
        "median_top_wall_err_pct", "adjacency_precision", "adjacency_recall", "max_overlap_m2",
        "wall_range_coverage", "area_range_coverage", "runtime_s"]


class Models:
    def __init__(self):
        from roomscan.video_poses import VGGTRunner
        from roomscan.video_scale import MetricDepth, load_bias
        self.runner, self.metric, self.bias = VGGTRunner(ROOT), MetricDepth(ROOT), load_bias()


def _lidar_frames(cap, rooms_dir, rid):
    by = {f.index: f for f in cap.frames}
    idx = [frame_index(p.name) for p in _images(rooms_dir / rid)]
    return [upright_frame(cap.load_depth(i), by[i].K, by[i].T_wc) for i in idx]


def _write_level(out_dir, depth, conf, T, K, relevel):
    if relevel:
        T = level(T, estimate_up(T))
    write_capture(out_dir, depth, conf, T, K, fps=1.0)
    if relevel:
        from roomscan.load_capture import load_stray as ls
        pts = fuse_points(ls(out_dir), stride=1, max_depth=6.0)
        up = refine_up(pts, estimate_normals(pts), np.array([0.0, 1.0, 0.0]))
        if np.degrees(np.arccos(np.clip(up[1], -1, 1))) > 0.3:
            T = level(T, up)
            write_capture(out_dir, depth, conf, T, K, fps=1.0)


def make_reconstruct(mode, models, cap, rooms_dir, log):
    def rec(imgs, fx, out_dir, rid):
        lid = _lidar_frames(cap, rooms_dir, rid)
        info = {}
        if mode in ("baseline", "walls"):
            out_dir, info = photos_to_capture(imgs, fx, out_dir, models.runner, models.metric, models.bias)
        elif mode == "all":
            depth = [d for d, _, _ in lid]
            conf = [np.where(d > 0, 2, 0).astype(np.uint8) for d in depth]
            write_capture(out_dir, depth, conf, [T for _, _, T in lid], lid[0][1], fps=1.0)
        else:
            r = models.runner.run(imgs)
            K = np.median(np.asarray(r["K"]), axis=0).copy()
            if fx:
                K[0, 0] = K[1, 1] = fx
            true = float(np.median([oracle_scale(r["depth"][k], r["conf"][k], lid[k][0]) for k in range(len(lid))]))
            da, _ = scale_from_depths(list(r["depth"]), models.metric.predict(imgs), list(r["conf"]), models.bias)
            s = true if mode in ("scale", "pose_scale") else da
            info = {"true_scale": true, "da_scale": da, "da_scale_err_pct": round(100 * (da / true - 1), 1)}
            depth = [d * s for d in r["depth"]]
            conf = _confidence_maps(r["conf"])
            if mode in ("pose", "pose_scale"):
                _write_level(out_dir, depth, conf, [T for _, _, T in lid], K, relevel=False)
            else:
                T = [np.array(x, float) for x in r["T"]]
                for x in T:
                    x[:3, 3] *= s
                _write_level(out_dir, depth, conf, T, K, relevel=True)
        log[rid] = info
        extent = "walls" if mode == "walls" else "floor"
        return measure_photo_room(out_dir, rid, tier="photo", **({"extent": extent} if mode == "walls" else {})), info
    return rec


def main():
    out, cap_dir, modes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    ref = json.loads((out / "lidar" / "plan.json").read_text())
    cap = load_stray(cap_dir)
    models = Models()
    table = {}
    for mode in modes:
        d = out / f"oracle_{mode}"
        log = {}
        plan = run_photos(out / "rooms", d, reconstruct=make_reconstruct(mode, models, cap, out / "rooms", log))
        validate(plan)
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text(json.dumps(plan, indent=2))
        rows, summary = evaluate(ref, plan)
        summary["per_room_scale"] = log
        (d / "eval.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2, default=float))
        table[mode] = {k: summary.get(k) for k in KEYS}
        print(mode, json.dumps(table[mode]), flush=True)
    (out / "oracle_table.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
