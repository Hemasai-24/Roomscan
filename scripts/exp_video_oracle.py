"""Experiment: where does the video tier's error come from? Re-run it with LiDAR "oracles" substituted.

modes:  baseline   - the shipped video tier (chained VGGT chunks, Depth Anything scale, best segment)
        scale      - same chunks/segments, TRUE scale (median LiDAR/VGGT depth ratio)
        pose       - LiDAR camera poses for all picked frames, VGGT depth x Depth Anything scale
        pose_scale - LiDAR camera poses, VGGT depth x TRUE scale
        all        - LiDAR poses + LiDAR depth of the picked frames (only frame picking + back end left)

usage: python scripts/exp_video_oracle.py <stray_capture_dir> <out_dir> <mode> [<mode> ...]"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_video_vs_lidar import evaluate                                     # noqa: E402
from roomscan.draw_plan import render                                       # noqa: E402
from roomscan.find_surfaces import CaptureError, classify_planes, extract_planes   # noqa: E402
from roomscan.load_capture import load_stray                                # noqa: E402
from roomscan.oracle import oracle_scale, upright_frame                     # noqa: E402
from roomscan.pipeline import run_lidar                                     # noqa: E402
from roomscan.point_cloud import estimate_normals, fuse_points              # noqa: E402
from roomscan.video_capture import estimate_up, level, refine_up, write_capture   # noqa: E402
from roomscan.video_frames import model_size, pick_sharpest                 # noqa: E402
from roomscan.video_pipeline import (FRAMES_PER_SEC, SCALE_EVERY, _confidence_maps,   # noqa: E402
                                     pick_segment)
from roomscan.video_scale import scale_from_depths                         # noqa: E402

KEYS = ["video_rooms", "lidar_rooms", "video_area_m2", "lidar_area_m2", "matched_walls_ge_1m",
        "median_abs_err_pct_ge_1m", "within_3pct_ge_1m", "lidar_inside_video_range", "video_footprint_iou_with_lidar"]


def _relevel_and_write(cap_dir, depth, conf, T, K, imgs, relevel=True):
    if relevel:
        T = level(T, estimate_up(T))
    write_capture(cap_dir, depth, conf, T, K, fps=FRAMES_PER_SEC, images=imgs)
    pts = fuse_points(load_stray(cap_dir), stride=1)
    nrm = estimate_normals(pts)
    if relevel:
        up = refine_up(pts, nrm, np.array([0.0, 1.0, 0.0]))
        if np.degrees(np.arccos(np.clip(up[1], -1, 1))) > 0.3:
            T = level(T, up)
            write_capture(cap_dir, depth, conf, T, K, fps=FRAMES_PER_SEC, images=imgs)
            pts = fuse_points(load_stray(cap_dir), stride=1)
            nrm = estimate_normals(pts)
    return classify_planes(extract_planes(pts, nrm))["floor_spread"]


def build(mode, cap, frames_idx, imgs, out, runner, metric, bias):
    by = {f.index: f for f in cap.frames}
    lid = [upright_frame(cap.load_depth(i), by[i].K, by[i].T_wc) for i in frames_idx]
    info = {}
    if mode == "all":
        depth = [d for d, _, _ in lid]
        conf = [np.where(d > 0, 2, 0).astype(np.uint8) for d in depth]
        cap_dir = out / "capture"
        write_capture(cap_dir, depth, conf, [T for _, _, T in lid], lid[0][1], fps=FRAMES_PER_SEC, images=imgs)
        return cap_dir, info
    from roomscan.video_poses import run_chunks
    merged = run_chunks(runner, imgs)
    K = np.median(merged["K"], axis=0)
    order = np.asarray(merged["idx"])

    def scales(keep):
        sub = keep[::SCALE_EVERY]
        true = float(np.median([oracle_scale(merged["depth"][k], merged["conf"][k], lid[order[k]][0]) for k in sub]))
        da, _ = scale_from_depths([merged["depth"][k] for k in sub], metric.predict(imgs[order[sub]]),
                                  [merged["conf"][k] for k in sub], bias)
        return true, da

    if mode in ("pose", "pose_scale"):
        keep = np.arange(len(order))
        true, da = scales(keep)
        s = true if mode == "pose_scale" else da
        info = {"true_scale": true, "da_scale": da, "da_scale_err_pct": round(100 * (da / true - 1), 1),
                "frames_used": len(keep), "pose_segments": int(merged["segments"])}
        cap_dir = out / "capture"
        _relevel_and_write(cap_dir, [merged["depth"][k] * s for k in keep],
                           _confidence_maps([merged["conf"][k] for k in keep]),
                           [lid[order[k]][2] for k in keep], K, imgs[order[keep]], relevel=False)
        return cap_dir, info
    # baseline / scale: the shipped segment logic, scale substituted
    stats, infos = {}, {}
    for sg in np.argsort(-np.bincount(merged["seg"])):
        keep = np.where(merged["seg"] == sg)[0]
        if len(keep) < 12 and stats:
            continue
        true, da = scales(keep)
        s = true if mode == "scale" else da
        T = [merged["T"][k].copy() for k in keep]
        for Tk in T:
            Tk[:3, 3] *= s
        try:
            spread = _relevel_and_write(out / f"seg_{sg}", [merged["depth"][k] * s for k in keep],
                                        _confidence_maps([merged["conf"][k] for k in keep]), T, K, imgs[order[keep]])
        except (CaptureError, ValueError):
            continue
        stats[int(sg)] = (len(keep), spread)
        infos[int(sg)] = {"true_scale": true, "da_scale": da, "da_scale_err_pct": round(100 * (da / true - 1), 1)}
    best = pick_segment(stats)
    info = {"segment_used": best, "frames_used": stats[best][0], "frames_picked": len(order),
            "pose_segments": int(merged["segments"]), **infos[best]}
    return out / f"seg_{best}", info


def main():
    cap_dir, out_root, modes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    out_root.mkdir(parents=True, exist_ok=True)
    ref_path = out_root / "lidar_plan.json"
    if ref_path.exists():
        L = json.loads(ref_path.read_text())
    else:
        L = run_lidar(cap_dir)
        ref_path.write_text(json.dumps(L, indent=2))
    cap = load_stray(cap_dir)
    video = cap_dir / "rgb.mp4"
    idx, imgs = pick_sharpest(video, per_sec=FRAMES_PER_SEC, size=model_size(video, 90), rotate=90)
    from roomscan.video_poses import VGGTRunner
    from roomscan.video_scale import MetricDepth, load_bias
    runner, metric, bias = VGGTRunner(ROOT), MetricDepth(ROOT), load_bias()
    table = {}
    for mode in modes:
        out = out_root / f"oracle_{mode}"
        try:
            cdir, info = build(mode, cap, list(idx), imgs, out, runner, metric, bias)
            V = run_lidar(cdir, tier="video", stride=1, capture_id=f"video_{mode}", walkable_path=True)
        except CaptureError as e:
            print(mode, "FAILED", e, flush=True)
            table[mode] = {"error": str(e)}
            continue
        V["meta"]["video"] = info
        (out / "plan.json").write_text(json.dumps(V, indent=2))
        render(V, out / "plan")
        summary, rows = evaluate(L, V, cap_dir.parent.name)
        table[mode] = {**{k: summary.get(k) for k in KEYS}, **{k: info.get(k) for k in
                       ("da_scale_err_pct", "frames_used", "pose_segments")}}
        (out / "eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=float))
        print(mode, json.dumps(table[mode], default=float), flush=True)
    (out_root / "oracle_table.json").write_text(json.dumps(table, indent=2, default=float))


if __name__ == "__main__":
    main()
