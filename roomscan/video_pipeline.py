"""Video tier end to end: video -> sharp upright frames -> VGGT poses + relative depth (chunks) -> metric
scale -> level to gravity -> Stray-layout capture -> the same back end as LiDAR with tier="video"."""
import time
from pathlib import Path

import numpy as np

from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.load_capture import load_stray
from roomscan.pipeline import check_video_length, run_lidar
from roomscan.point_cloud import estimate_normals, fuse_points
from roomscan.video_capture import estimate_up, level, refine_up, write_capture
from roomscan.video_frames import pick_sharpest
from roomscan.video_poses import VGGTRunner, run_chunks
from roomscan.video_scale import MetricDepth, load_bias, scale_from_depths

FRAMES_PER_SEC = 3.0
SIZE = (392, 518)        # upright portrait; both multiples of VGGT's 14-px patch
SCALE_EVERY = 3          # metric depth on every 3rd kept frame (0.27 s each)


def pick_segment(stats, min_frames=12):
    """stats: {segment: (n_frames, floor_spread_m)}. Prefer the flattest floor among segments with enough
    frames (a smeared floor means the poses inside that piece disagree); fall back to the largest."""
    ok = {k: v for k, v in stats.items() if v[0] >= min_frames}
    if not ok:
        return max(stats, key=lambda k: stats[k][0])
    return min(ok, key=lambda k: ok[k][1])


def _confidence_maps(conf):
    """VGGT confidence -> Stray 0/2: keep the more confident half of each frame."""
    return [np.where(c >= np.median(c), 2, 0).astype(np.uint8) for c in conf]


def _segment_capture(cap_dir, merged, keep, imgs, metric_model, bias):
    """Scale, level and write one pose segment as a Stray-layout capture; return (floor_spread, info)."""
    T = [merged["T"][k].copy() for k in keep]
    depth = [merged["depth"][k] for k in keep]
    conf = [merged["conf"][k] for k in keep]
    K = np.median(merged["K"][keep], axis=0)
    frames_rgb = imgs[merged["idx"][keep]]
    sub = list(range(0, len(keep), SCALE_EVERY))
    scale, spread = scale_from_depths([depth[k] for k in sub], metric_model.predict(imgs[merged["idx"][keep]][sub]),
                                      [conf[k] for k in sub], bias)
    for Tk in T:
        Tk[:3, 3] *= scale
    depth = [d * scale for d in depth]
    T = level(T, estimate_up(T))
    confs = _confidence_maps(conf)
    write_capture(cap_dir, depth, confs, T, K, fps=FRAMES_PER_SEC, images=frames_rgb)
    pts = fuse_points(load_stray(cap_dir), stride=1)
    nrm = estimate_normals(pts)
    up = refine_up(pts, nrm, np.array([0.0, 1.0, 0.0]))
    if np.degrees(np.arccos(np.clip(up[1], -1, 1))) > 0.3:
        T = level(T, up)
        write_capture(cap_dir, depth, confs, T, K, fps=FRAMES_PER_SEC, images=frames_rgb)
        pts = fuse_points(load_stray(cap_dir), stride=1)
        nrm = estimate_normals(pts)
    floor_spread = classify_planes(extract_planes(pts, nrm))["floor_spread"]
    return floor_spread, {"metric_scale": round(scale, 5), "metric_scale_spread": round(spread, 4),
                          "focal_px": round(float(K[0, 0]), 1), "floor_spread_m": round(float(floor_spread), 4)}


def video_to_capture(video, out_dir, rotate=0, root=Path(__file__).resolve().parents[1]):
    t0 = time.time()
    check_video_length(video)
    idx, imgs = pick_sharpest(video, per_sec=FRAMES_PER_SEC, size=SIZE, rotate=rotate)
    merged = run_chunks(VGGTRunner(root), imgs)
    metric_model, bias = MetricDepth(root), load_bias()
    sizes = np.bincount(merged["seg"])
    stats, infos = {}, {}
    for sg in np.argsort(-sizes):
        keep = np.where(merged["seg"] == sg)[0]
        if len(keep) < 12 and stats:
            continue
        cand = Path(out_dir) / f"video_segment_{sg}"
        spread, info = _segment_capture(cand, merged, keep, imgs, metric_model, bias)
        stats[int(sg)], infos[int(sg)] = (len(keep), spread), info
    best = pick_segment(stats)
    cap_dir = Path(out_dir) / f"video_segment_{best}"
    warnings = []
    if merged["segments"] > 1:
        warnings.append(f"video poses split into {merged['segments']} pieces (mirror/glass or fast turns); "
                        f"using the most consistent one: {stats[best][0]} of {len(idx)} frames")
    info = {"frames_picked": len(idx), "frames_used": stats[best][0], "pose_segments": int(merged["segments"]),
            "segment_used": best, "segments": {k: {"frames": v[0], **infos[k]} for k, v in stats.items()},
            "metric_bias": bias, **infos[best], "video_to_capture_s": round(time.time() - t0, 1)}
    return cap_dir, info, warnings


def run_video(video, out_dir, rotate=0, damage=False, detect=None):
    cap_dir, info, warnings = video_to_capture(video, out_dir, rotate=rotate)
    plan = run_lidar(cap_dir, tier="video", stride=1, capture_id=Path(video).stem, walkable_path=True,
                     damage=damage, detect=detect)
    plan["meta"]["video"] = info
    plan["warnings"] = warnings + plan["warnings"]
    return plan
