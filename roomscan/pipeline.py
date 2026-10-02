"""Runs the steps in order for one capture: load, points, surfaces, rooms, per-room measurements and doors,
room connections, output. The video tier first turns a video into a Stray-layout capture
(roomscan/video_pipeline.py) and then runs the same steps with tier="video"."""
import time
from pathlib import Path

import numpy as np

from roomscan.connect_rooms import adjacency
from roomscan.find_doors_windows import capture_rays, find_openings
from roomscan.find_surfaces import CaptureError, classify_planes, extract_planes
from roomscan.load_capture import is_stray, load_stray
from roomscan.measurements import measure_room
from roomscan.point_cloud import estimate_normals, fuse_points
from roomscan.room_outline import manhattan_angle, outline_from_mask
from roomscan.room_surfaces import surfaces_for_room
from roomscan.save_plan import build_plan
from roomscan.split_rooms import split_rooms


VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
MIN_VIDEO_SECONDS = 2.0


def detect_tier(path):
    """What was handed in: a Stray Scanner folder -> lidar; a video file, or a folder holding one video -> video."""
    path = Path(path)
    if path.is_dir() and is_stray(path):
        return "lidar", path
    from roomscan.photo_folders import is_photo_set
    if is_photo_set(path):
        return "photo", path
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        return "video", path
    if path.is_dir():
        vids = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT)
        if len(vids) == 1:
            return "video", vids[0]
    raise SystemExit(f"{path}: no capture found (expected a Stray Scanner folder or one video file; "
                     f"photo folders: photo tier)")


def check_video_length(video):
    from roomscan.video_frames import video_info
    n, fps = video_info(video)
    if n / fps < MIN_VIDEO_SECONDS:
        raise CaptureError(f"video too short ({n / fps:.1f} s): walk through the rooms for at least "
                           f"{MIN_VIDEO_SECONDS:.0f} s")
    return n, fps


def camera_path(cap, angle, step=0.05):
    """Plan-view points along the walked path (camera centres, densified): known walkable floor."""
    from roomscan.room_outline import to_plan
    c = np.array([f.T_wc[:3, 3] for f in cap.frames])
    seg = [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / step))) for a, b in zip(c[:-1], c[1:])]
    return to_plan(np.concatenate(seg) if seg else c, angle)


def run_lidar(capture_dir, drift_correction=False, tier="lidar", stride=5, capture_id=None, walkable_path=False):
    t0 = time.time()
    cap = load_stray(capture_dir)
    drift = None
    if drift_correction:
        from roomscan.drift_correction import correct_drift
        cap, drift = correct_drift(cap)
    pts = fuse_points(cap, stride=stride)
    classes = classify_planes(extract_planes(pts, estimate_normals(pts)))
    angle = manhattan_angle(classes["walls"])
    masks, grid = split_rooms(classes, angle, extra_free=camera_path(cap, angle) if walkable_path else None)
    rays = list(capture_rays(cap, stride=stride))
    rooms = []
    for k, m in enumerate(masks):
        s = surfaces_for_room(classes, m, grid, angle)
        verts, edges = outline_from_mask(m, grid, s["walls"], angle)
        wall_h = (s["ceiling"].height if s["ceiling"] else s["floor"].height + 2.6) - s["floor"].height
        ops = find_openings(rays, edges, angle, s["floor"].height, wall_h)
        rooms.append(measure_room(s, verts, edges, angle, ops, tier=tier, room_id=f"room_{k}"))
    meta = {"n_frames": len(cap.frames), "n_points": int(len(pts)), "n_rooms": len(rooms),
            "manhattan_angle_rad": round(angle, 6), "drift_correction": drift_correction,
            "runtime_s": round(time.time() - t0, 1)}
    if drift:
        meta["drift"] = drift
    return build_plan(capture_id or Path(capture_dir).name, tier, rooms, meta, adjacency(rooms, masks, grid))


run_lidar_single = run_lidar
