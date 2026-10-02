"""Runs the steps in order for one LiDAR capture: load, points, surfaces, rooms, per-room
measurements and doors, room connections, output."""
import time
from pathlib import Path

from roomscan.connect_rooms import adjacency
from roomscan.find_doors_windows import capture_rays, find_openings
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.load_capture import load_stray
from roomscan.measurements import measure_room
from roomscan.point_cloud import estimate_normals, fuse_points
from roomscan.room_outline import manhattan_angle, outline_from_mask
from roomscan.room_surfaces import surfaces_for_room
from roomscan.save_plan import build_plan
from roomscan.split_rooms import split_rooms


def run_lidar(capture_dir, drift_correction=False):
    t0 = time.time()
    cap = load_stray(capture_dir)
    drift = None
    if drift_correction:
        from roomscan.drift_correction import correct_drift
        cap, drift = correct_drift(cap)
    pts = fuse_points(cap)
    classes = classify_planes(extract_planes(pts, estimate_normals(pts)))
    angle = manhattan_angle(classes["walls"])
    masks, grid = split_rooms(classes, angle)
    rays = list(capture_rays(cap))
    rooms = []
    for k, m in enumerate(masks):
        s = surfaces_for_room(classes, m, grid, angle)
        verts, edges = outline_from_mask(m, grid, s["walls"], angle)
        wall_h = (s["ceiling"].height if s["ceiling"] else s["floor"].height + 2.6) - s["floor"].height
        ops = find_openings(rays, edges, angle, s["floor"].height, wall_h)
        rooms.append(measure_room(s, verts, edges, angle, ops, tier="lidar", room_id=f"room_{k}"))
    meta = {"n_frames": len(cap.frames), "n_points": int(len(pts)), "n_rooms": len(rooms),
            "manhattan_angle_rad": round(angle, 6), "drift_correction": drift_correction,
            "runtime_s": round(time.time() - t0, 1)}
    if drift:
        meta["drift"] = drift
    return build_plan(Path(capture_dir).name, "lidar", rooms, meta, adjacency(rooms, masks, grid))


run_lidar_single = run_lidar
