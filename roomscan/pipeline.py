"""Runs steps 1-7 in order for one LiDAR capture."""
import time
from pathlib import Path

from roomscan.point_cloud import fuse_points
from roomscan.find_doors_windows import capture_rays, find_openings
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.room_outline import footprint
from roomscan.load_capture import load_stray
from roomscan.measurements import measure_room
from roomscan.save_plan import build_plan


def run_lidar_single(capture_dir):
    t0 = time.time()
    cap = load_stray(capture_dir)
    pts = fuse_points(cap)
    classes = classify_planes(extract_planes(pts))
    verts, edges, angle = footprint(classes)
    wall_h = (classes["ceiling"].height if classes["ceiling"] else classes["floor"].height + 2.6) \
        - classes["floor"].height
    ops = find_openings(capture_rays(cap), edges, angle, classes["floor"].height, wall_h)
    room = measure_room(classes, verts, edges, angle, ops, tier="lidar")
    meta = {"n_frames": len(cap.frames), "n_points": int(len(pts)),
            "manhattan_angle_rad": round(angle, 6), "runtime_s": round(time.time() - t0, 1)}
    return build_plan(Path(capture_dir).name, "lidar", [room], meta)
