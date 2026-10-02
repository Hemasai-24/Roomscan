import time
from pathlib import Path

from fpp.geometry.fuse import fuse_points
from fpp.geometry.openings import capture_rays, find_openings
from fpp.geometry.planes import classify_planes, extract_planes
from fpp.geometry.room import footprint
from fpp.ingest.stray import load_stray
from fpp.measure import measure_room
from fpp.output import build_plan


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
