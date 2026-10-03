"""Photo tier, steps P2-P3: one room's photos -> a metric, level, Stray-layout capture (VGGT poses + depth,
Depth Anything scale) -> that ONE room measured by the LiDAR back end.

"One room" = the floor region the cameras stand in. Photos also see through doors into other rooms; those
regions are not this room. When the floor is not in view, it is assumed 1.4 m below the cameras (people hold
a phone at chest height); when the geometry still fails, the room is the box around nearby points, with
very wide ranges. Both cases add a warning."""
import numpy as np

from roomscan.find_doors_windows import capture_rays, find_openings
from roomscan.find_surfaces import CaptureError, Plane, classify_planes, extract_planes
from roomscan.load_capture import load_stray
from roomscan.measurements import measure_room
from roomscan.pipeline import camera_path
from roomscan.point_cloud import estimate_normals, fuse_points
from roomscan.room_outline import manhattan_angle, outline_from_mask, to_plan
from roomscan.room_surfaces import surfaces_for_room
from roomscan.split_rooms import split_rooms
from roomscan.video_capture import estimate_up, level, refine_up, write_capture
from roomscan.video_scale import scale_from_depths

CAMERA_HEIGHT_PRIOR = 1.4      # metres above floor
MAX_DEPTH = 6.0                # photos look across rooms
MIN_INLIERS = 800
FALLBACK_REL = 0.5             # +-50 % ranges when only a box around the points is known
MAX_PHOTO_CEILING = 3.5        # a higher "ceiling" from a few photos is almost always a mis-fit plane


def _confidence_maps(conf):
    return [np.where(c >= np.median(c), 2, 0).astype(np.uint8) for c in conf]


def photos_to_capture(images, fx_exif, out_dir, runner, metric, bias):
    r = runner.run(images)
    K = np.median(np.asarray(r["K"]), axis=0).copy()
    source = "model"
    if fx_exif:
        K[0, 0] = K[1, 1] = fx_exif
        source = "exif"
    scale, spread = scale_from_depths(list(r["depth"]), metric.predict(images), list(r["conf"]), bias)
    T = [np.array(x, float) for x in r["T"]]
    for x in T:
        x[:3, 3] *= scale
    depth = [d * scale for d in r["depth"]]
    conf = _confidence_maps(r["conf"])
    T = level(T, estimate_up(T))
    write_capture(out_dir, depth, conf, T, K, fps=1.0, images=images)
    pts = fuse_points(load_stray(out_dir), stride=1, max_depth=MAX_DEPTH)
    up = refine_up(pts, estimate_normals(pts), np.array([0.0, 1.0, 0.0]))
    if np.degrees(np.arccos(np.clip(up[1], -1, 1))) > 0.3:
        T = level(T, up)
        write_capture(out_dir, depth, conf, T, K, fps=1.0, images=images)
    return out_dir, {"n_photos": len(T), "focal_px": round(float(K[0, 0]), 1), "focal_source": source,
                     "metric_scale": round(float(scale), 5), "metric_scale_spread": round(float(spread), 4)}


def _floor_prior(planes, pts, cams, warnings):
    """Classes with a floor assumed CAMERA_HEIGHT_PRIOR below the cameras (floor not in view). Any horizontal
    plane found stays a ceiling candidate (a ceiling must not become the floor)."""
    h = float(np.median(cams[:, 1])) - CAMERA_HEIGHT_PRIOR
    near = pts[np.abs(pts[:, 1] - h) < 0.1]
    if len(near) < 50:                       # nothing at floor height: the area under the cameras
        g = np.mgrid[-0.5:0.5:0.05, -0.5:0.5:0.05].reshape(2, -1).T
        near = np.concatenate([np.c_[c[0] + g[:, 0], np.full(len(g), h), c[2] + g[:, 1]] for c in cams])
    floor = Plane(np.array([0.0, 1.0, 0.0]), -h, near, 0.05, "floor")
    warnings.append(f"floor not seen in the photos: assumed {CAMERA_HEIGHT_PRIOR} m below the camera")
    horiz = [p for p in planes if abs(p.normal[1]) > 0.95]
    walls = [p for p in planes if abs(p.normal[1]) < 0.1]
    if not walls:
        raise CaptureError("no walls found")
    for w in walls:
        w.kind = "wall"
    cands = [p for p in horiz if p.height > h + 1.9]
    ceiling = max(cands, key=lambda p: len(p.inliers)) if cands else None
    return {"floor": floor, "ceiling": ceiling, "walls": walls, "other": [], "horizontal": horiz,
            "floor_spread": 0.05}


def _classes(planes, pts, cams, warnings):
    """Floor/ceiling/walls; the floor must be below the cameras, else it is the floor prior."""
    try:
        c = classify_planes(planes)
    except CaptureError as e:
        if "floor" not in str(e):
            raise
        return _floor_prior(planes, pts, cams, warnings)
    if c["floor"].height > float(np.median(cams[:, 1])) - 0.5:
        return _floor_prior(planes, pts, cams, warnings)
    return c


def _rect_room(u0, v0, u1, v1, room_id, warnings):
    """Schema room record for a box known only roughly (geometry failed)."""
    poly = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
    m = lambda v, unit="m": {"value": round(v, 4), "lo": round(v * (1 - FALLBACK_REL), 4),
                             "hi": round(v * (1 + FALLBACK_REL), 4), "unit": unit}
    walls = []
    for i in range(4):
        a, b = poly[i - 1], poly[i]
        walls.append({"id": f"{room_id}_w{i}", "surface_id": f"{room_id}_w{i}", "start": a, "end": b,
                      "length": m(float(np.hypot(b[0] - a[0], b[1] - a[1]))), "plane_supported": False})
    area = (u1 - u0) * (v1 - v0)
    warnings.append("room geometry not recovered from the photos: box around the points, +-50 % ranges")
    return {"id": room_id, "name": room_id, "polygon": poly, "walls": walls, "floor_area": m(area, "m2"),
            "perimeter": m(2 * (u1 - u0 + v1 - v0)),
            "ceiling_height": {"value": 2.5, "lo": 1.9, "hi": 3.2, "unit": "m"}, "ceiling_observed": False,
            "openings": [], "warnings": list(warnings)}


def _fallback(pts, cams, angle, room_id, warnings):
    q, c = to_plan(pts, angle), to_plan(cams, angle)
    near = q[np.min(np.linalg.norm(q[:, None] - c[None], axis=2), axis=1) < 4.0] if len(q) else q
    if len(near) < 20:
        near = np.concatenate([c - 1.0, c + 1.0])
    (u0, v0), (u1, v1) = np.percentile(near, 5, axis=0), np.percentile(near, 95, axis=0)
    return _rect_room(float(u0), float(v0), float(u1), float(v1), room_id, warnings)


def measure_photo_room(cap_dir, room_id, tier="photo"):
    cap = load_stray(cap_dir)
    cams = np.array([f.T_wc[:3, 3] for f in cap.frames])
    fwd = np.array([f.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0]) for f in cap.frames])
    pts = fuse_points(cap, stride=1, max_depth=MAX_DEPTH)
    nrm = estimate_normals(pts)
    planes = extract_planes(pts, nrm, min_inliers=MIN_INLIERS)
    warnings = []
    out = {"angle": 0.0, "warnings": warnings}
    try:
        classes = _classes(planes, pts, cams, warnings)
        angle = manhattan_angle(classes["walls"])
        out["angle"] = angle
        masks, grid = split_rooms(classes, angle, extra_free=camera_path(cap, angle))
        cq = to_plan(cams, angle)
        ij = grid.cells(cq)
        hit = [m for m in masks if any(0 <= x < m.shape[1] and 0 <= y < m.shape[0] and m[y, x] for x, y in ij)]
        if not hit:
            raise CaptureError("cameras are not inside any recovered floor region")
        mask = np.logical_or.reduce(hit)
        s = surfaces_for_room(classes, mask, grid, angle)
        if s["ceiling"] is not None and s["ceiling"].height - s["floor"].height > MAX_PHOTO_CEILING:
            warnings.append(f"ceiling reading {s['ceiling'].height - s['floor'].height:.2f} m is implausible "
                            f"for a few photos (> {MAX_PHOTO_CEILING} m): reported as not observed")
            s["ceiling"] = None
        verts, edges = outline_from_mask(mask, grid, s["walls"], angle)
        wall_h = (s["ceiling"].height if s["ceiling"] else s["floor"].height + 2.6) - s["floor"].height
        ops = find_openings(capture_rays(cap, stride=1, max_depth=MAX_DEPTH), edges, angle, s["floor"].height,
                            wall_h)
        room = measure_room(s, verts, edges, angle, ops, tier=tier, room_id=room_id)
        room["floor_level"] = None          # each photo room has its own frame: levels not comparable
        room["warnings"] = warnings + room["warnings"]
        out["geometry"] = {"s": s, "edges": edges, "verts": verts, "angle": angle}
        out["cap_dir"] = cap_dir
    except (CaptureError, ValueError) as e:
        warnings.append(f"geometry failed: {e}")
        room = _fallback(pts, cams, out["angle"], room_id, warnings)
    out.update(room=room, cameras_plan=to_plan(cams, out["angle"]), forward_plan=to_plan(fwd, out["angle"]))
    out["warnings"] = room["warnings"]
    return out
