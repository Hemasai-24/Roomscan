"""Damage for a whole capture: pick views, detect damage, measure it on the rooms' own surfaces, filter
false alarms, then concealed-damage rules and repair scope (Plan 5, task 4). Same for all three tiers:
every tier hands over a Stray-layout capture (depth + pose + intrinsics per frame) and its rooms' planes."""
import gc
from pathlib import Path

import cv2
import numpy as np

from roomscan.damage_detect import upright_turns
from roomscan.damage_measure import measure_damage
from roomscan.damage_rules import concealed_flags
from roomscan.repair_scope import scope_items
from roomscan.room_outline import to_plan

VIEWS_PER_SEC = 2.0
MAX_VIEWS = 150
DETECT_WIDTH = 1280          # px; thin cracks need the detail (Stray video is 1920 wide)
NEAR_OPENING = 0.20          # m between damage edge and an opening side
MIN_VIEWS = {"lidar": 2, "video": 2, "photo": 1}
ALL = {"water_stain", "crack", "mold", "peeling_paint", "hole"}
# From the false-alarm review on the (undamaged) sample flat: ceiling lights/vents read as "hole"; shiny floor
# tiles (reflections, grout) read as stains, cracks and holes, so floors keep only mould; the skirting-board
# joint at the wall base reads as "crack"/"stain", so wall damage lying entirely in that band is dropped.
ALLOWED = {"wall": ALL, "ceiling": ALL - {"hole"}, "floor": {"mold"}}
SKIRTING_BAND = 0.12         # m above the floor


def wall_plane(edge, angle):
    """World plane n.x + d = 0 of an outline edge (plan u = c*x - s*z, v = s*x + c*z with c, s of -angle)."""
    c, s = np.cos(-angle), np.sin(-angle)
    n = np.array([c, 0.0, -s]) if edge.axis == "u" else np.array([s, 0.0, c])
    return n, -float(edge.offset)


WALL_MATCH = 0.30            # m: a fitted wall plane this close to an outline edge is that edge's real surface


def _fitted_plane(edge, walls, angle):
    """The fitted wall plane behind an outline edge (outlines are not always snapped onto the wall face):
    same direction, offset within WALL_MATCH, overlapping along the edge. None if no plane matches."""
    k = 0 if edge.axis == "u" else 1
    lo, hi = min(edge.start, edge.end), max(edge.start, edge.end)
    best = None
    for w in walls:
        n = to_plan(np.asarray(w.normal, float)[None], angle)[0]
        if abs(n[k]) < 0.9:
            continue
        q = to_plan(w.inliers, angle)
        off = float(np.median(q[:, k]))
        overlap = min(hi, q[:, 1 - k].max()) - max(lo, q[:, 1 - k].min())
        if abs(off - edge.offset) < WALL_MATCH and overlap > 0.2 * (hi - lo):
            if best is None or abs(off - edge.offset) < best[0]:
                best = (abs(off - edge.offset), w)
    return None if best is None else best[1]


def room_surface_list(room_id, s, edges, angle, verts):
    fh = float(s["floor"].height)
    out = []
    for i, e in enumerate(edges):
        w = _fitted_plane(e, s.get("walls", []), angle)
        n, d = (np.asarray(w.normal, float), float(w.d)) if w is not None else wall_plane(e, angle)
        out.append({"surface_id": f"{room_id}_w{i}", "room_id": room_id, "kind": "wall", "normal": n, "d": d,
                    "floor_h": fh, "extent": {"angle": angle, "axis": e.axis, "lo": min(e.start, e.end),
                                              "hi": max(e.start, e.end)}})
    poly = [[float(x), float(y)] for x, y in verts]
    for kind in ("floor", "ceiling"):
        p = s.get(kind)
        if p is not None:
            out.append({"surface_id": f"{room_id}_{kind}", "room_id": room_id, "kind": kind,
                        "normal": np.asarray(p.normal, float), "d": float(p.d), "floor_h": fh,
                        "extent": {"angle": angle, "polygon": poly}})
    return out


def pick_view_indices(cap, per_sec=VIEWS_PER_SEC, max_views=MAX_VIEWS):
    ts = np.array([f.timestamp for f in cap.frames])
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    step = max(1, int(round(1.0 / (per_sec * max(dt, 1e-6)))))
    idx = [f.index for f in cap.frames[::step]]
    if len(idx) > max_views:
        idx = [idx[k] for k in np.linspace(0, len(idx) - 1, max_views).round().astype(int)]
    return idx


def _frame_images(cap_dir, indices, depth_size):
    """Colour image per frame index, at the depth image's aspect: packaged captures keep rgb/*.jpg,
    Stray exports have rgb.mp4 (decoded once, sequentially)."""
    cap_dir = Path(cap_dir)
    if (cap_dir / "rgb").is_dir():
        return {i: cv2.cvtColor(cv2.imread(str(cap_dir / "rgb" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
                for i in indices}
    from roomscan.video_frames import read_frames
    w, h = depth_size
    size = (DETECT_WIDTH, int(round(DETECT_WIDTH * h / w)))
    want = set(indices)
    return {i: img for i, img in read_frames(cap_dir / "rgb.mp4", size) if i in want}


def damage_for_capture(cap, cap_dir, surfaces, tier, detect=None):
    if not surfaces:
        return []
    if detect is None:
        from roomscan.damage_detect import detect_damage as detect
    frames = {f.index: f for f in cap.frames}
    idx = pick_view_indices(cap)
    imgs = _frame_images(cap_dir, idx, cap.depth_size)
    views = []
    for i in idx:
        if i not in imgs:
            continue
        f = frames[i]
        k = upright_turns(f.T_wc)
        dets = detect(np.ascontiguousarray(np.rot90(imgs[i], k)))
        for d in dets:
            d["mask"] = np.ascontiguousarray(np.rot90(d["mask"], -k))
        views.append({"depth": cap.load_depth(i), "K": f.K, "T_wc": f.T_wc, "detections": dets})
    return measure_damage(views, surfaces, tier)


def _kind(surface_id):
    tail = surface_id.rsplit("_", 1)[-1]
    return tail if tail in ("floor", "ceiling") else "wall"


def filter_damage(damage, tier):
    out = []
    for d in damage:
        kind = _kind(d["surface_id"])
        if d["class"] not in ALLOWED[kind] or d["n_views"] < MIN_VIEWS[tier]:
            continue
        if kind == "wall" and d["bottom_above_floor"]["value"] + d["height"]["value"] < SKIRTING_BAND:
            continue
        d = dict(d)
        if tier == "photo":
            d["single_view"] = d["n_views"] == 1
        out.append(d)
    return out


def annotate(damage, rooms_by_id, angle_by_room):
    """Offset along the wall (m from the wall's start, for drawing) and near_opening (rule R4)."""
    for d in damage:
        d.setdefault("near_opening", False)
        if _kind(d["surface_id"]) != "wall":
            continue
        room = rooms_by_id[d["room_id"]]
        wall = next(w for w in room["walls"] if w["id"] == d["surface_id"])
        a, b = np.array(wall["start"], float), np.array(wall["end"], float)
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        q = to_plan(np.array([d["center"]], float), angle_by_room[d["room_id"]])[0]
        off = float((q - a) @ u)
        d["offset_along_wall"] = round(off, 4)
        half = d["width"]["value"] / 2
        for op in room.get("openings", []):
            if op["wall_id"] != wall["id"]:
                continue
            sides = (op["offset_along_wall"], op["offset_along_wall"] + op["width"]["value"])
            gap = min(abs(s - e) for s in sides for e in (off - half, off + half))
            inside = sides[0] <= off <= sides[1]
            if gap < NEAR_OPENING or inside:
                d["near_opening"] = True
    return damage


def finish_plan(plan, damage):
    for i, d in enumerate(damage):
        d["id"] = f"dmg_{i}"
    plan["damage"] = damage
    plan["concealed_damage_flags"] = concealed_flags(damage, plan["rooms"], plan["adjacency"])
    plan["scope_items"] = scope_items(damage, plan["concealed_damage_flags"], plan["rooms"])
    return plan


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def release():
    from roomscan.damage_detect import release_detectors
    release_detectors()
    free_gpu()


def box_to_original(box, orig_shape, k):
    """A box found in np.rot90(image, k) back in the original image's pixel coordinates."""
    h, w = orig_shape[:2]
    coords = np.rot90(np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1), k)
    rh, rw = coords.shape[:2]
    x0, y0, x1, y1 = box
    cs = [coords[int(np.clip(y, 0, rh - 1)), int(np.clip(x, 0, rw - 1))] for x in (x0, x1) for y in (y0, y1)]
    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def openings_for_capture(cap, cap_dir, surfaces, tier, detect=None):
    """Doors/windows measured from the image (roomscan/visual_openings.py) on the same views as damage."""
    from roomscan.visual_openings import measure_openings
    if not surfaces:
        return []
    if detect is None:
        from roomscan.damage_detect import detect_openings as detect
    frames = {f.index: f for f in cap.frames}
    idx = pick_view_indices(cap)
    imgs = _frame_images(cap_dir, idx, cap.depth_size)
    views = []
    for i in idx:
        if i not in imgs:
            continue
        f = frames[i]
        k = upright_turns(f.T_wc)
        img = imgs[i]
        dets = detect(np.ascontiguousarray(np.rot90(img, k)))
        sx, sy = cap.depth_size[0] / img.shape[1], cap.depth_size[1] / img.shape[0]   # boxes to depth-image pixels
        for d in dets:
            x0, y0, x1, y1 = box_to_original(d["box"], img.shape, k)
            d["box"] = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        views.append({"K": f.K, "T_wc": f.T_wc, "detections": dets, "size": cap.depth_size})
    return measure_openings(views, surfaces, tier)


def place_openings(rooms_by_id, found, tier):
    """Attach image-measured openings to their rooms. Photo/video: they replace the depth-carved openings of
    that room (mostly phantoms there). LiDAR: added only where no carved opening overlaps (e.g. closed doors)."""
    by_room = {rid: [] for rid in rooms_by_id} if tier in ("photo", "video") else {}
    for o in found:
        by_room.setdefault(o["room_id"], []).append(o)
    for rid, ops in by_room.items():
        room = rooms_by_id.get(rid)
        if room is None:
            continue
        walls = {w["id"]: w for w in room["walls"]}
        placed = []
        for o in ops:
            w = walls.get(o["wall_id"])
            if w is None:
                continue
            a, b = np.array(w["start"], float), np.array(w["end"], float)
            horizontal = abs(b[0] - a[0]) >= abs(b[1] - a[1])
            k = 0 if horizontal else 1
            sign = 1.0 if b[k] >= a[k] else -1.0
            off = (o["centre_along"] - a[k]) * sign - o["width"]["value"] / 2
            placed.append({"wall_id": o["wall_id"], "type": o["type"], "offset_along_wall": round(max(off, 0.0), 4),
                           "width": o["width"], "height": o["height"], "sill_height": o["sill_height"],
                           "source": "image", "n_views": o["n_views"]})
        if tier in ("photo", "video"):
            room["openings"] = placed
        else:
            keep = list(room["openings"])
            for p in placed:
                clash = any(c["wall_id"] == p["wall_id"] and
                            abs((c["offset_along_wall"] + c["width"]["value"] / 2) -
                                (p["offset_along_wall"] + p["width"]["value"] / 2)) < 0.5 for c in keep)
                if not clash:
                    keep.append(p)
            room["openings"] = keep
        for j, o in enumerate(room["openings"]):
            o["id"] = f"{rid}_o{j}"
