"""Doors and windows measured from the image: a detector draws a box around each door/window; rays through
the box's left/right and top/bottom edges are intersected with the wall the opening sits on, giving width,
height and sill height in metres. Works for closed doors (which depth alone cannot see as openings).
The same opening seen in several views is merged (median size, n_views)."""
import numpy as np

from roomscan.measurements import TIER_REL, Z95
from roomscan.room_outline import to_plan

EXTENT_MARGIN = 0.3      # m: a hit this far beyond a wall's ends still counts as on that wall
MERGE_DIST = 0.4         # m along the wall: same opening in two views
EDGE_PX = 3.0            # px: box-edge uncertainty


def _ray(K, T, u, v):
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    return T[:3, 3], T[:3, :3] @ d


def _hit(o, d, wall):
    n = wall["normal"] / np.linalg.norm(wall["normal"])
    den = float(n @ d)
    if abs(den) < 1e-9:
        return None, None
    t = -(float(n @ o) + wall["d"]) / den
    if t <= 0:
        return None, None
    return o + t * d, t


def _along(p, wall):
    ext = wall["extent"]
    q = to_plan(p[None], ext["angle"])[0]
    return float(q[1] if ext["axis"] == "u" else q[0])


def _on_wall(p, wall):
    ext = wall["extent"]
    a = _along(p, wall)
    return ext["lo"] - EXTENT_MARGIN <= a <= ext["hi"] + EXTENT_MARGIN


def _measure_one(det, K, T, walls):
    x0, y0, x1, y1 = det["box"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    o, d = _ray(K, T, cx, cy)
    best = None
    for w in walls:
        p, t = _hit(o, d, w)
        if p is not None and _on_wall(p, w) and (best is None or t < best[1]):
            best = (w, t)
    if best is None:
        return None
    w = best[0]
    pts = {}
    for name, (u, v) in {"l": (x0, cy), "r": (x1, cy), "t": (cx, y0), "b": (cx, y1)}.items():
        p, _ = _hit(*_ray(K, T, u, v), w)
        if p is None:
            return None
        pts[name] = p
    width = abs(_along(pts["r"], w) - _along(pts["l"], w))
    height = abs(pts["t"][1] - pts["b"][1])
    label = det["label"].lower()
    return {"wall": w, "type": "door" if "door" in label else "window", "width": width, "height": height,
            "sill": pts["b"][1] - w["floor_h"], "centre": (_along(pts["l"], w) + _along(pts["r"], w)) / 2,
            "px_m": best[1] / K[0, 0], "score": det["score"]}


def _measure(v, sigma, unit="m"):
    h = Z95 * sigma
    return {"value": round(v, 4), "lo": round(max(v - h, 0.0), 4), "hi": round(v + h, 4), "unit": unit}


def measure_openings(views, walls, tier="lidar"):
    """views: [{"K", "T_wc", "detections": [{"label", "score", "box": (x0, y0, x1, y1)}]}] (labels door/window);
    walls: surfaces as built for damage (normal, d, floor_h, extent). Returns merged openings with ranges."""
    walls = [w for w in walls if w["kind"] == "wall"]
    found = [m for v in views for det in v["detections"]
             if ("door" in det["label"].lower() or "window" in det["label"].lower())
             for m in [_measure_one(det, v["K"], v["T_wc"], walls)] if m is not None]
    groups = []
    for m in found:
        for g in groups:
            if g[0]["wall"] is m["wall"] and g[0]["type"] == m["type"] and abs(g[0]["centre"] - m["centre"]) < MERGE_DIST:
                g.append(m)
                break
        else:
            groups.append([m])
    out = []
    rel = TIER_REL[tier]
    for g in groups:
        med = {k: float(np.median([m[k] for m in g])) for k in ("width", "height", "sill", "centre", "px_m")}
        edge = EDGE_PX * med["px_m"]
        s_w = np.hypot(np.hypot(edge, edge), max(rel * med["width"], 0.005))
        s_h = np.hypot(np.hypot(edge, edge), max(rel * med["height"], 0.005))
        out.append({"wall_id": g[0]["wall"]["surface_id"], "room_id": g[0]["wall"]["room_id"], "type": g[0]["type"],
                    "centre_along": round(med["centre"], 4), "width": _measure(med["width"], s_w),
                    "height": _measure(med["height"], s_h), "sill_height": _measure(max(med["sill"], 0.0), s_h),
                    "n_views": len(g), "score": max(m["score"] for m in g), "source": "image"})
    return out
