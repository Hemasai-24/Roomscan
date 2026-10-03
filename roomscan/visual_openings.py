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
BORDER_FRAC = 0.01       # a box this close to the photo edge is cut off: not measurable
# plausible sizes (m): width, height, sill ranges per type
PLAUSIBLE = {"door": {"width": (0.6, 1.2), "height": (1.8, 2.4), "sill": (-0.2, 0.15)},
             "window": {"width": (0.3, 2.5), "height": (0.3, 2.0), "sill": (0.3, 1.6)}}


def cut_sides(box, size):
    w, h = size
    x0, y0, x1, y1 = box
    return {"side": x0 <= BORDER_FRAC * w or x1 >= (1 - BORDER_FRAC) * w,
            "vertical": y0 <= BORDER_FRAC * h or y1 >= (1 - BORDER_FRAC) * h}


def _plausible(m):
    r = PLAUSIBLE[m["type"]]
    keys = ("width",) if m.get("height_unknown") else ("width", "height", "sill")
    return all(r[k][0] <= m[k] <= r[k][1] for k in keys)


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


def _edges(det):
    """Edge midpoints in the camera image: given (turned frames), or from an axis-aligned upright box."""
    if "edges" in det:
        return det["edges"]
    x0, y0, x1, y1 = det["box"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"l": (x0, cy), "r": (x1, cy), "t": (cx, y0), "b": (cx, y1), "c": (cx, cy)}


def _measure_one(det, K, T, walls):
    e = _edges(det)
    o, d = _ray(K, T, *e["c"])
    best = None
    for w in walls:
        p, t = _hit(o, d, w)
        if p is not None and _on_wall(p, w) and (best is None or t < best[1]):
            best = (w, t)
    if best is None:
        return None
    w = best[0]
    pts = {}
    for name in ("l", "r", "t", "b"):
        u, v = e[name]
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
    found = []
    for v in views:
        for det in v["detections"]:
            if not ("door" in det["label"].lower() or "window" in det["label"].lower()):
                continue
            cut = det.get("cut") or (cut_sides(det["box"], v["size"]) if "size" in v else {"side": False, "vertical": False})
            if cut["side"]:                      # left/right edge outside the photo: width unknown
                continue
            m = _measure_one(det, v["K"], v["T_wc"], walls)
            if m is None:
                continue
            m["height_unknown"] = cut["vertical"]  # top/bottom outside: width is still measured
            if _plausible(m):
                found.append(m)
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
        med = {k: float(np.median([m[k] for m in g])) for k in ("width", "centre", "px_m")}
        edge = EDGE_PX * med["px_m"]
        s_w = np.hypot(np.hypot(edge, edge), max(rel * med["width"], 0.005))
        o = {"wall_id": g[0]["wall"]["surface_id"], "room_id": g[0]["wall"]["room_id"], "type": g[0]["type"],
             "centre_along": round(med["centre"], 4), "width": _measure(med["width"], s_w),
             "n_views": len(g), "score": max(m["score"] for m in g), "source": "image"}
        full = [m for m in g if not m["height_unknown"]]   # height/sill only from views showing the whole frame
        if full:
            hh, sill = float(np.median([m["height"] for m in full])), float(np.median([m["sill"] for m in full]))
            s_h = np.hypot(np.hypot(edge, edge), max(rel * hh, 0.005))
            o.update(height=_measure(hh, s_h), sill_height=_measure(max(sill, 0.0), s_h))
        out.append(o)
    return out
