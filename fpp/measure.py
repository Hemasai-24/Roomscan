"""Assemble a room record (schema 'room') from geometry results."""
import numpy as np
from shapely.geometry import Polygon

from fpp.uncertainty import TIER_REL, Measure, edge_sigma, interval

UNSEEN_CEILING_MAX = 2.9    # prior: residential ceilings rarely exceed this
UNSEEN_CEILING_MARGIN = 0.3


def _ceiling(classes, tier, calib, warnings):
    floor, ceil = classes["floor"], classes["ceiling"]
    if ceil is not None:
        s = np.hypot(floor.rms / np.sqrt(len(floor.inliers) / 50), ceil.rms / np.sqrt(len(ceil.inliers) / 50))
        s = max(np.hypot(s, classes.get("floor_spread", 0.0)), 0.005 if tier == "lidar" else 0.02)
        h = ceil.height - floor.height
        return interval(h, np.hypot(s, TIER_REL[tier] * h), calib=calib), True
    top = max(float(np.percentile(w.inliers[:, 1], 99.5)) for w in classes["walls"]) - floor.height
    warnings.append(f"ceiling not observed: height is at least {top:.2f} m (highest wall point)")
    lo, hi = top - 0.01, max(top + UNSEEN_CEILING_MARGIN, UNSEEN_CEILING_MAX)
    return Measure((lo + hi) / 2, lo, hi), False


def measure_room(classes, verts, edges, angle, openings, tier="lidar", calib=1.0, room_id="room_0"):
    warnings = []
    n = len(edges)
    walls = []
    for i, e in enumerate(edges):
        L = abs(e.end - e.start)
        s_prev = edge_sigma(edges[i - 1], tier)
        s_next = edge_sigma(edges[(i + 1) % n], tier)
        sig = np.sqrt(s_prev ** 2 + s_next ** 2 + (TIER_REL[tier] * L) ** 2)
        walls.append({"id": f"{room_id}_w{i}", "surface_id": f"{room_id}_w{i}",
                      "start": [round(float(x), 4) for x in verts[i - 1]],
                      "end": [round(float(x), 4) for x in verts[i]],
                      "length": interval(L, sig, calib=calib).to_dict(),
                      "plane_supported": e.support > 0})
    poly = Polygon(verts)
    perim = poly.length
    s_off = np.mean([edge_sigma(e, tier) for e in edges])
    area = poly.area
    area_sig = np.hypot(perim * s_off / np.sqrt(2), 2 * TIER_REL[tier] * area)
    ceiling, observed = _ceiling(classes, tier, calib, warnings)
    ops = []
    for j, o in enumerate(openings):
        s_w = np.hypot(0.01 if tier == "lidar" else 0.03, TIER_REL[tier] * o.width)
        ops.append({"id": f"{room_id}_o{j}", "wall_id": f"{room_id}_w{o.edge_index}", "type": o.kind,
                    "offset_along_wall": round(o.s0, 4),
                    "width": interval(o.width, s_w, calib=calib).to_dict(),
                    "height": interval(o.height, s_w, calib=calib).to_dict(),
                    "sill_height": interval(o.bottom, s_w, calib=calib).to_dict()})
    return {"id": room_id, "name": room_id,
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in verts],
            "walls": walls,
            "floor_area": interval(area, area_sig, unit="m2", calib=calib).to_dict(),
            "perimeter": interval(perim, 2 * s_off * np.sqrt(n), calib=calib).to_dict(),  # each offset moves 2 walls
            "ceiling_height": ceiling.to_dict(), "ceiling_observed": observed,
            "openings": ops, "warnings": warnings}
