"""Damage masks -> measured damage on building surfaces (area, width, height, height above floor).

Each masked pixel with depth becomes a 3D point; points are assigned to the nearest wall, floor or
ceiling plane (within 5 cm) — damage on furniture lands on no plane and is dropped. A mask that
crosses a corner splits into one record per surface. The same damage seen in several views
(same class, same surface, centres within 30 cm) is merged."""
import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon

from roomscan.measurements import TIER_REL, Z95
from roomscan.room_outline import to_plan

ON_SURFACE = 0.05     # m
MIN_PIXELS = 20
MERGE_DIST = 0.40     # m (a view that sees only part of a stain shifts its centre)
UP = np.array([0.0, 1.0, 0.0])
EXTENT_MARGIN = 0.10  # m beyond a wall's ends / a room's outline still counts as on that surface
MARGIN_PENALTY = 0.02


def _inside_extent(pts, ext, margin=EXTENT_MARGIN):
    """Optional surface extent: walls {"angle", "axis", "lo", "hi"} (along-wall range in plan coords),
    floors/ceilings {"angle", "polygon"} (room outline in plan coords)."""
    q = to_plan(pts, ext["angle"])
    if "polygon" in ext:
        poly = Polygon(ext["polygon"]).buffer(margin)
        return shapely.contains_xy(poly, q[:, 0], q[:, 1])
    along = q[:, 1] if ext["axis"] == "u" else q[:, 0]
    return (along >= ext["lo"] - margin) & (along <= ext["hi"] + margin)


def _basis(normal, kind):
    if kind == "wall":
        t1 = np.cross(UP, normal)
    else:
        t1 = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
    t1 = t1 / np.linalg.norm(t1)
    return t1, np.cross(normal, t1)


def _pixels(view, mask):
    depth, K, T = view["depth"], view["K"], view["T_wc"]
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    v, u = np.nonzero(mask & (depth > 0.1))
    z = depth[v, u]
    dc = np.c_[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(z)]
    pts = (dc * z[:, None]) @ T[:3, :3].T + T[:3, 3]
    rays = dc @ T[:3, :3].T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    cos_a = 1.0 / np.linalg.norm(dc, axis=1)
    return pts, rays, z, cos_a, K


def _record(cls, score, s, pts, rays, z, cos_a, K, tier):
    n = s["normal"] / np.linalg.norm(s["normal"])
    cos_t = np.clip(np.abs(rays @ n), 0.2, 1.0)
    area = float(np.sum(z ** 2 * cos_a / (K[0, 0] * K[1, 1] * cos_t)))
    t1, t2 = _basis(n, s["kind"])
    pix = float(np.median(z / K[0, 0]))
    a, b = pts @ t1, pts @ t2
    width = float(np.percentile(a, 99) - np.percentile(a, 1)) + pix
    height = float(np.percentile(b, 99) - np.percentile(b, 1)) + pix
    n_px = len(pts)
    w_px, h_px = max(width / pix, 1), max(height / pix, 1)
    boundary_rel = 2 * (w_px + h_px) / n_px              # +-0.5 px all round the outline
    scale_rel = TIER_REL[tier]
    area_rel = np.hypot(boundary_rel, 2 * scale_rel)

    def m(v, rel, unit="m"):
        h = Z95 * rel * v
        return {"value": round(v, 4), "lo": round(v - h, 4), "hi": round(v + h, 4), "unit": unit}

    bottom = float(np.percentile(pts @ UP, 1)) - s.get("floor_h", 0.0)
    return {"class": cls, "score": float(score), "surface_id": s["surface_id"], "room_id": s["room_id"],
            "area": m(area, area_rel, "m2"), "width": m(width, np.hypot(1 / w_px, scale_rel)),
            "height": m(height, np.hypot(1 / h_px, scale_rel)),
            "bottom_above_floor": {"value": round(bottom, 4), "lo": round(bottom - pix * 2, 4),
                                   "hi": round(bottom + pix * 2, 4), "unit": "m"},
            "center": [round(float(x), 4) for x in pts.mean(0)], "n_views": 1, "near_opening": False}


COPLANAR_COS, COPLANAR_D = 0.98, 0.05


def _plane_groups(surfaces):
    """Group index per surface: surfaces on the same physical plane (a jagged outline cuts one wall into
    several edges; neighbouring rooms can share a wall face) share a group."""
    n = np.array([s["normal"] / np.linalg.norm(s["normal"]) for s in surfaces])
    d = np.array([s["d"] for s in surfaces], float)
    group = list(range(len(surfaces)))
    for i in range(len(surfaces)):
        for j in range(i):
            c = float(n[i] @ n[j])
            if abs(c) > COPLANAR_COS and abs(d[i] - np.sign(c) * d[j]) < COPLANAR_D:
                group[i] = group[j]
                break
    return np.array(group)


def _per_view(view, surfaces, tier, groups):
    out = []
    planes = np.array([s["normal"] / np.linalg.norm(s["normal"]) for s in surfaces])
    ds = np.array([s["d"] for s in surfaces])
    for det in view["detections"]:
        pts, rays, z, cos_a, K = _pixels(view, det["mask"])
        if len(pts) < MIN_PIXELS:
            continue
        dist = np.abs(pts @ planes.T + ds)
        for k, srf in enumerate(surfaces):
            if "extent" in srf:
                loose = _inside_extent(pts, srf["extent"])
                strict = _inside_extent(pts, srf["extent"], margin=0.0)
                dist[~loose, k] = np.inf
                dist[loose & ~strict, k] += MARGIN_PENALTY      # prefer the surface whose own extent holds it
        best = dist.argmin(1)
        ok = dist[np.arange(len(pts)), best] < ON_SURFACE
        for g in np.unique(groups[best[ok]]):            # one region per physical plane: its biggest surface
            members = ok & (groups[best] == g)
            best[members] = np.bincount(best[members]).argmax()
        for k in np.unique(best[ok]):
            sel = ok & (best == k)
            if sel.sum() >= MIN_PIXELS:
                r = _record(det["class"], det["score"], surfaces[k], pts[sel], rays[sel], z[sel], cos_a[sel], K, tier)
                r["_plane"] = int(groups[k])
                out.append(r)
    return out


def _merge(records):
    groups = []
    for r in records:
        for g in groups:
            if g[0]["class"] == r["class"] and g[0]["_plane"] == r["_plane"] and \
                    np.linalg.norm(np.subtract(g[0]["center"], r["center"])) < MERGE_DIST:
                g.append(r)
                break
        else:
            groups.append([r])
    out = []
    for i, g in enumerate(groups):
        best = dict(g[int(np.argsort([x["area"]["value"] for x in g])[len(g) // 2])])   # median-area view
        best.pop("_plane")
        best["n_views"] = len(g)
        best["score"] = max(x["score"] for x in g)
        best["id"] = f"dmg_{i}"
        out.append(best)
    return out


def measure_damage(views, surfaces, tier="lidar"):
    """views: [{"depth", "K", "T_wc", "detections": [{"class", "score", "mask"}]}];
    surfaces: [{"surface_id", "room_id", "kind", "normal", "d", "floor_h"}] with plane n.x + d = 0."""
    if not surfaces:
        return []
    groups = _plane_groups(surfaces)
    return _merge([r for v in views for r in _per_view(v, surfaces, tier, groups)])
