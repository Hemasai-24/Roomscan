"""Deterministic iterative RANSAC planes + floor/ceiling/wall classification."""
from dataclasses import dataclass

import numpy as np
import open3d as o3d

UP = np.array([0.0, 1.0, 0.0])
MIN_CEILING_ABOVE_FLOOR = 1.9


@dataclass
class Plane:
    normal: np.ndarray
    d: float
    inliers: np.ndarray
    rms: float
    kind: str = "other"

    @property
    def height(self) -> float:
        return -self.d / self.normal[1]


def _refit(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    if n[1] < 0 or (abs(n[1]) < 0.5 and n[0] + n[2] < 0):
        n = -n
    d = -float(n @ c)
    rms = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
    return n, d, rms


def extract_planes(points, dist=0.02, min_inliers=1500, max_planes=30):
    o3d.utility.random.seed(0)
    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    planes = []
    for _ in range(max_planes):
        if len(rest.points) < min_inliers:
            break
        _, idx = rest.segment_plane(dist, 3, 2000)
        if len(idx) < min_inliers:
            break
        pts = np.asarray(rest.points)[idx]
        n, d, rms = _refit(pts)
        planes.append(Plane(n, d, pts, rms))
        rest = rest.select_by_index(idx, invert=True)
    return planes


def classify_planes(planes):
    horiz = [p for p in planes if abs(p.normal @ UP) > 0.95]
    walls = [p for p in planes if abs(p.normal @ UP) < 0.1]
    for w in walls:
        w.kind = "wall"
    big = max(len(p.inliers) for p in horiz)
    floor = min((p for p in horiz if len(p.inliers) >= 0.2 * big), key=lambda p: p.height)
    floor.kind = "floor"
    cands = [p for p in horiz if p.height > floor.height + MIN_CEILING_ABOVE_FLOOR
             and len(p.inliers) >= 0.2 * big]
    ceiling = max(cands, key=lambda p: len(p.inliers)) if cands else None
    if ceiling is not None:
        ceiling.kind = "ceiling"
    other = [p for p in planes if p.kind == "other"]
    return {"floor": floor, "ceiling": ceiling, "walls": walls, "other": other}
