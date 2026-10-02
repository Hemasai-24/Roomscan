"""Deterministic iterative RANSAC planes + floor/ceiling/wall classification."""
from dataclasses import dataclass

import numpy as np

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


def _ransac(pts, dist, iters, rng, score_n=20000, batch=250):
    """Seeded single-threaded RANSAC: hypotheses scored on a fixed subsample. Returns inlier mask."""
    sub = pts[rng.choice(len(pts), min(score_n, len(pts)), replace=False)]
    best_n, best_cnt = None, -1
    for _ in range(0, iters, batch):
        tri = pts[rng.integers(0, len(pts), (batch, 3))]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-9
        n = n[ok] / norm[ok, None]
        d = -np.einsum("ij,ij->i", n, tri[ok, 0])
        cnt = (np.abs(sub @ n.T + d) < dist).sum(0)
        j = int(np.argmax(cnt))
        if cnt[j] > best_cnt:
            best_cnt, best_n = cnt[j], (n[j], d[j])
    n, d = best_n
    return np.abs(pts @ n + d) < dist


def extract_planes(points, dist=0.02, min_inliers=1500, max_planes=30, iters=2000):
    rng = np.random.default_rng(0)
    rest = np.asarray(points, float)
    planes = []
    for _ in range(max_planes):
        if len(rest) < min_inliers:
            break
        m = _ransac(rest, dist, iters, rng)
        if m.sum() < min_inliers:
            break
        pts = rest[m]
        n, d, rms = _refit(pts)
        planes.append(Plane(n, d, pts, rms))
        rest = rest[~m]
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
