"""Step 3: find flat surfaces (RANSAC planes) and label them floor / ceiling / wall / other."""
from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 1.0, 0.0])
MIN_CEILING_ABOVE_FLOOR = 1.9
FLOOR_MERGE = 0.05    # horizontal planes this close to the floor are the same floor (drift / tilt)


class CaptureError(ValueError):
    """The capture lacks what a floor plan needs (no floor, no walls); message is user-facing."""


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


def _ransac(pts, nrm, dist, iters, rng, agree, score_n=20000, batch=250):
    """Seeded single-threaded RANSAC: hypotheses scored on a fixed subsample. Returns inlier mask.
    With normals, a point supports a plane only if it faces the same way (|n_point . n_plane| > agree)."""
    idx = rng.choice(len(pts), min(score_n, len(pts)), replace=False)
    sub = pts[idx]
    sub_n = None if nrm is None else nrm[idx]
    best_n, best_cnt = None, -1
    for _ in range(0, iters, batch):
        tri = pts[rng.integers(0, len(pts), (batch, 3))]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-9
        n = n[ok] / norm[ok, None]
        d = -np.einsum("ij,ij->i", n, tri[ok, 0])
        good = np.abs(sub @ n.T + d) < dist
        if sub_n is not None:
            good &= np.abs(sub_n @ n.T) > agree
        cnt = good.sum(0)
        j = int(np.argmax(cnt))
        if cnt[j] > best_cnt:
            best_cnt, best_n = cnt[j], (n[j], d[j])
    n, d = best_n
    m = np.abs(pts @ n + d) < dist
    if nrm is not None:
        m &= np.abs(nrm @ n) > agree
    return m


def extract_planes(points, normals=None, dist=0.02, min_inliers=1500, max_planes=40, iters=2000, agree=0.85):
    rng = np.random.default_rng(0)
    rest = np.asarray(points, float)
    rest_n = None if normals is None else np.asarray(normals, float)
    planes = []
    for _ in range(max_planes):
        if len(rest) < min_inliers:
            break
        m = _ransac(rest, rest_n, dist, iters, rng, agree)
        if m.sum() < min_inliers:
            break
        pts = rest[m]
        n, d, rms = _refit(pts)
        planes.append(Plane(n, d, pts, rms))
        rest = rest[~m]
        if rest_n is not None:
            rest_n = rest_n[~m]
    return planes


def classify_planes(planes):
    horiz = [p for p in planes if abs(p.normal @ UP) > 0.95]
    walls = [p for p in planes if abs(p.normal @ UP) < 0.1]
    for w in walls:
        w.kind = "wall"
    if not horiz:
        raise CaptureError("no floor found: keep the floor in view while scanning")
    if not walls:
        raise CaptureError("no walls found: point the phone at the walls while scanning")
    big = max(len(p.inliers) for p in horiz)
    floor0 = min((p for p in horiz if len(p.inliers) >= 0.2 * big), key=lambda p: p.height)
    parts = [p for p in horiz if abs(p.height - floor0.height) < FLOOR_MERGE]
    w = np.array([len(p.inliers) for p in parts], float)
    hs = np.array([p.height for p in parts])
    between = float(np.sqrt(np.sum(w * (hs - np.average(hs, weights=w)) ** 2) / w.sum()))
    pts = np.concatenate([p.inliers for p in parts])
    n, d, rms = _refit(pts)
    spread = max(between, float(np.std(pts[:, 1])))   # gravity-aligned height scatter; a tilted fit would hide a step
    floor = Plane(n, d, pts, rms, "floor")
    for p in parts:
        p.kind = "merged"
    cands = [p for p in horiz if p.height > floor.height + MIN_CEILING_ABOVE_FLOOR
             and len(p.inliers) >= 0.2 * big]
    ceiling = max(cands, key=lambda p: len(p.inliers)) if cands else None
    if ceiling is not None:
        ceiling.kind = "ceiling"
    other = [p for p in planes if p.kind == "other"]
    return {"floor": floor, "ceiling": ceiling, "walls": walls, "other": other, "floor_spread": spread,
            "horizontal": horiz}
