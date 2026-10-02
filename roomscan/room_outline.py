"""Step 4: room outline seen from above, squared to the main wall directions and snapped to fitted walls."""
from dataclasses import dataclass

import cv2
import numpy as np

SNAP_DIST = 0.15


@dataclass
class Edge:
    axis: str      # "u": line u = offset ; "v": line v = offset
    offset: float
    start: float
    end: float
    support: int
    rms: float


@dataclass
class Grid:
    """Top-down raster in plan coordinates: cell (col, row) = floor((q - lo) / cell)."""
    lo: np.ndarray
    cell: float
    shape: tuple

    def cells(self, q):
        return ((q - self.lo) / self.cell).astype(int)

    def raster(self, q):
        ij = self.cells(q)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[1]) & (ij[:, 1] < self.shape[0])
        m = np.zeros(self.shape, np.uint8)
        m[ij[ok, 1], ij[ok, 0]] = 1
        return m


def manhattan_angle(walls):
    ang = np.array([np.arctan2(w.normal[2], w.normal[0]) for w in walls])
    wts = np.array([len(w.inliers) for w in walls], float)
    return float(np.angle(np.sum(wts * np.exp(4j * ang))) / 4)


def to_plan(points, angle):
    c, s = np.cos(-angle), np.sin(-angle)
    x, z = points[:, 0], points[:, 2]
    return np.c_[c * x - s * z, s * x + c * z]


def _wall_lines(walls, angle):
    lines = []
    for w in walls:
        q = to_plan(w.inliers, angle)
        n = to_plan(w.normal[None], angle)[0]
        axis = "u" if abs(n[0]) > abs(n[1]) else "v"
        k = 0 if axis == "u" else 1
        lines.append((axis, float(np.median(q[:, k])), q[:, 1 - k].min(), q[:, 1 - k].max(),
                      len(w.inliers), w.rms))
    return lines


def _rectilinear(contour, eps):
    pts = cv2.approxPolyDP(contour, eps, True)[:, 0, :].astype(float)
    segs = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        axis = "u" if abs(b[0] - a[0]) < abs(b[1] - a[1]) else "v"   # "u": vertical segment, constant u
        off = (a[0] + b[0]) / 2 if axis == "u" else (a[1] + b[1]) / 2
        if segs and segs[-1][0] == axis:
            prev = segs.pop()
            off = (prev[1] + off) / 2
        segs.append((axis, off))
    if len(segs) > 1 and segs[0][0] == segs[-1][0]:
        a, b = segs.pop(), segs.pop(0)
        segs.insert(0, (a[0], (a[1] + b[1]) / 2))
    return segs


def _snap(segs, lines):
    """Snap each segment to the nearest collinear wall plane whose extent overlaps it."""
    out = []
    n = len(segs)
    for i, (axis, off) in enumerate(segs):
        a, b = sorted((segs[i - 1][1], segs[(i + 1) % n][1]))   # segment extent along its line
        cands = [l for l in lines if l[0] == axis and abs(l[1] - off) < SNAP_DIST
                 and min(b, l[3]) - max(a, l[2]) > 0.1 * (b - a)]
        if cands:
            best = min(cands, key=lambda l: (round(abs(l[1] - off), 2), -l[4]))
            out.append((axis, best[1], best[4], best[5]))
        else:
            out.append((axis, off, 0, 0.02))
    return out


def _drop_degenerate(segs, min_len=0.05):
    """Remove zero-length walls: if segments i-1 and i+1 are collinear, segment i (and i+1) vanish."""
    segs = list(segs)
    changed = True
    while changed and len(segs) > 4:
        changed = False
        for i in range(len(segs)):
            p, nx = segs[i - 1], segs[(i + 1) % len(segs)]
            if p[0] == nx[0] and abs(p[1] - nx[1]) < min_len:
                j = (i + 1) % len(segs)
                segs = [x for k, x in enumerate(segs) if k not in (i, j)]
                changed = True
                break
    return segs


def footprint(classes, cell: float = 0.02):
    floor, walls = classes["floor"], classes["walls"]
    angle = manhattan_angle(walls)
    q_floor = to_plan(floor.inliers, angle)
    q_walls = np.concatenate([to_plan(w.inliers, angle) for w in walls])
    q = np.concatenate([q_floor, q_walls])
    lo = q.min(0) - 0.1
    ij = ((q - lo) / cell).astype(int)
    img = np.zeros(ij.max(0)[::-1] + 3, np.uint8)
    img[ij[:, 1], ij[:, 0]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    segs = [(a, o * cell + lo[0 if a == "u" else 1]) for a, o in _rectilinear(cnt, eps=0.08 / cell)]
    segs = _drop_degenerate(segs)
    if len(segs) < 4:          # non-rectilinear blob: fall back to its axis-aligned bounding box
        mn, mx = q.min(0), q.max(0)
        segs = [("v", mn[1]), ("u", mx[0]), ("v", mx[1]), ("u", mn[0])]
    snapped = _drop_degenerate(_snap(segs, _wall_lines(walls, angle)))
    n = len(snapped)
    verts = []
    for i in range(n):
        a, b = snapped[i], snapped[(i + 1) % n]
        verts.append((a[1], b[1]) if a[0] == "u" else (b[1], a[1]))
    verts = np.array(verts)
    edges = []
    for i in range(n):
        p0, p1 = verts[i - 1], verts[i]
        axis, off, sup, rms = snapped[i]
        k_ = 1 if axis == "u" else 0
        edges.append(Edge(axis, off, float(p0[k_]), float(p1[k_]), sup, rms))
    area2 = np.sum(verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1])
    if area2 < 0:
        # reversed vertex j == original vertex n-1-j, so reversed edge j == original edge (n-j) % n, flipped
        verts = verts[::-1]
        edges = [Edge(e.axis, e.offset, e.end, e.start, e.support, e.rms)
                 for e in (edges[(n - j) % n] for j in range(n))]
    return verts, edges, angle
