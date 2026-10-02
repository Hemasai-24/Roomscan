"""Manhattan-aligned rectilinear room footprint, edges snapped to fitted wall planes."""
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
    segs = _rectilinear(cnt, eps=0.08 / cell)
    lines = _wall_lines(walls, angle)
    snapped = []
    for axis, off_px in segs:
        k_ = 0 if axis == "u" else 1
        off = off_px * cell + lo[k_]
        cands = [l for l in lines if l[0] == axis and abs(l[1] - off) < SNAP_DIST]
        if cands:
            best = max(cands, key=lambda l: l[4])
            snapped.append((axis, best[1], best[4], best[5]))
        else:
            snapped.append((axis, off, 0, cell))
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
