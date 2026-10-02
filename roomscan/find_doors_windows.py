"""Step 5: doors and windows = parts of a wall that depth rays passed THROUGH.
A wall area the camera never looked at has no such rays, so it is never reported as an opening."""
from dataclasses import dataclass

import cv2
import numpy as np

from roomscan.point_cloud import backproject
from roomscan.room_outline import to_plan

BEYOND = 0.15          # ray must end this far past the wall plane
MIN_WIDTH = 0.5
MIN_HEIGHT = 0.4
DOOR_BOTTOM = 0.15     # opening reaching within 15 cm of floor is a door
MIN_HITS = 3           # rays per cell to count as free


@dataclass
class Opening:
    edge_index: int
    kind: str
    s0: float
    s1: float
    bottom: float
    top: float
    evidence: int

    @property
    def width(self):
        return self.s1 - self.s0

    @property
    def height(self):
        return self.top - self.bottom


def capture_rays(cap, stride=5, max_depth=6.0, pixel_step=2):
    for f in cap.frames[::stride]:
        d = cap.load_depth(f.index)[::pixel_step, ::pixel_step]
        K = f.K.copy()
        K[:2] /= pixel_step
        yield f.T_wc[:3, 3], backproject(d, K, f.T_wc, max_depth)


def _crossings(center, ends, edge, angle):
    c2 = to_plan(center[None], angle)[0]
    e2 = to_plan(ends, angle)
    k = 0 if edge.axis == "u" else 1
    a, b = c2[k] - edge.offset, e2[:, k] - edge.offset
    sign = np.sign(a)
    m = (sign != 0) & (np.sign(b) == -sign) & (np.abs(b) > BEYOND)
    t = a / (a - b[m])
    along = c2[1 - k] + t * (e2[m, 1 - k] - c2[1 - k])
    y = center[1] + t * (ends[m, 1] - center[1])
    return along, y


def find_openings(rays, edges, angle, floor_h, wall_h, cell=0.02):
    ncols = [int(abs(e.end - e.start) / cell) + 1 for e in edges]
    nrows = int(wall_h / cell) + 1
    grids = [np.zeros((nrows, n), np.int32) for n in ncols]
    for center, ends in rays:
        for i, e in enumerate(edges):
            along, y = _crossings(center, ends, e, angle)
            lo, hi = min(e.start, e.end), max(e.start, e.end)
            s = (along - lo) if e.end >= e.start else (hi - along)
            h = y - floor_h
            ok = (s >= 0) & (s <= hi - lo) & (h >= 0) & (h < wall_h)
            np.add.at(grids[i], ((h[ok] / cell).astype(int), (s[ok] / cell).astype(int)), 1)
    out = []
    for i, g in enumerate(grids):
        L = abs(edges[i].end - edges[i].start)
        if edges[i].support == 0:      # no fitted wall: scan boundary, not a wall with a hole
            continue
        free = (g >= MIN_HITS).astype(np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        for j in range(1, n):
            x, yv, w, hgt, _ = stats[j]
            width, height = w * cell, hgt * cell
            if width < MIN_WIDTH or height < MIN_HEIGHT or height > wall_h - 0.05 or width > L - 0.05:
                continue
            bottom = yv * cell
            kind = "door" if bottom < DOOR_BOTTOM else "window"
            out.append(Opening(i, kind, x * cell, min((x + w) * cell, L), bottom, bottom + height,
                               int(g[lab == j].sum())))
    return out
