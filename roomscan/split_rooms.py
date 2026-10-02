"""Step 4a: split the floor (seen from above) into rooms, cutting at narrow places (doorways).

Floor pixels far from any wall (> `core`) are room centres; grow them until they meet
(watershed): the meeting lines are the doorways. Only tall walls block, so a kitchen counter
does not cut a room. Pieces sharing a long open boundary are one open-plan room."""
import cv2
import numpy as np

from roomscan.room_outline import Grid, to_plan


def is_tall_wall(plane, floor_h, low=0.2, high=1.4):
    h = plane.inliers[:, 1] - floor_h
    return np.percentile(h, 2) < low and np.percentile(h, 98) > high


def _fill_holes(m):
    inv = (~m).astype(np.uint8)
    pad = np.pad(inv, 1, constant_values=1)
    cv2.floodFill(pad, None, (0, 0), 2)
    return m | (pad[1:-1, 1:-1] == 1)


def split_rooms(classes, angle, cell=0.05, core=0.45, min_area=1.0, merge_len=1.2, extra_free=None, extra_radius=0.5):
    """extra_free: optional (N,2) plan points known to be walkable (video tier: the camera path), each
    widened to a disc of extra_radius; walls still cut it. LiDAR leaves it None."""
    fh = classes["floor"].height
    floor_q = to_plan(classes["floor"].inliers, angle)
    walls = [w for w in classes["walls"] if is_tall_wall(w, fh)]
    wall_pts = np.concatenate([w.inliers[(w.inliers[:, 1] - fh > 0.1) & (w.inliers[:, 1] - fh < 2.0)]
                               for w in walls]) if walls else np.zeros((0, 3))
    wall_q = to_plan(wall_pts, angle)
    extra = np.zeros((0, 2)) if extra_free is None else np.asarray(extra_free, float)
    allq = np.concatenate([floor_q, wall_q, extra])
    lo = allq.min(0) - 0.3
    W, H = ((allq.max(0) - lo) / cell).astype(int) + 8
    g = Grid(lo, cell, (H, W))
    free = cv2.morphologyEx(g.raster(floor_q), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if len(extra):
        r = int(round(extra_radius / cell))
        free |= cv2.dilate(g.raster(extra), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    wall = cv2.dilate(g.raster(wall_q), np.ones((3, 3), np.uint8))
    free[wall > 0] = 0
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * cell
    n, cores = cv2.connectedComponents((dist > core).astype(np.uint8))
    markers = cores.astype(np.int32)
    markers[free == 0] = n                       # walls + outside: their own basin
    cv2.watershed(cv2.cvtColor(free * 255, cv2.COLOR_GRAY2BGR), markers)
    rooms = [(markers == k) & (free > 0) for k in range(1, n)]
    rooms = _merge_open(rooms, merge_len, cell)
    rooms = _grow_to_walls([_fill_holes(m) for m in rooms], grow=2)
    return [m for m in rooms if m.sum() * cell * cell >= min_area], g


def _grow_to_walls(rooms, grow):
    """Give back the wall band removed from the floor (walls were dilated), never into another room."""
    k = np.ones((2 * grow + 1,) * 2, np.uint8)
    taken = np.zeros_like(rooms[0]) if rooms else None
    for m in rooms:
        taken |= m
    out = []
    for m in rooms:
        g = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~(taken & ~m)
        taken |= g
        out.append(g)
    return out


def _merge_open(rooms, merge_len, cell):
    k = np.ones((5, 5), np.uint8)
    merged = True
    while merged:
        merged = False
        for i in range(len(rooms)):
            grown = cv2.dilate(rooms[i].astype(np.uint8), k) > 0
            for j in range(i + 1, len(rooms)):
                if (grown & rooms[j]).sum() * cell / 2 > merge_len:   # boundary band is ~2 cells thick
                    rooms[i] = rooms[i] | rooms[j]
                    del rooms[j]
                    merged = True
                    break
            if merged:
                break
    return rooms
