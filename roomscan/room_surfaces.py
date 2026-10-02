"""Step 4b: one room's own floor height, ceiling and walls."""
import cv2
import numpy as np

from roomscan.find_surfaces import MIN_CEILING_ABOVE_FLOOR, Plane, _refit
from roomscan.room_outline import to_plan

MIN_CEILING_COVER = 0.3     # a ceiling plane must cover this share of the room
WALL_NEAR = 0.3             # walls within this distance of the room belong to it


def _inside(mask, grid, pts, angle, grow_cells=0):
    m = mask.astype(np.uint8)
    if grow_cells:
        m = cv2.dilate(m, np.ones((2 * grow_cells + 1,) * 2, np.uint8))
    ij = grid.cells(to_plan(pts, angle))
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < grid.shape[1]) & (ij[:, 1] < grid.shape[0])
    out = np.zeros(len(pts), bool)
    out[ok] = m[ij[ok, 1], ij[ok, 0]] > 0
    return out


def surfaces_for_room(classes, mask, grid, angle):
    fl = classes["floor"].inliers
    fin = fl[_inside(mask, grid, fl, angle)]
    if len(fin) < 50:
        fin = fl
    n, d, rms = _refit(fin)
    floor = Plane(n, d, fin, rms, "floor")
    spread = max(classes.get("floor_spread", 0.0), float(np.std(fin[:, 1])))
    cells = mask.sum()
    best, best_cover = None, 0.0
    for p in classes.get("horizontal", []):
        if p.height < floor.height + MIN_CEILING_ABOVE_FLOOR:
            continue
        inside = p.inliers[_inside(mask, grid, p.inliers, angle)]
        if len(inside) == 0:
            continue
        cover = grid.raster(to_plan(inside, angle))[mask].sum() / cells
        if cover > best_cover:
            best, best_cover = p, cover
    ceiling = None
    if best is not None and best_cover >= MIN_CEILING_COVER:
        inside = best.inliers[_inside(mask, grid, best.inliers, angle)]
        cn, cd, crms = _refit(inside)
        ceiling = Plane(cn, cd, inside, crms, "ceiling")
    grow = int(round(WALL_NEAR / grid.cell))
    walls = [w for w in classes["walls"] if _inside(mask, grid, w.inliers, angle, grow).mean() > 0.2]
    return {"floor": floor, "ceiling": ceiling, "walls": walls or classes["walls"],
            "other": [], "horizontal": [], "floor_spread": spread}
