# Plan 2: Rooms, House Plan and Drift Correction


**Goal:** `python run.py <whole-floor capture>` outputs one plan with every room separated,
measured and connected (which room opens into which), and the phone's drift corrected — with
a script that shows the plan with drift correction off vs on.

**Approach in plain words:**
1. Fix the surface finder so a horizontal "slice" through all the walls is no longer mistaken
   for a floor (each point must face the same way as the plane it joins).
2. Look at the floor from above and cut it into rooms at the narrow places (doorways).
3. Measure each room on its own: its own floor height, its own ceiling, its own walls and doors.
4. Two rooms are connected if a door of one leads into the other.
5. Drift: walk through the recording in short chunks; nudge each chunk (tiny turn + shift) so its
   walls land exactly on the walls already seen. Report how sharp the walls are with and without.
6. Repeatability: run both whole-floor scans of the same flat and compare the same walls.

**Tech Stack:** as Plan 1 (NumPy, SciPy, Open3D, OpenCV, Shapely, pytest). No new dependencies.

**Spec:** `docs/design.md`

**Evidence this plan is built on (prototyped on the sample data, 2026-10-02):**
- Plain RANSAC on `single_scan_floor_only` found only 3 walls and ~30 fake horizontal planes at
  every height (0.15, 0.33, 0.43 m ...). With point normals required to agree with the plane
  (|n·n_plane| > 0.85) it found 32 walls and no fake slices. On `single_scan_with_ceiling`:
  9 → 28 walls, and ceilings at 2.29 / 2.44 / 2.56 / 2.93 / 3.1 m — different per room.
- Watershed on the floor-from-above with a 0.35 m core distance splits the flat into
  corridor + rooms, cut at doorways. Short furniture planes (cabinets) inside rooms must be
  excluded from the wall mask, and each room mask's holes (furniture) must be filled.
- In `single_scan_with_ceiling` several walls are visibly doubled in the top view → drift.

## Global Constraints
- Same input → identical output (seeded, no threaded RANSAC).
- World +Y is up (gravity); plan coordinates (u, v) come from `room_outline.to_plan`.
- Every measurement keeps its 95% range; per-room ceilings may be unobserved (flagged).
- Commit after every task; plain commit messages, no co-author trailers.
- Pytest command: `.venv/bin/python -m pytest -q`.

## Review Focus
1. **Open-plan room** (kitchen + living with no wall between): must stay ONE room, not be cut
   where a sofa narrows the floor. Test in Task 3 (L-shaped room stays one).
2. **Two rooms whose floors touch only through a doorway**: must become TWO rooms with an
   adjacency via that door. Test in Task 3 and Task 5.
3. **Tall wardrobe against a wall**: counts as wall (acceptable), but a 0.9 m kitchen counter
   must not split a room. Test in Task 3.
4. **Drift correction on an already-good capture** must not make it worse. Test in Task 6.
5. **Room seen only through a doorway** (tiny floor patch): dropped if under 1.0 m², not
   reported as a room. Test in Task 3.

---

### Task 1: Points must face the plane they join (fix fake horizontal planes)

**In plain words:** a horizontal plane at 0.8 m height crosses every wall in a thin band. Plain
RANSAC counts those wall points as support and "finds" a floor at 0.8 m. Each 3D point gets a
direction (its surface normal, from its neighbours); a point only supports a plane if it faces
the same way.

**Files:**
- Modify: `roomscan/point_cloud.py` (add `estimate_normals`)
- Modify: `roomscan/find_surfaces.py` (`extract_planes` takes normals; `classify_planes` also returns `"horizontal"`)
- Modify: `roomscan/pipeline.py`
- Test: `tests/test_find_surfaces.py`, `tests/test_point_cloud.py`

**Interfaces:**
- Produces: `estimate_normals(points, radius=0.08, max_nn=20) -> np.ndarray[(N,3)]` (unit, sign arbitrary)
- Produces: `extract_planes(points, normals=None, dist=0.02, min_inliers=1500, max_planes=40, iters=2000, agree=0.85)`
  — when `normals` is None, behaves as before.
- Produces: `classify_planes(...)` dict gains key `"horizontal": list[Plane]` (all horizontal planes, any kind)

- [ ] **Step 1: Failing tests**

`tests/test_point_cloud.py` (append):
```python
def test_normals_of_a_floor_point_up():
    from roomscan.point_cloud import estimate_normals
    xs, zs = np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 2, 0.02))
    p = np.c_[xs.ravel(), np.zeros(xs.size), zs.ravel()]
    n = estimate_normals(p)
    assert np.all(np.abs(n[:, 1]) > 0.99)
```
`tests/test_find_surfaces.py` (append):
```python
def test_no_fake_horizontal_slices_through_walls():
    from roomscan.point_cloud import estimate_normals
    # tall narrow room: lots of wall area, little floor -> a plain RANSAC slice at mid-height
    # through all four walls collects more points than a real small plane
    p = room_points([(0, 0), (1.2, 0), (1.2, 6), (0, 6)], 3.0, ceiling=False)
    planes = extract_planes(p, estimate_normals(p), min_inliers=300)
    horiz = [pl for pl in planes if abs(pl.normal[1]) > 0.95]
    assert all(abs(pl.height) < 0.03 for pl in horiz)        # only the floor


@pytest.mark.sample
def test_sample_floor_only_finds_many_walls():
    from tests.conftest import SAMPLE
    from roomscan.load_capture import load_stray
    from roomscan.point_cloud import fuse_points, estimate_normals
    root = SAMPLE / "single_scan_floor_only" / "1a8384c3f6"
    if not root.exists():
        pytest.skip("sample_data not present")
    pts = fuse_points(load_stray(root), stride=10)
    c = classify_planes(extract_planes(pts, estimate_normals(pts)))
    assert len(c["walls"]) >= 15
    fh = c["floor"].height
    fake = [p for p in c["horizontal"] if 0.2 < p.height - fh < 0.7 and len(p.inliers) > 15000]
    assert fake == []
```
(add `import pytest` at the top of `tests/test_find_surfaces.py`)

- [ ] **Step 2: Run, verify they fail** — `.venv/bin/python -m pytest -q tests/test_find_surfaces.py tests/test_point_cloud.py`
  Expected: FAIL (`estimate_normals` missing / `extract_planes` has no normals argument).

- [ ] **Step 3: Implement**

`roomscan/point_cloud.py` (append):
```python
def estimate_normals(points, radius: float = 0.08, max_nn: int = 20):
    """Surface direction at each point, from its neighbours (sign is arbitrary)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pc.normals)
```
`roomscan/find_surfaces.py`: replace `_ransac` and `extract_planes` with:
```python
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
```
In `classify_planes`, add `"horizontal": horiz` to the returned dict.
`roomscan/pipeline.py`: `pts = fuse_points(cap)` → also `nrm = estimate_normals(pts)` and
`extract_planes(pts, nrm)` (import `estimate_normals`).

- [ ] **Step 4: Run, verify pass** — `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "Require point normals to agree with planes; removes fake horizontal slices"`

---

### Task 2: Draw the plan the right way round (not mirrored)

**In plain words:** we drew "z" pointing up the page; seen from above, z actually points
down the page, so every drawing was a mirror image. Flip the vertical axis of the drawing.

**Files:** Modify `roomscan/draw_plan.py`; Test `tests/test_draw_plan.py`

- [ ] **Step 1: Failing test** — `tests/test_draw_plan.py`:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from roomscan.draw_plan import setup_axes


def test_plan_axes_not_mirrored():
    fig, ax = plt.subplots()
    setup_axes(ax)
    assert ax.yaxis_inverted()      # screen-up is -z when looking down from +y
    plt.close(fig)
```
- [ ] **Step 2: Run, verify fail** (`ImportError: setup_axes`).
- [ ] **Step 3: Implement** — in `draw_plan.py` add:
```python
def setup_axes(ax):
    """Top-down view: looking down from +Y, plan v (world z, rotated) points DOWN the page."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()
```
and in `render` replace `ax.set_aspect("equal")` / `ax.axis("off")` with `setup_axes(ax)`
(call it after drawing).
- [ ] **Step 4: Run, verify pass.**  **Step 5: Commit** — `"Fix mirrored plan drawing"`

---

### Task 3: Split the floor into rooms (new file `roomscan/split_rooms.py`)

**In plain words:** make a top-down picture of the floor we saw (white = floor), and paint the
walls black. For each floor pixel, compute its distance to the nearest wall. Room centres are
far from walls; doorways are narrow, so near walls. Pixels further than 0.35 m from any wall
form "room cores". Grow each core outward until cores meet (watershed); the meeting lines are
the doorways. Then: fill furniture holes, drop tiny pieces (< 1 m²), and merge two pieces if
they share a long open boundary (> 1.2 m, wider than any door) — that is one open-plan room.

Only "tall" walls (points from 0.2 m up to at least 1.4 m above the floor) block, so a kitchen
counter (0.9 m) does not cut a room in two.

**Files:**
- Create: `roomscan/split_rooms.py`
- Modify: `roomscan/room_outline.py` (add the `Grid` class shown in Task 4)
- Modify: `tests/synthetic_rooms.py` (add `two_rooms_with_door`)
- Test: `tests/test_split_rooms.py`

**Interfaces:**
- Produces (defined in `roomscan/room_outline.py` to avoid a circular import; `split_rooms` imports it):
  `Grid(lo:np.ndarray[2], cell:float, shape:tuple[int,int])` with
  `.cells(q:np.ndarray[(N,2)]) -> np.ndarray[(N,2)] int (col,row)` and `.raster(q) -> np.uint8 (H,W)`
- Produces: `is_tall_wall(plane, floor_h, low=0.2, high=1.4) -> bool`
- Produces: `split_rooms(classes, angle, cell=0.05, core=0.35, min_area=1.0, merge_len=1.2) -> tuple[list[np.ndarray(bool,H,W)], Grid]`

- [ ] **Step 1: Synthetic two-room generator** — append to `tests/synthetic_rooms.py`:
```python
def two_rooms_with_door(door=(1.0, 1.9), door_h=2.05, h=2.5, step=0.02):
    """Room A x in [0,4], room B x in [4,7]; both z in [0,3]. Shared wall x=4 has a door at z in `door`."""
    a = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], h, step=step)
    b = room_points([(4, 0), (7, 0), (7, 3), (4, 3)], h, step=step, seed=1)
    p = np.concatenate([a, b])
    hole = (np.abs(p[:, 0] - 4) < 0.03) & (p[:, 2] > door[0]) & (p[:, 2] < door[1]) & (p[:, 1] < door_h) \
        & (p[:, 1] > 0.02)
    return p[~hole]
```
- [ ] **Step 2: Failing tests** — `tests/test_split_rooms.py`:
```python
import numpy as np
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.point_cloud import estimate_normals
from roomscan.room_outline import manhattan_angle
from roomscan.split_rooms import split_rooms
from tests.synthetic_rooms import room_points, table_points, two_rooms_with_door


def _classes(p):
    return classify_planes(extract_planes(p, estimate_normals(p), min_inliers=500))


def _split(p):
    c = _classes(p)
    return split_rooms(c, manhattan_angle(c["walls"]))


def test_two_rooms_split_at_door():
    rooms, g = _split(two_rooms_with_door())
    areas = sorted(m.sum() * g.cell ** 2 for m in rooms)
    assert len(rooms) == 2
    np.testing.assert_allclose(areas, [9.0, 12.0], rtol=0.12)


def test_l_shaped_room_stays_one():
    rooms, _ = _split(room_points([(0, 0), (5, 0), (5, 2), (3, 2), (3, 4), (0, 4)], 2.5))
    assert len(rooms) == 1


def test_counter_does_not_split_room():
    p = room_points([(0, 0), (6, 0), (6, 3), (0, 3)], 2.5)
    counter = np.concatenate([table_points(2.9, 0.0, 3.1, 2.2, top=0.9),
                              np.c_[np.full(500, 2.9), np.linspace(0, 0.9, 500), np.linspace(0, 2.2, 500)]])
    rooms, _ = _split(np.concatenate([p, counter]))
    assert len(rooms) == 1
```
- [ ] **Step 3: Run, verify fail** (`ModuleNotFoundError: roomscan.split_rooms`).
- [ ] **Step 4: Implement `roomscan/split_rooms.py`:**
```python
"""Step 4a: split the floor (seen from above) into rooms, cutting at narrow places (doorways)."""
from dataclasses import dataclass

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


def split_rooms(classes, angle, cell=0.05, core=0.35, min_area=1.0, merge_len=1.2):
    fh = classes["floor"].height
    floor_q = to_plan(classes["floor"].inliers, angle)
    walls = [w for w in classes["walls"] if is_tall_wall(w, fh)]
    wall_pts = np.concatenate([w.inliers[(w.inliers[:, 1] - fh > 0.1) & (w.inliers[:, 1] - fh < 2.0)]
                               for w in walls]) if walls else np.zeros((0, 3))
    wall_q = to_plan(wall_pts, angle)
    allq = np.concatenate([floor_q, wall_q])
    lo = allq.min(0) - 0.3
    W, H = ((allq.max(0) - lo) / cell).astype(int) + 8
    g = Grid(lo, cell, (H, W))
    free = cv2.morphologyEx(g.raster(floor_q), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    wall = cv2.dilate(g.raster(wall_q), np.ones((3, 3), np.uint8))
    free[wall > 0] = 0
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * cell
    n, cores = cv2.connectedComponents((dist > core).astype(np.uint8))
    markers = cores.astype(np.int32)
    markers[free == 0] = n                       # walls + outside: their own basin
    cv2.watershed(cv2.cvtColor(free * 255, cv2.COLOR_GRAY2BGR), markers)
    rooms = [(markers == k) & (free > 0) for k in range(1, n)]
    rooms = _merge_open(rooms, merge_len, cell)
    rooms = [_fill_holes(m) for m in rooms]
    return [m for m in rooms if m.sum() * cell * cell >= min_area], g


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
```
- [ ] **Step 5: Run, verify pass** — `.venv/bin/python -m pytest -q tests/test_split_rooms.py`. If the
  L-shape splits, print the shared-boundary length and adjust `merge_len` (do not weaken the test).
- [ ] **Step 6: Commit** — `"Split floor into rooms at doorways (distance transform + watershed)"`

---

### Task 4: Measure each room on its own, and connect rooms

**In plain words:** for each room piece: its outline comes from its own floor area (snapped to
nearby walls), its floor height from the floor points inside it, its ceiling from the ceiling
plane that covers most of it (or "not observed"), its doors from its own walls. Then two rooms
are connected if a door of one sits within 0.4 m of the other room.

**Files:**
- Modify: `roomscan/room_outline.py` (add `outline_from_mask`; `footprint` reuses it)
- Create: `roomscan/room_surfaces.py` (`surfaces_for_room`)
- Create: `roomscan/connect_rooms.py` (`adjacency`)
- Modify: `roomscan/pipeline.py` (`run_lidar` multi-room; keep `run_lidar_single` name as alias)
- Modify: `roomscan/save_plan.py` (`build_plan(..., adjacency=[])`), `roomscan/draw_plan.py`
- Test: `tests/test_rooms_end_to_end.py`

**Interfaces:**
- Produces: `outline_from_mask(mask:np.ndarray(bool), grid:Grid, walls:list[Plane], angle:float) -> (verts, edges)`
  (same `verts`/`edges` contract as `footprint`)
- Produces: `surfaces_for_room(classes, mask, grid, angle) -> dict` (same keys as `classify_planes`)
- Produces: `adjacency(rooms:list[dict], masks:list[np.ndarray], grid:Grid) -> list[dict]` —
  each `{"room_a": id, "room_b": id, "via": opening_id | None}`
- Produces: `run_lidar(capture_dir, drift_correction=False) -> dict` (plan)

- [ ] **Step 1: Failing tests** — `tests/test_rooms_end_to_end.py`:
```python
import numpy as np
from shapely.geometry import Polygon
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.point_cloud import estimate_normals
from roomscan.room_outline import manhattan_angle, outline_from_mask
from roomscan.room_surfaces import surfaces_for_room
from roomscan.split_rooms import split_rooms
from tests.synthetic_rooms import two_rooms_with_door


def _setup():
    p = two_rooms_with_door()
    c = classify_planes(extract_planes(p, estimate_normals(p), min_inliers=500))
    a = manhattan_angle(c["walls"])
    masks, g = split_rooms(c, a)
    return c, a, masks, g


def test_each_room_outline_matches_its_walls():
    c, a, masks, g = _setup()
    areas = []
    for m in masks:
        verts, edges = outline_from_mask(m, g, c["walls"], a)
        areas.append(Polygon(verts).area)
        assert len(edges) == 4
    np.testing.assert_allclose(sorted(areas), [9.0, 12.0], rtol=0.03)


def test_rooms_do_not_overlap():
    c, a, masks, g = _setup()
    polys = [Polygon(outline_from_mask(m, g, c["walls"], a)[0]) for m in masks]
    assert polys[0].intersection(polys[1]).area < 0.05


def test_per_room_ceiling_found():
    c, a, masks, g = _setup()
    for m in masks:
        s = surfaces_for_room(c, m, g, a)
        assert s["ceiling"] is not None and abs(s["ceiling"].height - 2.5) < 0.02
```
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**

`roomscan/room_outline.py` — add `Grid` (moved here from the Task 3 sketch; Task 3 adds it here):
```python
@dataclass
class Grid:
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
```
then split `footprint` so the raster/contour part is reusable:
```python
def outline_from_mask(mask, grid, walls, angle):
    """Rectilinear outline of one room's floor mask, edges snapped to nearby fitted walls."""
    img = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    segs = [(ax, o * grid.cell + grid.lo[0 if ax == "u" else 1] + grid.cell / 2)
            for ax, o in _rectilinear(cnt, eps=0.08 / grid.cell)]
    segs = _drop_degenerate(segs)
    if len(segs) < 4:
        ys, xs = np.nonzero(mask)
        u0, v0 = grid.lo + np.array([xs.min(), ys.min()]) * grid.cell
        u1, v1 = grid.lo + np.array([xs.max() + 1, ys.max() + 1]) * grid.cell
        segs = [("v", v0), ("u", u1), ("v", v1), ("u", u0)]
    snapped = _drop_degenerate(_snap(segs, _wall_lines(walls, angle), snap_dist=MASK_SNAP_DIST))
    return _assemble(snapped)
```
with `MASK_SNAP_DIST = 0.25` (the mask edge stops ~10-20 cm short of the wall face because
walls are dilated out of the floor), `_snap(segs, lines, snap_dist=SNAP_DIST)` taking the
distance as a parameter, and `_assemble(snapped) -> (verts, edges)` = the existing vertex/edge
construction + counter-clockwise fix moved out of `footprint` unchanged. `footprint(classes)`
becomes: build its raster as now, then `return (*outline_from_mask(raster > 0, Grid(lo, cell,
img.shape), walls, angle), angle)` — its existing tests must still pass.

`roomscan/room_surfaces.py`:
```python
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
```

`roomscan/connect_rooms.py`:
```python
"""Step 4c: which rooms connect to which (through a door), for the whole-property plan."""
import numpy as np

NEAR = 0.4   # a door this close to another room leads into it


def _door_mid(room, op):
    w = next(x for x in room["walls"] if x["id"] == op["wall_id"])
    a, b = np.array(w["start"]), np.array(w["end"])
    u = (b - a) / np.linalg.norm(b - a)
    return a + u * (op["offset_along_wall"] + op["width"]["value"] / 2)


def _dist_to_mask(pt, mask, grid):
    ys, xs = np.nonzero(mask)
    cells = grid.lo + (np.c_[xs, ys] + 0.5) * grid.cell
    return float(np.min(np.linalg.norm(cells - pt, axis=1)))


def adjacency(rooms, masks, grid):
    out = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            via = None
            for r, other in ((i, j), (j, i)):
                for op in rooms[r]["openings"]:
                    if op["type"] == "door" and _dist_to_mask(_door_mid(rooms[r], op), masks[other], grid) < NEAR:
                        via = op["id"]
                        break
                if via:
                    break
            if via:
                out.append({"room_a": rooms[i]["id"], "room_b": rooms[j]["id"], "via": via})
    return out
```

`roomscan/pipeline.py` — new `run_lidar` (keep `run_lidar_single = run_lidar`):
```python
def run_lidar(capture_dir, drift_correction=False):
    t0 = time.time()
    cap = load_stray(capture_dir)
    drift = None
    if drift_correction:
        cap, drift = correct_drift(cap)          # Task 5; import inside the if until then
    pts = fuse_points(cap)
    classes = classify_planes(extract_planes(pts, estimate_normals(pts)))
    angle = manhattan_angle(classes["walls"])
    masks, grid = split_rooms(classes, angle)
    rays = list(capture_rays(cap))
    rooms = []
    for k, m in enumerate(masks):
        s = surfaces_for_room(classes, m, grid, angle)
        verts, edges = outline_from_mask(m, grid, s["walls"], angle)
        wall_h = (s["ceiling"].height if s["ceiling"] else s["floor"].height + 2.6) - s["floor"].height
        ops = find_openings(rays, edges, angle, s["floor"].height, wall_h)
        rooms.append(measure_room(s, verts, edges, angle, ops, tier="lidar", room_id=f"room_{k}"))
    meta = {"n_frames": len(cap.frames), "n_points": int(len(pts)), "n_rooms": len(rooms),
            "manhattan_angle_rad": round(angle, 6), "drift_correction": drift_correction,
            "runtime_s": round(time.time() - t0, 1)}
    if drift:
        meta["drift"] = drift
    return build_plan(Path(capture_dir).name, "lidar", rooms, meta, adjacency(rooms, masks, grid))
```
`save_plan.build_plan(capture_id, tier, rooms, meta, adjacency=())` stores `list(adjacency)`.
`draw_plan.render`: label each room with its `id`; draw a dashed line between connected room
centroids (light grey) so adjacency is visible.

- [ ] **Step 4: Run, verify pass**; then run `run.py` on `single_room` and both whole-floor scans; look at
  each `plan.png`. Expected: separate rooms, no overlaps, doors on shared walls. Record room counts and
  runtimes in the commit message.
- [ ] **Step 5: Commit** — `"Measure each room separately and connect rooms through doors"`

---

### Task 5: Drift correction (new file `roomscan/drift_correction.py`)

**In plain words:** cut the recording into chunks of 300 frames (~5 s). The first chunk is the
reference. For each next chunk: (1) turn it slightly so its walls point exactly along the
house's two main directions; (2) shift it so its wall points land on the walls already placed
(match each wall point to the nearest already-placed wall point facing the same way, take the
median gap, shift by it; repeat 3 times); same for floor height. Add the corrected chunk to the
placed walls. Corrections are blended smoothly between chunk centres so there are no jumps.

**Files:** Create `roomscan/drift_correction.py`; Test `tests/test_drift_correction.py`;
Modify `tests/synthetic_rooms.py` (add `render_box_depth`).

**Interfaces:**
- Produces: `correct_drift(cap, chunk=300, stride=5) -> tuple[StrayCapture, dict]` — corrected copy
  (frames' `T_wc` changed) and report `{"chunks": n, "max_yaw_deg": .., "max_shift_m": ..}`
- Produces: `wall_sharpness(points, normals, walls) -> float` — median over walls of the robust
  spread (1.4826·MAD, metres) of points within ±0.10 m of each wall. Doubled walls → large.

- [ ] **Step 1: Synthetic depth renderer** — append to `tests/synthetic_rooms.py`:
```python
def render_box_depth(K, T_wc, lo, hi, w, h):
    """Depth (metres, along camera z) seen from inside an axis-aligned box room lo..hi."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_w = d_cam @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    t = np.full(u.shape, np.inf)
    for ax in range(3):
        for b in (lo[ax], hi[ax]):
            with np.errstate(divide="ignore", invalid="ignore"):
                ti = (b - o[ax]) / d_w[..., ax]
            t = np.where((ti > 1e-6) & (ti < t), ti, t)
    return t      # d_cam has z = 1, so ray parameter t equals depth
```
- [ ] **Step 2: Failing tests** — `tests/test_drift_correction.py`:
```python
import numpy as np
from scipy.spatial.transform import Rotation
from roomscan.drift_correction import correct_drift, wall_sharpness
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.load_capture import load_stray
from roomscan.point_cloud import estimate_normals, fuse_points
from tests.synthetic_rooms import render_box_depth

LO, HI = np.array([0.0, -1.4, 0.0]), np.array([5.0, 1.2, 4.0])


def _poses(n):
    """Camera walks a loop inside the box, turning to look at all walls."""
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = np.array([2.5 + 1.2 * np.cos(a), 0.0, 2.0 + 1.0 * np.sin(a)])
        R = Rotation.from_euler("y", np.degrees(a) * 2, degrees=True).as_matrix()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
        out.append(T)
    return out


def _capture(stray_factory, drift):
    n = 600
    true = _poses(n)
    stored = []
    for i, T in enumerate(true):
        f = i / n                                    # drift grows along the walk
        D = np.eye(4)
        D[:3, :3] = Rotation.from_euler("y", drift[0] * f, degrees=True).as_matrix()
        D[:3, 3] = [drift[1] * f, 0, drift[2] * f]
        stored.append(D @ T)
    K = np.array([[1600 * 64 / 1920, 0, 32], [0, 1600 * 64 / 1920, 24], [0, 0, 1]])
    def depth(i):
        return render_box_depth(K, true[i], LO, HI, 64, 48)
    poses = [(S[:3, 3], Rotation.from_matrix(S[:3, :3]).as_quat()) for S in stored]
    return load_stray(stray_factory(depth, poses, w=64, h=48)), true


def _sharp(cap):
    pts = fuse_points(cap, stride=3, max_depth=8)
    nrm = estimate_normals(pts)
    walls = classify_planes(extract_planes(pts, nrm, min_inliers=200))["walls"]
    return wall_sharpness(pts, nrm, walls)


def test_drift_correction_sharpens_doubled_walls(stray_factory):
    cap, _ = _capture(stray_factory, drift=(3.0, 0.08, -0.05))
    before = _sharp(cap)
    fixed, report = correct_drift(cap, chunk=100, stride=3)
    after = _sharp(fixed)
    assert before > 0.01
    assert after < before * 0.5


def test_drift_correction_leaves_clean_capture_alone(stray_factory):
    cap, _ = _capture(stray_factory, drift=(0.0, 0.0, 0.0))
    fixed, report = correct_drift(cap, chunk=100, stride=3)
    assert report["max_shift_m"] < 0.01 and report["max_yaw_deg"] < 0.2
```
- [ ] **Step 3: Run, verify fail** (`ModuleNotFoundError`).
- [ ] **Step 4: Implement `roomscan/drift_correction.py`:**
```python
"""Drift correction: re-align the walk chunk by chunk so walls seen again land on walls seen before.

Each chunk gets a small turn about the vertical axis (to the house's main wall directions) and a
shift (median gap between its wall/floor points and the already-placed ones). Corrections are
blended linearly between chunk centres. 'Poses used as-is' is an automatic fail in the brief."""
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from roomscan.point_cloud import backproject, estimate_normals
from roomscan.room_outline import manhattan_angle

MATCH_DIST = 0.20
ITERS = 3


def _chunk_points(cap, frames, stride, max_depth=4.0):
    pts = [backproject(cap.load_depth(f.index), f.K, f.T_wc, max_depth) for f in frames[::stride]]
    p = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(p) > 60000:
        p = p[np.random.default_rng(0).choice(len(p), 60000, replace=False)]
    return p


def _yaw_of(points, normals):
    vert = np.abs(normals[:, 1]) < 0.2
    if vert.sum() < 100:
        return None
    ang = np.arctan2(normals[vert, 2], normals[vert, 0])
    return float(np.angle(np.sum(np.exp(4j * ang))) / 4)


def _rigid(yaw, shift, center):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center + shift
    return T


def _apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def correct_drift(cap, chunk=300, stride=5):
    frames = cap.frames
    chunks = [frames[i:i + chunk] for i in range(0, len(frames), chunk)]
    ref = None            # placed points + normals so far
    ref_angle = None
    params = []           # (yaw, shift, center) per chunk
    prev = (0.0, np.zeros(3))
    for ch in chunks:
        p = _chunk_points(cap, ch, stride)
        if len(p) < 500:
            params.append((prev[0], prev[1], np.mean([f.T_wc[:3, 3] for f in ch], 0)))
            continue
        center = np.mean([f.T_wc[:3, 3] for f in ch], 0)
        yaw, shift = prev
        n = estimate_normals(p)
        if ref is None:
            ref_angle = _yaw_of(p, n) or 0.0
            yaw, shift = 0.0, np.zeros(3)
        else:
            a = _yaw_of(p, n)
            if a is not None:
                d = (ref_angle - a + np.pi / 4) % (np.pi / 2) - np.pi / 4
                yaw = -d
            tree = cKDTree(ref[0])
            for _ in range(ITERS):
                q = _apply(_rigid(yaw, shift, center), p)
                qn = n @ _rigid(yaw, shift, center)[:3, :3].T
                dist, idx = tree.query(q, distance_upper_bound=MATCH_DIST)
                ok = np.isfinite(dist)
                ok[ok] &= np.abs(np.einsum("ij,ij->i", qn[ok], ref[1][idx[ok]])) > 0.9
                if ok.sum() < 200:
                    break
                rn = ref[1][idx[ok]]
                gap = np.einsum("ij,ij->i", ref[0][idx[ok]] - q[ok], rn)   # along the ref normal
                for ax in range(3):
                    sel = np.abs(rn[:, ax]) > 0.9
                    if sel.sum() > 50:
                        shift[ax] += float(np.median(gap[sel] * np.sign(rn[sel, ax])))
        T = _rigid(yaw, shift, center)
        placed = (_apply(T, p), n @ T[:3, :3].T)
        ref = placed if ref is None else (np.concatenate([ref[0], placed[0]]), np.concatenate([ref[1], placed[1]]))
        if len(ref[0]) > 400000:
            keep = np.random.default_rng(0).choice(len(ref[0]), 400000, replace=False)
            ref = (ref[0][keep], ref[1][keep])
        params.append((yaw, shift.copy(), center))
        prev = (yaw, shift.copy())
    # blend per frame between chunk centres
    centres = np.array([(i + 0.5) * chunk for i in range(len(params))])
    yaws = np.array([q[0] for q in params])
    shifts = np.array([q[1] for q in params])
    cents = np.array([q[2] for q in params])
    out = deepcopy(cap)
    for k, f in enumerate(out.frames):
        y = np.interp(k, centres, yaws)
        s = np.array([np.interp(k, centres, shifts[:, a]) for a in range(3)])
        c = np.array([np.interp(k, centres, cents[:, a]) for a in range(3)])
        f.T_wc = _rigid(y, s, c) @ f.T_wc
    report = {"chunks": len(params), "max_yaw_deg": round(float(np.degrees(np.max(np.abs(yaws)))), 3),
              "max_shift_m": round(float(np.max(np.linalg.norm(shifts, axis=1))), 4)}
    return out, report


def wall_sharpness(points, normals, walls):
    """Median over walls of the robust thickness (1.4826*MAD, m) of points within 10 cm of the wall."""
    vals = []
    vert = np.abs(normals[:, 1]) < 0.3
    for w in walls:
        d = points[vert] @ w.normal + w.d
        lo, hi = w.inliers.min(0), w.inliers.max(0)
        inb = np.all((points[vert] >= lo - 0.1) & (points[vert] <= hi + 0.1), axis=1)
        dd = d[(np.abs(d) < 0.10) & inb]
        if len(dd) > 100:
            vals.append(1.4826 * np.median(np.abs(dd - np.median(dd))))
    return float(np.median(vals)) if vals else float("nan")
```
- [ ] **Step 5: Run, verify pass.** If the "sharpens" test fails, print per-chunk `(yaw, shift)` against
  the injected drift before changing anything (systematic debugging; do not weaken thresholds).
- [ ] **Step 6: Wire into `run.py`:** add `--drift-correction {on,off}` (default `on`); pass to `run_lidar`.
- [ ] **Step 7: Commit** — `"Add chunk-wise plane-anchored drift correction"`

---

### Task 6: Drift on/off comparison and repeatability scripts

**In plain words:** two small scripts that produce the numbers and pictures the brief asks for.
`scripts/drift_ablation.py <capture>`: runs with correction off and on, saves both plans side by
side and a table (wall sharpness in mm, total floor area, room count). `scripts/repeatability.py
<capture A> <capture B>`: lines up the two plans (try the 4 right-angle turns, then shift to
best overlap), pairs up walls by position, and reports each pair's length difference against
the gate (≤ 1 cm or ≤ 0.5%).

**Files:** Create `scripts/drift_ablation.py`, `scripts/repeatability.py`, `roomscan/compare_plans.py`;
Test `tests/test_compare_plans.py`.

**Interfaces:**
- Produces: `align_plans(plan_a, plan_b) -> (R2:np.ndarray[2,2], t:np.ndarray[2])` mapping B's plan coords onto A's
- Produces: `match_walls(plan_a, plan_b, R2, t, max_mid=0.3) -> list[dict]` each
  `{"a": wall_id, "b": wall_id, "len_a", "len_b", "diff_m", "pass": bool}`

- [ ] **Step 1: Failing test** — `tests/test_compare_plans.py`:
```python
import numpy as np
from roomscan.compare_plans import align_plans, match_walls


def _plan(poly, rot_deg=0.0, shift=(0, 0), scale=1.0):
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    P = (np.array(poly, float) * scale) @ R.T + shift
    walls = [{"id": f"w{i}", "start": list(P[i - 1]), "end": list(P[i]),
              "length": {"value": float(np.linalg.norm(P[i] - P[i - 1]))}} for i in range(len(P))]
    return {"rooms": [{"id": "r0", "polygon": P.tolist(), "walls": walls}]}


def test_same_room_rotated_and_shifted_matches_exactly():
    poly = [(0, 0), (4.2, 0), (4.2, 3.1), (2.0, 3.1), (2.0, 4.0), (0, 4.0)]
    a, b = _plan(poly), _plan(poly, rot_deg=90, shift=(10, -3))
    R2, t = align_plans(a, b)
    m = match_walls(a, b, R2, t)
    assert len(m) == 6 and all(x["pass"] for x in m)


def test_length_difference_detected():
    poly = [(0, 0), (4.0, 0), (4.0, 3.0), (0, 3.0)]
    a, b = _plan(poly), _plan(poly, scale=1.01)          # 1% longer walls: fails 0.5% / 1 cm gate
    R2, t = align_plans(a, b)
    assert not all(x["pass"] for x in match_walls(a, b, R2, t))
```
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `roomscan/compare_plans.py`:**
```python
"""Compare two plans of the same place (repeatability): align them, pair walls, diff lengths."""
import numpy as np


def _walls(plan):
    for r in plan["rooms"]:
        for w in r["walls"]:
            yield r["id"], w


def _mid(w, R2=np.eye(2), t=np.zeros(2)):
    return ((np.array(w["start"]) + np.array(w["end"])) / 2) @ R2.T + t


def align_plans(plan_a, plan_b):
    """Try the 4 right-angle turns; for each, shift centroids together, refine by nearest wall midpoints."""
    ma = np.array([_mid(w) for _, w in _walls(plan_a)])
    mb = np.array([_mid(w) for _, w in _walls(plan_b)])
    best = None
    for k in range(4):
        a = k * np.pi / 2
        R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).round(12)
        t = ma.mean(0) - (mb @ R2.T).mean(0)
        for _ in range(5):
            mb2 = mb @ R2.T + t
            d = np.linalg.norm(ma[:, None] - mb2[None], axis=2)
            j = d.argmin(1)
            ok = d[np.arange(len(ma)), j] < 0.5
            if ok.sum() == 0:
                break
            t = t + np.median(ma[ok] - mb2[j[ok]], axis=0)
        cost = np.median(np.linalg.norm(ma[:, None] - (mb @ R2.T + t)[None], axis=2).min(1))
        if best is None or cost < best[0]:
            best = (cost, R2, t)
    return best[1], best[2]


def match_walls(plan_a, plan_b, R2, t, max_mid=0.3):
    bs = list(_walls(plan_b))
    out = []
    for _, wa in _walls(plan_a):
        da = np.array(wa["end"]) - np.array(wa["start"])
        best = None
        for _, wb in bs:
            db = (np.array(wb["end"]) - np.array(wb["start"])) @ R2.T
            parallel = abs(abs(da @ db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9) - 1) < 0.01
            dm = np.linalg.norm(_mid(wa) - _mid(wb, R2, t))
            if parallel and dm < max_mid and (best is None or dm < best[0]):
                best = (dm, wb)
        if best:
            la, lb = wa["length"]["value"], best[1]["length"]["value"]
            diff = abs(la - lb)
            out.append({"a": wa["id"], "b": best[1]["id"], "len_a": la, "len_b": lb, "diff_m": round(diff, 4),
                        "pass": bool(diff <= 0.01 or diff <= 0.005 * max(la, lb))})
    return out
```
- [ ] **Step 4: Write the two scripts.**

`scripts/drift_ablation.py`:
```python
"""Drift correction on vs off on one capture: plan pictures + table (brief: drift accountability)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.drift_correction import correct_drift, wall_sharpness   # noqa: E402
from roomscan.draw_plan import render                                  # noqa: E402
from roomscan.find_surfaces import classify_planes, extract_planes     # noqa: E402
from roomscan.load_capture import load_stray                           # noqa: E402
from roomscan.pipeline import run_lidar                                # noqa: E402
from roomscan.point_cloud import estimate_normals, fuse_points         # noqa: E402


def sharpness(cap):
    pts = fuse_points(cap)
    nrm = estimate_normals(pts)
    return wall_sharpness(pts, nrm, classify_planes(extract_planes(pts, nrm))["walls"])


def main():
    cap_dir = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"outputs/ablation/{cap_dir.name}")
    rows = {}
    for mode in ("off", "on"):
        plan = run_lidar(cap_dir, drift_correction=(mode == "on"))
        (out / mode).mkdir(parents=True, exist_ok=True)
        (out / mode / "plan.json").write_text(json.dumps(plan, indent=2))
        render(plan, out / mode / "plan")
        cap = load_stray(cap_dir)
        if mode == "on":
            cap, _ = correct_drift(cap)
        rows[mode] = {"wall_sharpness_mm": round(sharpness(cap) * 1000, 2),
                      "total_floor_area_m2": round(sum(r["floor_area"]["value"] for r in plan["rooms"]), 2),
                      "rooms": len(plan["rooms"]), "drift": plan["meta"].get("drift")}
    (out / "ablation.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```
`scripts/repeatability.py`:
```python
"""Two captures of the same place, same tier: does the same room give the same walls?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.compare_plans import align_plans, match_walls   # noqa: E402


def main():
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    R2, t = align_plans(a, b)
    rows = match_walls(a, b, R2, t)
    ok = sum(r["pass"] for r in rows)
    print(f"matched walls: {len(rows)}, within gate (<=1 cm or <=0.5%): {ok}")
    for r in sorted(rows, key=lambda r: -r["diff_m"])[:15]:
        print(f"  {r['a']:>12} vs {r['b']:>12}: {r['len_a']:.3f} vs {r['len_b']:.3f} m, diff {r['diff_m']*100:.1f} cm"
              f" {'PASS' if r['pass'] else 'FAIL'}")
    Path(sys.argv[3] if len(sys.argv) > 3 else "outputs/repeatability.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```
- [ ] **Step 5: Run tests; then run both scripts on the sample data:**
  `.venv/bin/python scripts/drift_ablation.py sample_data/single_scan_with_ceiling/c7d28f72c6`
  `.venv/bin/python run.py sample_data/single_scan_floor_only/1a8384c3f6 --out outputs/rep_a`
  `.venv/bin/python run.py sample_data/single_scan_with_ceiling/c7d28f72c6 --out outputs/rep_b`
  `.venv/bin/python scripts/repeatability.py outputs/rep_a/plan.json outputs/rep_b/plan.json`
  Record the real numbers (sharpness off/on, walls passing) in the commit message and in
  `docs/TRADEOFFS.md` if a gate is missed.
- [ ] **Step 6: Commit** — `"Add drift on/off comparison and repeatability scripts"`

---

### Task 7: Interview notes for this plan

**Files:** Create `docs/INTERVIEW_NOTES.md` (one short section per pipeline step: what it does,
why this method, what it gets wrong, what we would do with more time). Include Plan 1 steps and
the decisions above (normals, watershed, merge rule, drift method, repeatability matching).
Commit — `"Add interview notes"`.
