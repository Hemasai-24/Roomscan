# Plan 1: LiDAR Single-Room Pipeline Implementation Plan

> Note (after implementation): the package `fpp/` was renamed to `roomscan/` with clearer file names
> (see README "Code map"). Code below shows the original names.


**Goal:** `python run.py sample_data/single_room/c00a170fe1` produces `plan.json` (schema-valid) and `plan.svg/png` with walls, floor area, ceiling height and openings, each with a 95% interval.

**Architecture:** Stray Scanner capture → back-projected, voxel-downsampled world point cloud → deterministic RANSAC planes classified floor/ceiling/wall → Manhattan-aligned rectilinear footprint refined to wall-plane offsets → openings by free-space ray carving through each wall → per-measurement intervals → JSON + render.

**Tech Stack:** Python 3.10, NumPy, SciPy, Open3D 0.19, OpenCV (<5), Shapely 2, jsonschema, matplotlib, pytest.

**Spec:** `docs/design.md`

**Later plans (not in scope here):** Plan 2 multi-room segmentation + stitch + drift ablation; Plan 3 video tier; Plan 4 photo tier; Plan 5 damage + rules + scope; Plan 6 benchmark harness, fix loop, docs.

## Global Constraints

- Stray Scanner format (verified on sample data 2026-10-02): `odometry.csv` header `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...` (values space-padded); pose is **camera→world applied to OpenCV-convention camera points** (x right, y down, z forward); **world +Y is up**. Verified: floor plane normal (−0.01, 1.00, 0.01), walls vertical and orthogonal. Do NOT apply an ARKit axis flip.
- Intrinsics in odometry are at RGB resolution (1920×1440); depth is 256×192 uint16 millimetres; scale intrinsics by `depth_w / rgb_w` (=0.1333).
- Confidence PNG values 0/1/2; default keep only 2.
- Everything deterministic: seed Open3D RANSAC (`o3d.utility.random.seed(0)`) and NumPy (`np.random.default_rng(0)`). Same input → byte-identical JSON.
- Units: metres in JSON, every measurement is `{"value", "lo", "hi", "unit"}` (95% interval).
- One command per capture: `python run.py <capture_dir> [--out DIR]`.
- No network calls at run time.
- Commit after every task with a descriptive message ending in the Co-Authored-By line.

## Review Focus

1. **Glass shower / mirror** (single_room contains a glass shower door): LiDAR returns through glass create free-space evidence → a phantom opening. Expected: openings narrower than 0.5 m or taller-than-wall are rejected; remaining glass cases are flagged in `warnings`, not silently emitted. Test in Task 6.
2. **Unobserved wall section** (camera never looked at part of a wall): must NOT become an opening. Expected: no free-space evidence → no opening. Test in Task 6.
3. **Ceiling not observed** (`single_scan_floor_only`): expected `ceiling_observed: false`, interval widened to at least ±0.15 m, a warning emitted — never a confident number. Test in Task 4/7.
4. **Furniture tops** (tables, counters ~0.9 m) as horizontal planes: must classify as `other`, not floor/ceiling. Test in Task 4.
5. **Non-rectangular room** (L-shape): footprint must keep the notch, not collapse to a bounding box. Test in Task 5.

---

### Task 1: Project scaffolding + Stray Scanner ingest

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `fpp/__init__.py`, `fpp/ingest/__init__.py`, `fpp/ingest/stray.py`
- Test: `tests/conftest.py`, `tests/test_stray.py`

**Interfaces:**
- Produces:
  - `Frame(index:int, timestamp:float, T_wc:np.ndarray[4,4], K:np.ndarray[3,3])` — `K` already scaled to depth resolution
  - `StrayCapture(root:Path, frames:list[Frame], depth_size:tuple[int,int])` with `.load_depth(index:int, min_conf:int=2) -> np.ndarray[float32, (H,W)]` metres, 0 = invalid
  - `is_stray(root) -> bool`, `load_stray(root, rgb_width:int=1920) -> StrayCapture`
  - test fixture `make_stray(tmp_path, depth_fn, poses) -> Path`

- [ ] **Step 1: Write requirements and pytest config**

`requirements.txt`:
```
numpy>=1.26,<3
scipy>=1.11
open3d==0.19.0
opencv-python-headless>=4.9,<5
shapely>=2.0
jsonschema>=4.20
matplotlib>=3.8
pytest>=8
```
`pytest.ini`:
```
[pytest]
testpaths = tests
markers =
    sample: needs sample_data/ present
```
`fpp/__init__.py` and `fpp/ingest/__init__.py`: empty.

- [ ] **Step 2: Write the fixture and failing tests**

`tests/conftest.py`:
```python
from pathlib import Path
import cv2
import numpy as np
import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"
SINGLE_ROOM = SAMPLE / "single_room" / "c00a170fe1"

HEADER = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n"


def make_stray(root: Path, depth_fn, poses, w=256, h=192, fx_rgb=1600.0):
    """Write a minimal Stray Scanner capture. poses: list of (xyz, quat_xyzw)."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    s = 1920 / w
    K = np.array([[fx_rgb, 0, w * s / 2], [0, fx_rgb, h * s / 2], [0, 0, 1]])
    np.savetxt(root / "camera_matrix.csv", K, delimiter=", ", fmt="%.4f")
    rows = []
    for i, (t, q) in enumerate(poses):
        depth_m = depth_fn(i)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), (depth_m * 1000).astype(np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full((h, w), 2, np.uint8))
        rows.append(f"{i * 0.0167:.6f}, {i:06d}, {t[0]}, {t[1]}, {t[2]}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, "
                    f"{fx_rgb}, {fx_rgb}, {K[0, 2]}, {K[1, 2]}, , \n")
    (root / "odometry.csv").write_text(HEADER + "".join(rows))
    return root


@pytest.fixture
def stray_factory(tmp_path):
    def _make(depth_fn, poses, **kw):
        return make_stray(tmp_path / "cap", depth_fn, poses, **kw)
    return _make


def need_sample():
    if not SINGLE_ROOM.exists():
        pytest.skip("sample_data not present")
```
`tests/test_stray.py`:
```python
import numpy as np
import pytest
from fpp.ingest.stray import is_stray, load_stray
from tests.conftest import SINGLE_ROOM, need_sample

IDENT = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def test_loads_poses_and_scales_intrinsics(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 2.0), [IDENT, ((1.0, 2.0, 3.0), (0, 0, 0, 1))])
    cap = load_stray(root)
    assert is_stray(root)
    assert len(cap.frames) == 2
    np.testing.assert_allclose(cap.frames[1].T_wc[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(cap.frames[0].K[0, 0], 1600 * 256 / 1920, rtol=1e-6)
    np.testing.assert_allclose(cap.frames[0].K[0, 2], 128.0, rtol=1e-6)
    assert cap.depth_size == (256, 192)


def test_depth_is_metres_and_low_confidence_masked(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 1.5), [IDENT])
    import cv2
    conf = np.full((192, 256), 2, np.uint8)
    conf[:10] = 1
    cv2.imwrite(str(root / "confidence" / "000000.png"), conf)
    d = load_stray(root).load_depth(0)
    assert d.dtype == np.float32
    assert np.all(d[:10] == 0)
    np.testing.assert_allclose(d[10:], 1.5)


def test_not_stray(tmp_path):
    assert not is_stray(tmp_path)


@pytest.mark.sample
def test_sample_single_room():
    need_sample()
    cap = load_stray(SINGLE_ROOM)
    assert len(cap.frames) == 1715
    assert cap.depth_size == (256, 192)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_stray.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.ingest.stray'`

- [ ] **Step 4: Implement `fpp/ingest/stray.py`**

```python
"""Loader for Stray Scanner exports (rgb.mp4, depth/, confidence/, odometry.csv)."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Frame:
    index: int
    timestamp: float
    T_wc: np.ndarray  # camera->world; OpenCV camera axes; world +Y up
    K: np.ndarray     # intrinsics at depth resolution


@dataclass
class StrayCapture:
    root: Path
    frames: list
    depth_size: tuple  # (w, h)

    def load_depth(self, index: int, min_conf: int = 2) -> np.ndarray:
        d = cv2.imread(str(self.root / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(self.root / "confidence" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        out = d.astype(np.float32) / 1000.0
        out[c < min_conf] = 0.0
        return out


def is_stray(root) -> bool:
    root = Path(root)
    return (root / "odometry.csv").exists() and (root / "depth").is_dir()


def load_stray(root, rgb_width: int = 1920) -> StrayCapture:
    root = Path(root)
    od = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1, usecols=range(13))
    od = np.atleast_2d(od)
    first = cv2.imread(str(root / "depth" / f"{int(od[0, 1]):06d}.png"), cv2.IMREAD_UNCHANGED)
    h, w = first.shape
    s = w / rgb_width
    frames = []
    for row in od:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T[:3, 3] = row[2:5]
        K = np.array([[row[9] * s, 0, row[11] * s], [0, row[10] * s, row[12] * s], [0, 0, 1.0]])
        frames.append(Frame(int(row[1]), float(row[0]), T, K))
    return StrayCapture(root, frames, (w, h))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stray.py -v`
Expected: 4 passed (sample test passes when sample_data present)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini fpp tests
git commit -m "Add Stray Scanner ingest with pose and intrinsics scaling"
```

---

### Task 2: Back-projection and point-cloud fusion

**Files:**
- Create: `fpp/geometry/__init__.py`, `fpp/geometry/fuse.py`
- Test: `tests/test_fuse.py`

**Interfaces:**
- Consumes: `StrayCapture`, `Frame` (Task 1)
- Produces:
  - `backproject(depth:np.ndarray, K:np.ndarray, T_wc:np.ndarray, max_depth:float=4.0) -> np.ndarray[(N,3)]`
  - `fuse_points(cap:StrayCapture, stride:int=5, max_depth:float=4.0, min_conf:int=2, voxel:float=0.02) -> np.ndarray[(N,3)]`

- [ ] **Step 1: Write failing tests**

`tests/test_fuse.py`:
```python
import numpy as np
from scipy.spatial.transform import Rotation
from fpp.geometry.fuse import backproject, fuse_points
from fpp.ingest.stray import load_stray


def K():
    return np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])


def test_backproject_identity_plane():
    pts = backproject(np.full((192, 256), 2.0, np.float32), K(), np.eye(4))
    np.testing.assert_allclose(pts[:, 2], 2.0)
    assert pts.shape == (192 * 256, 3)


def test_backproject_applies_pose_and_drops_invalid():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    T[:3, 3] = [1, 0, 0]
    d = np.full((192, 256), 2.0, np.float32)
    d[0, 0] = 0.0
    d[0, 1] = 9.0  # beyond max_depth
    pts = backproject(d, K(), T, max_depth=4.0)
    assert len(pts) == 192 * 256 - 2
    # camera z axis rotated +90 about y -> world +x; plane at x = 1 + 2
    np.testing.assert_allclose(pts[:, 0], 3.0, atol=1e-6)


def test_fuse_downsamples(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 2.0), [((0, 0, 0), (0, 0, 0, 1))] * 3)
    pts = fuse_points(load_stray(root), stride=1, voxel=0.05)
    assert 0 < len(pts) < 192 * 256
    np.testing.assert_allclose(pts[:, 2], 2.0, atol=0.03)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.geometry'`

- [ ] **Step 3: Implement `fpp/geometry/fuse.py`** (`fpp/geometry/__init__.py` empty)

```python
"""Depth back-projection into a fused world point cloud."""
import numpy as np
import open3d as o3d


def backproject(depth, K, T_wc, max_depth: float = 4.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m = (depth > 0.1) & (depth < max_depth)
    z = depth[m]
    x = (u[m] - K[0, 2]) / K[0, 0] * z
    y = (v[m] - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    return cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def fuse_points(cap, stride: int = 5, max_depth: float = 4.0, min_conf: int = 2, voxel: float = 0.02):
    chunks = [backproject(cap.load_depth(f.index, min_conf), f.K, f.T_wc, max_depth)
              for f in cap.frames[::stride]]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(chunks)))
    return np.asarray(pc.voxel_down_sample(voxel).points)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_fuse.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add fpp/geometry tests/test_fuse.py
git commit -m "Add depth back-projection and voxel-fused point cloud"
```

---

### Task 3: Synthetic room generator (test utility)

**Files:**
- Create: `tests/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces: `room_points(footprint:list[tuple[float,float]], height:float, step:float=0.02, ceiling:bool=True, noise:float=0.003, seed:int=0) -> np.ndarray[(N,3)]` — points on floor (y=0), ceiling (y=height) and walls of a polygon footprint in the x–z plane; `table_points(x0,z0,x1,z1,top=0.9,step=0.02) -> np.ndarray`

- [ ] **Step 1: Write failing test**

`tests/test_synth.py`:
```python
import numpy as np
from tests.synth import room_points


def test_box_room_extents():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.5)
    np.testing.assert_allclose(p.min(0), [0, 0, 0], atol=0.02)
    np.testing.assert_allclose(p.max(0), [4, 2.5, 3], atol=0.02)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_synth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.synth'`

- [ ] **Step 3: Implement `tests/synth.py`**

```python
"""Synthetic room point clouds for geometry tests. Plan coords (x, z); y is up."""
import numpy as np
from matplotlib.path import Path as MPath


def _grid(a0, a1, b0, b1, step):
    a = np.arange(a0, a1 + 1e-9, step)
    b = np.arange(b0, b1 + 1e-9, step)
    return np.meshgrid(a, b)


def room_points(footprint, height, step=0.02, ceiling=True, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    fp = np.asarray(footprint, float)
    xs, zs = _grid(fp[:, 0].min(), fp[:, 0].max(), fp[:, 1].min(), fp[:, 1].max(), step)
    inside = MPath(fp).contains_points(np.c_[xs.ravel(), zs.ravel()], radius=1e-9)
    fx, fz = xs.ravel()[inside], zs.ravel()[inside]
    parts = [np.c_[fx, np.zeros_like(fx), fz]]
    if ceiling:
        parts.append(np.c_[fx, np.full_like(fx, height), fz])
    for (x0, z0), (x1, z1) in zip(fp, np.roll(fp, -1, axis=0)):
        L = np.hypot(x1 - x0, z1 - z0)
        t, y = _grid(0, L, 0, height, step)
        t, y = t.ravel() / L, y.ravel()
        parts.append(np.c_[x0 + t * (x1 - x0), y, z0 + t * (z1 - z0)])
    p = np.concatenate(parts)
    return p + rng.normal(0, noise, p.shape)


def table_points(x0, z0, x1, z1, top=0.9, step=0.02):
    xs, zs = _grid(x0, x1, z0, z1, step)
    return np.c_[xs.ravel(), np.full(xs.size, top), zs.ravel()]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_synth.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tests/synth.py tests/test_synth.py
git commit -m "Add synthetic room point-cloud generator for tests"
```

---

### Task 4: Plane extraction and classification

**Files:**
- Create: `fpp/geometry/planes.py`
- Test: `tests/test_planes.py`

**Interfaces:**
- Consumes: point cloud `np.ndarray[(N,3)]` (Task 2), `room_points`, `table_points` (Task 3)
- Produces:
  - `Plane(normal:np.ndarray[3], d:float, inliers:np.ndarray[(M,3)], rms:float, kind:str)` — plane `normal·x + d = 0`; `kind ∈ {"floor","ceiling","wall","other"}`; horizontal planes have `normal[1] > 0`
  - `Plane.height -> float` (for horizontal planes: `-d / normal[1]`)
  - `extract_planes(points, dist=0.02, min_inliers=1500, max_planes=30) -> list[Plane]` (kind="other" before classification)
  - `classify_planes(planes) -> dict` with keys `"floor": Plane`, `"ceiling": Plane|None`, `"walls": list[Plane]`, `"other": list[Plane]`

- [ ] **Step 1: Write failing tests**

`tests/test_planes.py`:
```python
import numpy as np
from fpp.geometry.planes import classify_planes, extract_planes
from tests.synth import room_points, table_points


def test_box_room_planes():
    p = np.concatenate([room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6),
                        table_points(1, 1, 2, 1.8, top=0.9)])
    c = classify_planes(extract_planes(p))
    assert abs(c["floor"].height - 0.0) < 0.01
    assert abs(c["ceiling"].height - 2.6) < 0.01
    assert len(c["walls"]) == 4
    assert any(abs(o.height - 0.9) < 0.02 for o in c["other"])  # table is not floor/ceiling
    for w in c["walls"]:
        assert abs(w.normal[1]) < 0.05


def test_no_ceiling_returns_none():
    p = room_points([(0, 0), (3, 0), (3, 3), (0, 3)], 2.5, ceiling=False)
    c = classify_planes(extract_planes(p))
    assert c["ceiling"] is None


def test_deterministic():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6)
    a = extract_planes(p)
    b = extract_planes(p)
    assert [round(x.d, 6) for x in a] == [round(x.d, 6) for x in b]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_planes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.geometry.planes'`

- [ ] **Step 3: Implement `fpp/geometry/planes.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_planes.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add fpp/geometry/planes.py tests/test_planes.py
git commit -m "Add seeded RANSAC plane extraction and floor/ceiling/wall classification"
```

---

### Task 5: Room footprint and wall lengths

**Files:**
- Create: `fpp/geometry/room.py`
- Test: `tests/test_room.py`

**Interfaces:**
- Consumes: `classify_planes` output (Task 4)
- Produces:
  - `manhattan_angle(walls:list[Plane]) -> float` radians; rotation about +Y aligning dominant wall normals to plan axes
  - `to_plan(points, angle) -> np.ndarray[(N,2)]` — world (x,z) rotated by `-angle`
  - `Edge(axis:str, offset:float, start:float, end:float, support:int, rms:float)` — `axis "u"`: line `u = offset` spanning v∈[start,end]; `axis "v"`: line `v = offset`, u∈[start,end]; `support` = inliers of the snapping wall plane (0 if none)
  - `footprint(classes:dict, cell:float=0.02) -> tuple[np.ndarray[(K,2)], list[Edge], float]` — rectilinear polygon vertices in plan frame (CCW), its edges, and the Manhattan angle

- [ ] **Step 1: Write failing tests**

`tests/test_room.py`:
```python
import numpy as np
from shapely.geometry import Polygon
from fpp.geometry.planes import classify_planes, extract_planes
from fpp.geometry.room import footprint
from tests.synth import room_points


def _rot(p, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return p @ R.T


def test_rotated_box_lengths_and_area():
    p = _rot(room_points([(0, 0), (4.2, 0), (4.2, 3.1), (0, 3.1)], 2.5), 27)
    poly, edges, _ = footprint(classify_planes(extract_planes(p)))
    lengths = sorted(round(abs(e.end - e.start), 2) for e in edges)
    np.testing.assert_allclose(lengths, [3.1, 3.1, 4.2, 4.2], atol=0.01)
    assert abs(Polygon(poly).area - 4.2 * 3.1) < 0.05


def test_l_shape_keeps_notch():
    fp = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 4), (0, 4)]
    p = room_points(fp, 2.5)
    poly, edges, _ = footprint(classify_planes(extract_planes(p, min_inliers=800)))
    assert len(edges) == 6
    assert abs(Polygon(poly).area - (5 * 2 + 3 * 2)) < 0.08
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_room.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.geometry.room'`

- [ ] **Step 3: Implement `fpp/geometry/room.py`**

```python
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
```
Note on vertices: vertex `i` is the corner between segment `i` and segment `i+1`; edge `i` runs from vertex `i-1` to vertex `i`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_room.py -v`
Expected: 2 passed. If the L-shape test fails on edge count, print `segs` and tune `eps` (approxPolyDP tolerance) — do not loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add fpp/geometry/room.py tests/test_room.py
git commit -m "Add Manhattan rectilinear footprint with wall-plane edge snapping"
```

---

### Task 6: Openings by free-space carving

**Files:**
- Create: `fpp/geometry/openings.py`
- Test: `tests/test_openings.py`

**Interfaces:**
- Consumes: `StrayCapture` (Task 1), `Edge`, `to_plan` (Task 5), floor height (Task 4)
- Produces:
  - `Opening(edge_index:int, kind:str, s0:float, s1:float, bottom:float, top:float, evidence:int)` — `kind ∈ {"door","window"}`, `s0,s1` along-edge coordinates in metres measured from `edge.start` toward `edge.end`; heights above floor
  - `Opening.width -> float`, `Opening.height -> float`
  - `find_openings(rays:Iterable[tuple[np.ndarray[(3,)], np.ndarray[(N,3)]]], edges:list[Edge], angle:float, floor_h:float, wall_h:float, cell:float=0.02) -> list[Opening]` — `rays` yields `(camera_center_world, endpoint_world_points)` per frame
  - `capture_rays(cap, stride=5, max_depth=6.0, pixel_step=2) -> Iterator` — yields the above from a Stray capture

- [ ] **Step 1: Write failing tests**

`tests/test_openings.py`:
```python
import numpy as np
from fpp.geometry.openings import find_openings
from fpp.geometry.room import Edge


def _rays_through(wall_v, hole, behind=1.0, cam=(2.0, 1.4, 1.0), seed=0):
    """Camera inside room (v < wall_v), rays to points behind wall line v = wall_v.
    Points inside `hole` (u0, u1, y0, y1) are visible beyond the wall."""
    rng = np.random.default_rng(seed)
    u0, u1, y0, y1 = hole
    u = rng.uniform(u0, u1, 20000)
    y = rng.uniform(y0, y1, 20000)
    c = np.array(cam)
    # endpoint = point on wall hole extended `behind` metres past the wall, along the ray
    hit = np.c_[u, y, np.full_like(u, wall_v)]
    dirn = hit - c
    t = (wall_v + behind - c[2]) / dirn[:, 2]
    return [(c, c + dirn * t[:, None])]


def test_door_width():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)    # wall v=3 from u=0..4, plan(x,z)=(u,v)
    ops = find_openings(_rays_through(3.0, (1.0, 1.9, 0.0, 2.05)), [edge], 0.0, 0.0, 2.5)
    assert len(ops) == 1
    assert ops[0].kind == "door"
    assert abs(ops[0].width - 0.9) < 0.02
    assert abs(ops[0].s0 - 1.0) < 0.02


def test_window_not_door():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)
    ops = find_openings(_rays_through(3.0, (2.5, 3.7, 0.9, 2.1)), [edge], 0.0, 0.0, 2.5)
    assert [o.kind for o in ops] == ["window"]
    assert abs(ops[0].width - 1.2) < 0.02


def test_unobserved_wall_is_not_opening():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)
    assert find_openings([], [edge], 0.0, 0.0, 2.5) == []


def test_narrow_gap_rejected():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)
    assert find_openings(_rays_through(3.0, (1.0, 1.3, 0.0, 2.0)), [edge], 0.0, 0.0, 2.5) == []
```
Note: in tests `angle=0`, so plan `(u, v) = (x, z)` and the world point `(u, y, v)` is passed as `(x, y, z)`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_openings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.geometry.openings'`

- [ ] **Step 3: Implement `fpp/geometry/openings.py`**

```python
"""Openings = wall regions that depth rays pass THROUGH (free-space evidence).
An unobserved wall region has no evidence and is never an opening."""
from dataclasses import dataclass

import cv2
import numpy as np

from fpp.geometry.fuse import backproject
from fpp.geometry.room import to_plan

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
        free = (g >= MIN_HITS).astype(np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        for j in range(1, n):
            x, yv, w, hgt, _ = stats[j]
            width, height = w * cell, hgt * cell
            if width < MIN_WIDTH or height < MIN_HEIGHT or height > wall_h - 0.05:
                continue
            bottom = yv * cell
            kind = "door" if bottom < DOOR_BOTTOM else "window"
            out.append(Opening(i, kind, x * cell, (x + w) * cell, bottom, bottom + height,
                               int(g[lab == j].sum())))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_openings.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add fpp/geometry/openings.py tests/test_openings.py
git commit -m "Add opening detection via free-space ray carving through walls"
```

---

### Task 7: Measurements with intervals + room assembly

**Files:**
- Create: `fpp/uncertainty.py`, `fpp/measure.py`
- Test: `tests/test_measure.py`

**Interfaces:**
- Consumes: `classify_planes` dict (Task 4), `footprint` (Task 5), `Opening` (Task 6)
- Produces:
  - `TIER_SIGMA_FLOOR = {"lidar": 0.008, "video": 0.03, "photo": 0.08}` (metres, per-edge minimum σ; video/photo are relative, see `edge_sigma`)
  - `Measure(value:float, lo:float, hi:float, unit:str)` with `.to_dict()`
  - `interval(value, sigma, unit="m", calib=1.0) -> Measure` (95%: `1.96*sigma*calib`)
  - `edge_sigma(edge, tier) -> float`
  - `measure_room(classes, verts, edges, angle, openings, tier="lidar", calib=1.0) -> dict` (room dict matching schema in Task 8; also includes `"warnings": list[str]`)

- [ ] **Step 1: Write failing tests**

`tests/test_measure.py`:
```python
import numpy as np
from fpp.geometry.planes import classify_planes, extract_planes
from fpp.geometry.room import footprint
from fpp.measure import measure_room
from fpp.uncertainty import interval
from tests.synth import room_points


def test_interval_symmetric_95():
    m = interval(2.0, 0.01)
    assert m.lo < 2.0 < m.hi
    assert abs((m.hi - m.lo) / 2 - 0.0196) < 1e-9


def _room(ceiling=True):
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6, ceiling=ceiling)
    c = classify_planes(extract_planes(p))
    v, e, a = footprint(c)
    return measure_room(c, v, e, a, [])


def test_room_measures_contain_truth():
    r = _room()
    ch = r["ceiling_height"]
    assert ch["lo"] <= 2.6 <= ch["hi"]
    assert r["ceiling_observed"] is True
    fa = r["floor_area"]
    assert fa["lo"] <= 12.0 <= fa["hi"]
    for w in r["walls"]:
        L = w["length"]
        assert any(L["lo"] <= t <= L["hi"] for t in (3.0, 4.0))


def test_missing_ceiling_is_honest():
    r = _room(ceiling=False)
    assert r["ceiling_observed"] is False
    ch = r["ceiling_height"]
    assert ch["hi"] - ch["lo"] >= 0.3
    assert any("ceiling" in w for w in r["warnings"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_measure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.measure'`

- [ ] **Step 3: Implement `fpp/uncertainty.py`**

```python
"""95% intervals. Per-tier sigma floors; `calib` inflation is fitted on the benchmark (Plan 6)."""
from dataclasses import dataclass

import numpy as np

Z95 = 1.96
TIER_SIGMA_FLOOR = {"lidar": 0.008, "video": 0.03, "photo": 0.08}
TIER_REL = {"lidar": 0.0, "video": 0.015, "photo": 0.04}   # relative scale uncertainty


@dataclass
class Measure:
    value: float
    lo: float
    hi: float
    unit: str = "m"

    def to_dict(self):
        return {"value": round(self.value, 4), "lo": round(self.lo, 4),
                "hi": round(self.hi, 4), "unit": self.unit}


def interval(value, sigma, unit="m", calib=1.0):
    h = Z95 * sigma * calib
    return Measure(float(value), float(value - h), float(value + h), unit)


def edge_sigma(edge, tier):
    """Std of an edge's offset: plane-fit standard error, floored by tier, or raster if unsupported."""
    if edge.support == 0:
        return max(0.03, TIER_SIGMA_FLOOR[tier])
    return max(edge.rms / np.sqrt(edge.support / 50.0), TIER_SIGMA_FLOOR[tier])
```

- [ ] **Step 4: Implement `fpp/measure.py`**

```python
"""Assemble a room record (schema 'room') from geometry results."""
import numpy as np
from shapely.geometry import Polygon

from fpp.uncertainty import TIER_REL, edge_sigma, interval

UNSEEN_CEILING_SIGMA = 0.15


def _ceiling(classes, tier, calib, warnings):
    floor, ceil = classes["floor"], classes["ceiling"]
    if ceil is not None:
        s = np.hypot(floor.rms / np.sqrt(len(floor.inliers) / 50), ceil.rms / np.sqrt(len(ceil.inliers) / 50))
        s = max(s, 0.005 if tier == "lidar" else 0.02)
        h = ceil.height - floor.height
        return interval(h, np.hypot(s, TIER_REL[tier] * h), calib=calib), True
    top = max(float(np.percentile(w.inliers[:, 1], 99.5)) for w in classes["walls"]) - floor.height
    warnings.append("ceiling not observed: height is lower-bounded by highest wall point")
    return interval(top + UNSEEN_CEILING_SIGMA, UNSEEN_CEILING_SIGMA * 1.0, calib=calib), False


def measure_room(classes, verts, edges, angle, openings, tier="lidar", calib=1.0, room_id="room_0"):
    warnings = []
    n = len(edges)
    walls = []
    for i, e in enumerate(edges):
        L = abs(e.end - e.start)
        s_prev = edge_sigma(edges[i - 1], tier)
        s_next = edge_sigma(edges[(i + 1) % n], tier)
        sig = np.sqrt(s_prev ** 2 + s_next ** 2 + (TIER_REL[tier] * L) ** 2)
        walls.append({"id": f"{room_id}_w{i}", "surface_id": f"{room_id}_w{i}",
                      "start": [round(float(x), 4) for x in verts[i - 1]],
                      "end": [round(float(x), 4) for x in verts[i]],
                      "length": interval(L, sig, calib=calib).to_dict(),
                      "plane_supported": e.support > 0})
    poly = Polygon(verts)
    perim = poly.length
    s_off = np.mean([edge_sigma(e, tier) for e in edges])
    area = poly.area
    area_sig = np.hypot(perim * s_off / np.sqrt(2), 2 * TIER_REL[tier] * area)
    ceiling, observed = _ceiling(classes, tier, calib, warnings)
    ops = []
    for j, o in enumerate(openings):
        e = edges[o.edge_index]
        s_w = np.hypot(0.01 if tier == "lidar" else 0.03, TIER_REL[tier] * o.width)
        ops.append({"id": f"{room_id}_o{j}", "wall_id": f"{room_id}_w{o.edge_index}", "type": o.kind,
                    "offset_along_wall": round(o.s0, 4),
                    "width": interval(o.width, s_w, calib=calib).to_dict(),
                    "height": interval(o.height, s_w, calib=calib).to_dict(),
                    "sill_height": interval(o.bottom, s_w, calib=calib).to_dict()})
    return {"id": room_id, "name": room_id,
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in verts],
            "walls": walls,
            "floor_area": interval(area, area_sig, unit="m2", calib=calib).to_dict(),
            "perimeter": interval(perim, s_off * np.sqrt(n), calib=calib).to_dict(),
            "ceiling_height": ceiling.to_dict(), "ceiling_observed": observed,
            "openings": ops, "warnings": warnings}
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_measure.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add fpp/uncertainty.py fpp/measure.py tests/test_measure.py
git commit -m "Add per-measurement 95% intervals and room record assembly"
```

---

### Task 8: Published JSON schema, renderer, and `run.py`

**Files:**
- Create: `schema/plan.schema.json`, `fpp/output.py`, `fpp/render.py`, `fpp/pipeline.py`, `run.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `build_plan(capture_id:str, tier:str, rooms:list[dict], meta:dict) -> dict` (top-level document)
  - `validate(plan:dict) -> None` (raises `jsonschema.ValidationError`)
  - `render(plan:dict, out_stem:Path) -> None` writes `<stem>.svg` and `<stem>.png`
  - `run_lidar_single(capture_dir:Path) -> dict` (the plan)
  - CLI: `python run.py <capture_dir> [--out DIR]` → `DIR/plan.json`, `DIR/plan.svg`, `DIR/plan.png`; default `DIR = outputs/<capture_dir name>`

- [ ] **Step 1: Write the schema `schema/plan.schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Floor plan capture result",
  "type": "object",
  "required": ["schema_version", "capture", "rooms", "adjacency", "damage",
               "concealed_damage_flags", "scope_items", "warnings"],
  "$defs": {
    "measure": {"type": "object", "required": ["value", "lo", "hi", "unit"],
                "properties": {"value": {"type": "number"}, "lo": {"type": "number"},
                               "hi": {"type": "number"}, "unit": {"enum": ["m", "m2"]}}},
    "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
  },
  "properties": {
    "schema_version": {"const": "1.0"},
    "capture": {"type": "object", "required": ["id", "tier"],
                "properties": {"id": {"type": "string"}, "tier": {"enum": ["lidar", "video", "photo"]}}},
    "rooms": {"type": "array", "items": {
      "type": "object",
      "required": ["id", "polygon", "walls", "floor_area", "ceiling_height", "ceiling_observed", "openings"],
      "properties": {
        "polygon": {"type": "array", "items": {"$ref": "#/$defs/point"}, "minItems": 3},
        "walls": {"type": "array", "items": {"type": "object",
          "required": ["id", "surface_id", "start", "end", "length"],
          "properties": {"length": {"$ref": "#/$defs/measure"}}}},
        "floor_area": {"$ref": "#/$defs/measure"},
        "ceiling_height": {"$ref": "#/$defs/measure"},
        "ceiling_observed": {"type": "boolean"},
        "openings": {"type": "array", "items": {"type": "object",
          "required": ["id", "wall_id", "type", "width"],
          "properties": {"type": {"enum": ["door", "window", "opening"]},
                         "width": {"$ref": "#/$defs/measure"}}}}
      }}},
    "adjacency": {"type": "array"},
    "damage": {"type": "array"},
    "concealed_damage_flags": {"type": "array"},
    "scope_items": {"type": "array"},
    "warnings": {"type": "array", "items": {"type": "string"}}
  }
}
```

- [ ] **Step 2: Write failing tests**

`tests/test_pipeline.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fpp.output import build_plan, validate
from tests.conftest import SINGLE_ROOM, need_sample

ROOT = Path(__file__).resolve().parents[1]


def test_empty_plan_validates():
    validate(build_plan("x", "lidar", [], {}))


@pytest.mark.sample
def test_run_single_room_end_to_end(tmp_path):
    need_sample()
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), str(SINGLE_ROOM), "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    plan = json.loads((tmp_path / "plan.json").read_text())
    validate(plan)
    room = plan["rooms"][0]
    assert 2.0 < room["ceiling_height"]["value"] < 3.2
    assert 3.0 < room["floor_area"]["value"] < 30.0
    assert len(room["walls"]) >= 4
    assert (tmp_path / "plan.svg").stat().st_size > 1000


@pytest.mark.sample
def test_deterministic_output(tmp_path):
    need_sample()
    outs = []
    for k in range(2):
        d = tmp_path / str(k)
        subprocess.run([sys.executable, str(ROOT / "run.py"), str(SINGLE_ROOM), "--out", str(d)], check=True,
                       capture_output=True, timeout=600)
        p = json.loads((d / "plan.json").read_text())
        p["meta"].pop("runtime_s", None)
        outs.append(p)
    assert outs[0] == outs[1]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpp.output'`

- [ ] **Step 4: Implement `fpp/output.py`**

```python
import json
from pathlib import Path

import jsonschema

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema" / "plan.schema.json").read_text())


def build_plan(capture_id, tier, rooms, meta):
    warnings = [f"{r['id']}: {w}" for r in rooms for w in r.get("warnings", [])]
    return {"schema_version": "1.0", "capture": {"id": capture_id, "tier": tier},
            "rooms": rooms, "adjacency": [], "damage": [], "concealed_damage_flags": [],
            "scope_items": [], "warnings": warnings, "meta": meta}


def validate(plan):
    jsonschema.validate(plan, SCHEMA)
```

- [ ] **Step 5: Implement `fpp/render.py`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(m):
    return f"{m['value']:.2f} m (±{(m['hi'] - m['lo']) / 2 * 100:.0f} cm)"


def render(plan, out_stem):
    fig, ax = plt.subplots(figsize=(10, 10))
    for room in plan["rooms"]:
        poly = np.array(room["polygon"] + [room["polygon"][0]])
        ax.fill(poly[:, 0], poly[:, 1], color="#f3efe6", zorder=0)
        ax.plot(poly[:, 0], poly[:, 1], color="#333", lw=3, zorder=1)
        walls = {w["id"]: w for w in room["walls"]}
        for w in room["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            mid = (a + b) / 2
            d = b - a
            nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang   # keep text upright
            ax.text(*(mid + nrm * 0.25), _fmt(w["length"]), ha="center", va="center", fontsize=8, rotation=ang)
        for o in room["openings"]:
            w = walls[o["wall_id"]]
            a, b = np.array(w["start"]), np.array(w["end"])
            u = (b - a) / np.linalg.norm(b - a)
            p0 = a + u * o["offset_along_wall"]
            p1 = p0 + u * o["width"]["value"]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#2b8cbe" if o["type"] == "window" else "#e6550d",
                    lw=6, zorder=2)
        c = poly[:-1].mean(0)
        ax.text(*c, f"{room['name']}\n{room['floor_area']['value']:.2f} m²\n"
                    f"ceiling {room['ceiling_height']['value']:.2f} m", ha="center", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{plan['capture']['id']} · tier: {plan['capture']['tier']}")
    for ext in ("svg", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 6: Implement `fpp/pipeline.py` and `run.py`**

`fpp/pipeline.py`:
```python
import time
from pathlib import Path

from fpp.geometry.fuse import fuse_points
from fpp.geometry.openings import capture_rays, find_openings
from fpp.geometry.planes import classify_planes, extract_planes
from fpp.geometry.room import footprint
from fpp.ingest.stray import load_stray
from fpp.measure import measure_room
from fpp.output import build_plan


def run_lidar_single(capture_dir):
    t0 = time.time()
    cap = load_stray(capture_dir)
    pts = fuse_points(cap)
    classes = classify_planes(extract_planes(pts))
    verts, edges, angle = footprint(classes)
    wall_h = (classes["ceiling"].height if classes["ceiling"] else classes["floor"].height + 2.6) \
        - classes["floor"].height
    ops = find_openings(capture_rays(cap), edges, angle, classes["floor"].height, wall_h)
    room = measure_room(classes, verts, edges, angle, ops, tier="lidar")
    meta = {"n_frames": len(cap.frames), "n_points": int(len(pts)),
            "manhattan_angle_rad": round(angle, 6), "runtime_s": round(time.time() - t0, 1)}
    return build_plan(Path(capture_dir).name, "lidar", [room], meta)
```
`run.py`:
```python
"""One command per capture: python run.py <capture_dir> [--out DIR]"""
import argparse
import json
from pathlib import Path

from fpp.ingest.stray import is_stray
from fpp.output import validate
from fpp.pipeline import run_lidar_single
from fpp.render import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path("outputs") / a.capture_dir.name
    out.mkdir(parents=True, exist_ok=True)
    if not is_stray(a.capture_dir):
        raise SystemExit(f"{a.capture_dir}: not a Stray Scanner capture (video/photo tiers: Plans 3-4)")
    plan = run_lidar_single(a.capture_dir)
    validate(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    render(plan, out / "plan")
    for r in plan["rooms"]:
        print(f"{r['id']}: area {r['floor_area']['value']:.2f} m², ceiling {r['ceiling_height']['value']:.3f} m, "
              f"{len(r['walls'])} walls, {len(r['openings'])} openings")
    print(f"wrote {out}/plan.json, plan.svg, plan.png in {plan['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests to verify pass**

Run: `.venv/bin/pytest -v`
Expected: all pass. Then run `.venv/bin/python run.py sample_data/single_room/c00a170fe1` and open `outputs/c00a170fe1/plan.png` to eyeball: walls closed, dimensions plausible, door(s) on the right walls, glass shower not a phantom door (Review Focus 1). Record observed numbers in the commit message.

- [ ] **Step 8: Commit**

```bash
git add schema fpp/output.py fpp/render.py fpp/pipeline.py run.py tests/test_pipeline.py
git commit -m "Add plan schema, renderer and one-command LiDAR single-room run"
```

---

### Task 9: README quickstart

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# Floor-plan pipeline

Turn an iPhone capture into a dimensioned floor plan (JSON + SVG/PNG) with a 95%
interval on every measurement.

## Quickstart (clean Linux machine, Python 3.10+)
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python run.py <capture_dir>          # -> outputs/<name>/plan.json, plan.svg, plan.png

Capture with the free **Stray Scanner** iOS app (see `docs/capture_protocol.md`).

## Tests
    .venv/bin/pytest            # synthetic tests; sample-data tests run when sample_data/ exists

## Status
LiDAR tier, single room: implemented (Plan 1). Multi-room stitch, video, photo, damage: in progress.
```

- [ ] **Step 2: Verify the quickstart literally works** — run the two quickstart commands in a fresh clone (`git clone . /tmp/fpp-check && cd /tmp/fpp-check && ...`, pointing at the sample data path). Expected: plan files written.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README quickstart"
```
