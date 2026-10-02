import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from roomscan.pipeline import run_lidar
from roomscan.save_plan import validate
from tests.conftest import HEADER
from tests.synthetic_rooms import render_box_depth
from tests.test_drift_correction import LO, HI, _poses


def _box_capture(root, n=240):
    for d in ("depth", "confidence", "rgb"):
        (root / d).mkdir(parents=True, exist_ok=True)
    w, h, f = 64, 48, 1600 * 64 / 1920
    K = np.array([[f, 0, 32], [0, f, 24], [0, 0, 1]])
    rows = []
    for i, T in enumerate(_poses(n)):
        depth = render_box_depth(K, T, LO, HI, w, h)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), (depth * 1000).astype(np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full((h, w), 2, np.uint8))
        cv2.imwrite(str(root / "rgb" / f"{i:06d}.jpg"), np.full((h, w, 3), 200, np.uint8))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        t = T[:3, 3]
        rows.append(f"{i / 10:.6f}, {i:06d}, {t[0]}, {t[1]}, {t[2]}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, "
                    f"1600, 1600, 960, 720, , \n")
    (root / "odometry.csv").write_text(HEADER + "".join(rows))
    return root


def fake_detect(img):
    m = np.zeros(img.shape[:2], bool)
    m[20:28, 28:36] = True                     # small patch at the image centre: lands on a wall
    return [{"class": "water_stain", "score": 0.6, "mask": m, "box": (28, 20, 36, 28)}]


def test_lidar_run_fills_damage_flags_scope_and_floor_level(tmp_path):
    root = _box_capture(tmp_path / "cap")
    plan = run_lidar(root, stride=3, damage=True, detect=fake_detect)
    validate(plan)
    assert plan["damage"], "fake stains on walls must be measured"
    assert all(d["surface_id"].startswith(d["room_id"]) for d in plan["damage"])
    assert any(i["action"] == "stain_block_and_repaint" for i in plan["scope_items"])
    assert all(r["floor_level"] == 0.0 for r in plan["rooms"] if len(plan["rooms"]) == 1)
    assert "damage_s" in plan["meta"]


def test_run_py_no_damage_flag(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = _box_capture(tmp_path / "cap", n=120)
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "run.py"), str(root),
                        "--out", str(tmp_path / "o"), "--no-damage"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    plan = json.loads((tmp_path / "o" / "plan.json").read_text())
    assert plan["damage"] == [] and "damage_s" not in plan["meta"]


def test_lidar_run_without_damage_leaves_arrays_empty(tmp_path):
    root = _box_capture(tmp_path / "cap", n=120)
    plan = run_lidar(root, stride=3)
    assert plan["damage"] == [] and plan["scope_items"] == []
