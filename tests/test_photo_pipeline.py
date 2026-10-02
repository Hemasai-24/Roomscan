import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from roomscan.photo_pipeline import run_photos, widen_room
from roomscan.save_plan import validate
from tests.test_photo_links import _photo, _texture
from tests.test_stitch_rooms import info, make_room

import cv2


def _photo_set(tmp_path):
    shared = cv2.resize(_texture(99), (150, 150))
    for name, imgs in {"hall": [_photo(1), _photo(2, shared, at=(300, 200))],
                       "kitchen": [_photo(3, cv2.resize(shared, (170, 170)), at=(250, 180)), _photo(4)],
                       "box": [_photo(5)]}.items():
        (tmp_path / name).mkdir(parents=True)
        for k, im in enumerate(imgs):
            Image.fromarray(im).save(tmp_path / name / f"{k}.jpg", quality=95)
    return tmp_path


def _fake_reconstruct(imgs, fx, out_dir, rid):
    rooms = {"hall": (make_room("hall", [(0, 0), (4, 0), (4, 3), (0, 3)], doors=[(2, 1.0, 0.9)]),
                      [(2, 1.5), (2, 1.5)], [(-1, 0), (1, 0)]),
             "kitchen": (make_room("kitchen", [(0, 0), (3, 0), (3, 3), (0, 3)], doors=[(0, 1.2, 0.9)]),
                         [(1.5, 1.5), (1.5, 1.5)], [(-1, 0), (1, 0)])}
    room, cams, fwds = rooms[rid]
    for w in room["walls"]:
        w["length"] = {"value": w["length"]["value"], "lo": w["length"]["value"] * 0.95,
                       "hi": w["length"]["value"] * 1.05, "unit": "m"}
        w["surface_id"] = w["id"]
    a = room["floor_area"]["value"]
    room.update(name=rid, floor_area={"value": a, "lo": a * 0.9, "hi": a * 1.1, "unit": "m2"},
                ceiling_height={"value": 2.5, "lo": 2.2, "hi": 2.9, "unit": "m"}, ceiling_observed=False,
                warnings=[])
    for op in room["openings"]:
        op["width"].update(lo=op["width"]["value"] - 0.05, hi=op["width"]["value"] + 0.05, unit="m")
    return dict(info(room, cams, fwds), warnings=[]), {"n_photos": len(imgs), "metric_scale_spread": 0.05}


def test_photo_set_gives_one_stitched_plan(tmp_path):
    plan = run_photos(_photo_set(tmp_path / "in"), tmp_path / "out", reconstruct=_fake_reconstruct)
    validate(plan)
    assert plan["capture"]["tier"] == "photo"
    assert [r["id"] for r in plan["rooms"]] == ["hall", "kitchen"]
    assert [(e["room_a"], e["room_b"]) for e in plan["adjacency"]] == [("hall", "kitchen")]
    a, b = (Polygon(r["polygon"]) for r in plan["rooms"])
    assert a.intersection(b).area < 0.1
    assert any("box" in w and "at least 2" in w for w in plan["warnings"])


def test_widen_adds_scale_uncertainty():
    room = {"walls": [{"length": {"value": 4.0, "lo": 3.9, "hi": 4.1, "unit": "m"}}],
            "floor_area": {"value": 12.0, "lo": 11.0, "hi": 13.0, "unit": "m2"},
            "perimeter": {"value": 14.0, "lo": 13.5, "hi": 14.5, "unit": "m"},
            "ceiling_height": {"value": 2.5, "lo": 2.4, "hi": 2.6, "unit": "m"}, "openings": []}
    widen_room(room, 0.05)
    L = room["walls"][0]["length"]
    np.testing.assert_allclose((L["hi"] - L["lo"]) / 2, np.hypot(0.1, 1.96 * 0.05 * 4.0), rtol=1e-3)
    A = room["floor_area"]
    np.testing.assert_allclose((A["hi"] - A["lo"]) / 2, np.hypot(1.0, 1.96 * 0.10 * 12.0), rtol=1e-3)


def test_cli_detects_photo_set(tmp_path):
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "run.py"), str(tmp_path / "nothing"), "--tier", "photo"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "photo" in (r.stderr + r.stdout)
