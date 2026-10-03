import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from roomscan import pipeline
from roomscan.find_surfaces import CaptureError, classify_planes, extract_planes
from roomscan.point_cloud import estimate_normals
from roomscan.room_outline import manhattan_angle
from roomscan.split_rooms import split_rooms
from tests.synthetic_rooms import two_rooms_with_door

ROOT = Path(__file__).resolve().parents[1]


def _two_rooms():
    p = two_rooms_with_door()
    c = classify_planes(extract_planes(p, estimate_normals(p), min_inliers=500))
    a = manhattan_angle(c["walls"])
    masks, g = split_rooms(c, a)
    return c, masks, g, a


def test_one_failing_room_is_skipped_with_warning(monkeypatch):
    c, masks, g, a = _two_rooms()
    real = pipeline.outline_from_mask
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("A LinearRing must have at least 3 coordinate tuples")
        return real(*args, **kw)

    monkeypatch.setattr(pipeline, "outline_from_mask", flaky)
    rooms, kept, geoms, warnings = pipeline.measure_rooms(c, masks, g, a, rays=[], tier="lidar")
    assert len(rooms) == 1 and len(kept) == 1
    assert any("skipped" in w for w in warnings)


def test_no_rooms_is_a_capture_error():
    c, _, g, a = _two_rooms()
    with pytest.raises(CaptureError):
        pipeline.measure_rooms(c, [], g, a, rays=[], tier="lidar")


def test_nested_stray_folder_is_found(stray_factory, tmp_path):
    cap = stray_factory(lambda i: np.full((192, 256), 2.0), [((0, 0, 0), (0, 0, 0, 1))])
    assert pipeline.detect_tier(cap.parent) == ("lidar", cap)


def test_run_prints_one_line_on_capture_error(stray_factory, tmp_path):
    cap = stray_factory(lambda i: np.full((192, 256), 2.0), [((0, 0, 0), (0, 0, 0, 1))] * 3)   # one flat wall only
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), str(cap), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr and "cannot make a plan" in r.stderr
