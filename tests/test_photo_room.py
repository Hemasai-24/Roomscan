import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

from roomscan.load_capture import load_stray
from roomscan.photo_room import measure_photo_room, photos_to_capture
from tests.synthetic_rooms import render_box_depth

LO, HI = np.array([0.0, -1.4, 0.0]), np.array([5.0, 1.2, 4.0])     # 5 x 4 m room, camera ~1.4 m up
W, H = 120, 160
K = np.array([[110.0, 0, W / 2], [0, 110.0, H / 2], [0, 0, 1]])


def _upright(yaw_deg, pos):
    """Camera->world for an upright phone (image-down = world -Y) looking along yaw."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = pos
    return T


def _views():
    """Photos from the 4 corners looking at the opposite corners, plus 2 along the long walls."""
    return [_upright(y, p) for y, p in [(-135, (1.0, 0.0, 1.0)), (135, (4.0, 0.0, 1.0)), (45, (4.0, 0.0, 3.0)),
                                       (-45, (1.0, 0.0, 3.0)), (-90, (2.5, 0.0, 2.0)), (90, (2.5, 0.0, 2.0))]]


class FakeRunner:
    """Stands in for VGGT: returns the true geometry in a rotated frame at half scale (units unknown)."""

    def __init__(self, cut_floor=False):
        self.cut_floor = cut_floor
        self.G = np.eye(4)
        self.G[:3, :3] = Rotation.from_euler("xz", [20, -10], degrees=True).as_matrix()

    def run(self, images):
        T, D = [], []
        for Tw in _views()[: len(images)]:
            d = render_box_depth(K, Tw, LO, HI, W, H)
            if self.cut_floor:
                u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
                ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
                y = (ray * d[..., None]) @ Tw[:3, :3].T[:, 1] + Tw[1, 3]
                d = np.where(y < LO[1] + 0.05, 0.0, d)
            G = self.G.copy()
            T.append(G @ Tw)
            T[-1][:3, 3] *= 0.5
            D.append(d * 0.5)
        n = len(images)
        return {"T": T, "depth": np.stack(D), "conf": np.ones((n, H, W)), "K": np.stack([K] * n)}


class FakeMetric:
    """Stands in for Depth Anything: true depth times the calibrated bias."""

    def __init__(self, runner, bias):
        self.runner, self.bias = runner, bias

    def predict(self, images):
        return [d / 0.5 * self.bias for d in self.runner.run(images)["depth"]]


def _capture(tmp_path, cut_floor=False, fx=None):
    runner = FakeRunner(cut_floor)
    imgs = np.zeros((6, H, W, 3), np.uint8)
    return photos_to_capture(imgs, fx, tmp_path / "cap", runner, FakeMetric(runner, 1.4), 1.4)


def test_photos_become_a_metric_level_capture(tmp_path):
    cap_dir, info = _capture(tmp_path)
    cap = load_stray(cap_dir)
    assert len(cap.frames) == 6
    assert info["focal_source"] == "model"
    np.testing.assert_allclose(info["metric_scale"], 2.0, rtol=0.01)
    heights = [f.T_wc[1, 3] for f in cap.frames]
    assert np.ptp(heights) < 0.02                      # levelled: all cameras at the same height


def test_exif_focal_replaces_model_focal(tmp_path):
    cap_dir, info = _capture(tmp_path, fx=123.0)
    assert info["focal_source"] == "exif"
    assert abs(load_stray(cap_dir).frames[0].K[0, 0] - 123.0) < 0.5      # loader gives K at depth size


def test_one_room_measured_from_its_photos(tmp_path):
    cap_dir, _ = _capture(tmp_path)
    out = measure_photo_room(cap_dir, "kitchen")
    room = out["room"]
    assert room["id"] == "kitchen"
    assert abs(Polygon(room["polygon"]).area - 20.0) / 20.0 < 0.05
    assert out["cameras_plan"].shape == (6, 2)


def test_no_floor_in_view_uses_camera_height_prior(tmp_path):
    cap_dir, _ = _capture(tmp_path, cut_floor=True)
    out = measure_photo_room(cap_dir, "r")
    assert any("floor not seen" in w for w in out["warnings"])
    assert Polygon(out["room"]["polygon"]).area > 5.0
