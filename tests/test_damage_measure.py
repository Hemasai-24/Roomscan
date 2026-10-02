import numpy as np
import pytest
from roomscan.damage_measure import measure_damage

K = np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])
W, H = 256, 192
WALL_A = {"surface_id": "room_0_w0", "room_id": "room_0", "kind": "wall",
          "normal": np.array([0.0, 0.0, -1.0]), "d": 2.0, "floor_h": -1.4}      # plane z = 2
WALL_B = {"surface_id": "room_0_w1", "room_id": "room_0", "kind": "wall",
          "normal": np.array([-1.0, 0.0, 0.0]), "d": 0.5, "floor_h": -1.4}      # plane x = 0.5


def _mask(u0, u1, v0, v1):
    m = np.zeros((H, W), bool)
    m[v0:v1, u0:u1] = True
    return m


def _view(depth, mask, cls="water_stain"):
    return {"depth": depth, "K": K, "T_wc": np.eye(4),
            "detections": [{"class": cls, "score": 0.7, "mask": mask}]}


def test_square_on_wall_has_metric_size():
    # 30 x 20 px at z = 2 m, f = 200 px  ->  0.30 m x 0.20 m, area 0.06 m2
    depth = np.full((H, W), 2.0)
    d = measure_damage([_view(depth, _mask(113, 143, 96, 116))], [WALL_A], tier="lidar")
    assert len(d) == 1 and d[0]["surface_id"] == "room_0_w0" and d[0]["class"] == "water_stain"
    assert d[0]["area"]["value"] == pytest.approx(0.06, rel=0.1)
    assert d[0]["width"]["value"] == pytest.approx(0.30, rel=0.1)
    assert d[0]["height"]["value"] == pytest.approx(0.20, rel=0.15)
    assert d[0]["area"]["lo"] < 0.06 < d[0]["area"]["hi"]
    # image rows 96..116 at z=2 -> y = 0 .. 0.2 below camera (y down); floor at y=+1.4 in this test frame
    assert d[0]["n_views"] == 1


def test_damage_on_furniture_is_dropped():
    depth = np.full((H, W), 1.0)                     # mask lands on something 1 m away, not on a wall
    assert measure_damage([_view(depth, _mask(113, 143, 96, 116))], [WALL_A], tier="lidar") == []


def test_mask_across_a_corner_splits_per_wall():
    u = np.arange(W)
    dx = (u - K[0, 2]) / K[0, 0]
    z = np.where(dx > 0.25, 0.5 / np.maximum(dx, 1e-6), 2.0)   # side wall x=0.5 where the ray reaches it first
    depth = np.tile(z, (H, 1))
    d = measure_damage([_view(depth, _mask(100, 200, 90, 110))], [WALL_A, WALL_B], tier="lidar")
    assert sorted(x["surface_id"] for x in d) == ["room_0_w0", "room_0_w1"]


def test_same_damage_in_two_views_merges():
    depth = np.full((H, W), 2.0)
    v = _view(depth, _mask(113, 143, 96, 116))
    d = measure_damage([v, v], [WALL_A], tier="lidar")
    assert len(d) == 1 and d[0]["n_views"] == 2


def test_photo_tier_ranges_wider_than_lidar():
    depth = np.full((H, W), 2.0)
    v = [_view(depth, _mask(113, 143, 96, 116))]
    a = measure_damage(v, [WALL_A], tier="lidar")[0]["area"]
    b = measure_damage(v, [WALL_A], tier="photo")[0]["area"]
    assert (b["hi"] - b["lo"]) > (a["hi"] - a["lo"])
