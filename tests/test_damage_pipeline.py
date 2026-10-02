import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from roomscan.damage_pipeline import (annotate, damage_for_capture, filter_damage, finish_plan, room_surface_list,
                                      wall_plane)
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.load_capture import load_stray
from roomscan.point_cloud import estimate_normals
from roomscan.room_outline import footprint
from tests.synthetic_rooms import render_box_depth, room_points

LO, HI = np.array([0.0, -1.4, 0.0]), np.array([5.0, 1.2, 4.0])


def _box_room():
    p = room_points([(0, 0), (5, 0), (5, 4), (0, 4)], 2.6)
    p[:, 1] -= 1.4                                   # floor at y = -1.4, ceiling at 1.2
    c = classify_planes(extract_planes(p, estimate_normals(p), min_inliers=500))
    verts, edges, angle = footprint(c)
    return c, verts, edges, angle


def test_wall_planes_pass_through_their_wall_points():
    c, verts, edges, angle = _box_room()
    for e in edges:
        n, d = wall_plane(e, angle)
        dist = np.abs(np.concatenate([w.inliers for w in c["walls"]]) @ n + d)
        assert np.sum(dist < 0.03) > 2000             # many wall points lie on each edge's plane


def test_surface_list_ids_match_room_record_and_include_floor_and_ceiling():
    c, verts, edges, angle = _box_room()
    s = room_surface_list("room_0", c, edges, angle, verts)
    ids = [x["surface_id"] for x in s]
    assert ids[:len(edges)] == [f"room_0_w{i}" for i in range(len(edges))]
    assert "room_0_floor" in ids and "room_0_ceiling" in ids
    assert all(abs(x["floor_h"] + 1.4) < 0.02 for x in s)


def _dmg(cls, sid, n_views=2, center=(1.0, 0.0, 4.0), width=0.3):
    M = lambda v: {"value": v, "lo": v - 0.01, "hi": v + 0.01, "unit": "m"}
    return {"id": "x", "class": cls, "surface_id": sid, "room_id": "room_0", "area": dict(M(0.06), unit="m2"),
            "width": M(width), "height": M(0.2), "bottom_above_floor": M(1.0), "center": list(center),
            "n_views": n_views, "score": 0.5, "near_opening": False}


def test_filter_needs_two_views_for_lidar_and_video_but_not_photo():
    d = [_dmg("water_stain", "room_0_w0", n_views=1)]
    assert filter_damage(d, "lidar") == [] and filter_damage(d, "video") == []
    kept = filter_damage(d, "photo")
    assert len(kept) == 1 and kept[0]["single_view"] is True


def test_filter_class_per_surface():
    d = [_dmg("hole", "room_0_ceiling"), _dmg("crack", "room_0_floor"), _dmg("water_stain", "room_0_ceiling"),
         _dmg("crack", "room_0_w1")]
    assert sorted((x["class"], x["surface_id"]) for x in filter_damage(d, "lidar")) == \
        [("crack", "room_0_w1"), ("water_stain", "room_0_ceiling")]


def test_annotate_offset_along_wall_and_near_opening():
    room = {"id": "room_0", "walls": [{"id": "room_0_w0", "start": [0.0, 4.0], "end": [5.0, 4.0]}],
            "openings": [{"id": "room_0_o0", "wall_id": "room_0_w0", "type": "door", "offset_along_wall": 1.5,
                          "width": {"value": 0.9}}]}
    near = _dmg("crack", "room_0_w0", center=(1.25, 0.0, 4.0), width=0.2)     # right edge 1.35, door at 1.5
    far = _dmg("crack", "room_0_w0", center=(4.0, 0.0, 4.0), width=0.2)
    annotate([near, far], {"room_0": room}, {"room_0": 0.0})
    assert abs(near["offset_along_wall"] - 1.25) < 1e-6 and near["near_opening"] is True
    assert far["near_opening"] is False


def test_finish_plan_fills_flags_scope_and_ids():
    M = lambda v: {"value": v, "lo": v - 0.01, "hi": v + 0.01, "unit": "m"}
    room = {"id": "room_0", "walls": [{"id": "room_0_w0", "length": M(4.0)}], "openings": [],
            "ceiling_height": M(2.5), "floor_area": dict(M(12.0), unit="m2")}
    plan = {"rooms": [room], "adjacency": [], "damage": [], "concealed_damage_flags": [], "scope_items": []}
    finish_plan(plan, [_dmg("water_stain", "room_0_ceiling"), _dmg("water_stain", "room_0_w0")])
    assert [d["id"] for d in plan["damage"]] == ["dmg_0", "dmg_1"]
    assert [f["rule_id"] for f in plan["concealed_damage_flags"]] == ["R1"]
    assert {i["action"] for i in plan["scope_items"]} == {"stain_block_and_repaint", "inspect"}


def _camera_capture(root, n=2):
    """Camera at the room centre looking at the far wall z = 4 (OpenCV axes: x right, y down, z forward)."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    (root / "rgb").mkdir()
    R = np.diag([-1.0, -1.0, 1.0])                   # camera y down = world -y, camera z = world +z
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [2.5, 0.0, 2.0]
    w, h, f = 64, 48, 1600 * 64 / 1920
    K = np.array([[f, 0, 32], [0, f, 24], [0, 0, 1]])
    depth = render_box_depth(K, T, LO, HI, w, h)
    q = Rotation.from_matrix(R).as_quat()
    rows = []
    for i in range(n):
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), (depth * 1000).astype(np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full((h, w), 2, np.uint8))
        cv2.imwrite(str(root / "rgb" / f"{i:06d}.jpg"), np.full((h, w, 3), 200, np.uint8))
        rows.append(f"{i:.6f}, {i:06d}, 2.5, 0.0, 2.0, {q[0]}, {q[1]}, {q[2]}, {q[3]}, 1600, 1600, 960, 720, , \n")
    from tests.conftest import HEADER
    (root / "odometry.csv").write_text(HEADER + "".join(rows))
    return root


def test_damage_for_capture_with_fake_detector(tmp_path):
    c, verts, edges, angle = _box_room()
    surfaces = room_surface_list("room_0", c, edges, angle, verts)
    root = _camera_capture(tmp_path / "cap")

    def fake_detect(img):
        m = np.zeros(img.shape[:2], bool)
        m[10:20, 27:37] = True                       # 10 x 10 px patch at the image centre
        return [{"class": "water_stain", "score": 0.6, "mask": m, "box": (27, 10, 37, 20)}]

    d = damage_for_capture(load_stray(root), root, surfaces, "lidar", detect=fake_detect)
    assert len(d) == 1 and d[0]["n_views"] == 2
    sid = d[0]["surface_id"]
    e = edges[int(sid.rsplit("_w", 1)[1])]
    n, dd = wall_plane(e, angle)
    assert abs(np.array([2.5, 0.0, 4.0]) @ n + dd) < 0.05         # it is the far wall z = 4
    px = 2.0 / (1600 * 64 / 1920)                                  # 2 m away: one pixel = 3.75 cm
    assert abs(d[0]["area"]["value"] - (10 * px) ** 2) < 0.2 * (10 * px) ** 2
