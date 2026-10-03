import numpy as np
import pytest

from roomscan.visual_openings import measure_openings

K = np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])
# camera at origin looking along +z (OpenCV: y down); floor at y = +1.4 (world y up is -y here: use T to flip)
T = np.diag([1.0, -1.0, 1.0, 1.0])            # camera y-down -> world y-up; camera at height 0, floor at -1.4
WALL = {"surface_id": "r_w0", "room_id": "r", "kind": "wall", "normal": np.array([0.0, 0.0, -1.0]), "d": 2.0,
        "floor_h": -1.4, "extent": {"axis": "v", "lo": -3.0, "hi": 3.0, "angle": 0.0}}


def _px(x, y, z=2.0):
    """Image pixel of world point (x, y, z) for camera T (y flipped)."""
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * (-y) / z + K[1, 2]


def _view(label, x0, x1, y_bottom, y_top):
    u0, v_top = _px(x0, y_top)
    u1, v_bot = _px(x1, y_bottom)
    return {"K": K, "T_wc": T, "size": (256, 300),            # tall enough to contain a 2 m door at 2 m
            "detections": [{"label": label, "score": 0.6, "box": (u0, v_top, u1, v_bot)}]}


def test_door_width_height_on_a_wall():
    # door 0.8 m wide (x -0.4..0.4), from the floor (y -1.4) to 0.6 m above the camera (y +0.6): 2.0 m tall
    ops = measure_openings([_view("door", -0.4, 0.4, -1.4, 0.6)], [WALL], tier="lidar")
    assert len(ops) == 1 and ops[0]["type"] == "door" and ops[0]["wall_id"] == "r_w0"
    assert ops[0]["width"]["value"] == pytest.approx(0.8, abs=0.02)
    assert ops[0]["height"]["value"] == pytest.approx(2.0, abs=0.03)
    assert ops[0]["sill_height"]["value"] == pytest.approx(0.0, abs=0.03)


def test_window_sill_and_merge_across_views():
    v1 = _view("window", -0.6, 0.6, -0.5, 0.6)           # 1.2 m wide, sill 0.9 m above floor
    v2 = _view("window", -0.58, 0.62, -0.5, 0.6)         # same window, slightly different box
    ops = measure_openings([v1, v2], [WALL], tier="lidar")
    assert len(ops) == 1 and ops[0]["n_views"] == 2 and ops[0]["type"] == "window"
    assert ops[0]["width"]["value"] == pytest.approx(1.2, abs=0.03)
    assert ops[0]["sill_height"]["value"] == pytest.approx(0.9, abs=0.03)


def test_detection_not_on_any_wall_is_dropped():
    far_wall = dict(WALL, extent={"axis": "v", "lo": 5.0, "hi": 6.0, "angle": 0.0})
    assert measure_openings([_view("door", -0.4, 0.4, -1.4, 0.6)], [far_wall], tier="lidar") == []


def test_photo_tier_ranges_are_wider():
    v = [_view("door", -0.4, 0.4, -1.4, 0.6)]
    a = measure_openings(v, [WALL], tier="lidar")[0]["width"]
    b = measure_openings(v, [WALL], tier="photo")[0]["width"]
    assert (b["hi"] - b["lo"]) > (a["hi"] - a["lo"])


def test_box_in_a_turned_image_maps_back_to_the_original_pixels():
    from roomscan.damage_pipeline import box_to_original
    img = np.zeros((40, 60))                              # H 40, W 60
    for k in range(4):
        rot = np.rot90(img, k)
        # mark the pixel the original (x=10, y=5) lands on after the turn, find it, map its box back
        coords = np.rot90(np.stack(np.meshgrid(np.arange(60), np.arange(40)), -1), k)
        yy, xx = np.argwhere((coords[..., 0] == 10) & (coords[..., 1] == 5))[0]
        b = box_to_original((xx, yy, xx + 1, yy + 1), img.shape, k)
        assert abs(b[0] - 10) <= 1 and abs(b[1] - 5) <= 1


def _M(v, h=0.02):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": "m"}


def test_photo_openings_replace_carved_ones_at_the_right_offset():
    from roomscan.damage_pipeline import place_openings
    room = {"id": "r", "walls": [{"id": "r_w0", "start": [4.0, 0.0], "end": [0.0, 0.0]}],
            "openings": [{"id": "r_o0", "wall_id": "r_w0", "type": "window", "offset_along_wall": 0.1, "width": _M(0.6)}]}
    found = [{"room_id": "r", "wall_id": "r_w0", "type": "door", "centre_along": 1.0, "width": _M(0.8),
              "height": _M(2.0), "sill_height": _M(0.0), "n_views": 2}]
    place_openings({"r": room}, found, "photo")
    assert len(room["openings"]) == 1 and room["openings"][0]["type"] == "door"
    assert abs(room["openings"][0]["offset_along_wall"] - 2.6) < 1e-6      # wall runs 4 -> 0; door spans 1.4..0.6


def test_lidar_keeps_carved_openings_and_adds_only_new_ones():
    from roomscan.damage_pipeline import place_openings
    room = {"id": "r", "walls": [{"id": "r_w0", "start": [0.0, 0.0], "end": [4.0, 0.0]}],
            "openings": [{"id": "r_o0", "wall_id": "r_w0", "type": "door", "offset_along_wall": 0.6, "width": _M(0.8)}]}
    same = {"room_id": "r", "wall_id": "r_w0", "type": "door", "centre_along": 1.0, "width": _M(0.8),
            "height": _M(2.0), "sill_height": _M(0.0), "n_views": 1}
    closed = dict(same, centre_along=3.0)
    place_openings({"r": room}, [same, closed], "lidar")
    assert len(room["openings"]) == 2 and [o["id"] for o in room["openings"]] == ["r_o0", "r_o1"]


def test_box_cut_by_the_photo_edge_is_not_measured():
    v = _view("door", -0.4, 0.4, -1.4, 0.6)
    x0, y0, x1, y1 = v["detections"][0]["box"]
    v["detections"][0]["box"] = (0.0, y0, x1, y1)                   # left edge at the image border
    assert measure_openings([v], [WALL], tier="lidar") == []


def test_implausible_sizes_are_rejected():
    tiny = _view("door", -0.1, 0.1, -1.4, -1.0)                     # 0.2 x 0.4 m "door" (a cupboard knob area)
    tall_window = _view("window", 0.5, 0.6, -1.4, 0.6)              # 0.1 m wide window
    assert measure_openings([tiny, tall_window], [WALL], tier="lidar") == []


def test_photo_room_with_no_image_openings_drops_the_carved_phantoms():
    from roomscan.damage_pipeline import place_openings
    room = {"id": "r", "walls": [{"id": "r_w0", "start": [0.0, 0.0], "end": [4.0, 0.0]}],
            "openings": [{"id": "r_o0", "wall_id": "r_w0", "type": "window", "offset_along_wall": 0.1, "width": _M(0.6)}]}
    place_openings({"r": room}, [], "photo")
    assert room["openings"] == []


def test_door_cut_at_top_and_bottom_still_gives_its_width():
    v = _view("door", -0.4, 0.4, -1.4, 0.6)
    x0, y0, x1, y1 = v["detections"][0]["box"]
    v["detections"][0]["box"] = (x0, 0.0, x1, 299.0)                # top and bottom outside the frame
    ops = measure_openings([v], [WALL], tier="lidar")
    assert len(ops) == 1 and ops[0]["width"]["value"] == pytest.approx(0.8, abs=0.02)
    assert "height" not in ops[0]                                   # unknown, not invented


def test_edges_are_read_in_the_upright_image_for_turned_frames():
    """A sideways video frame: the detector sees the upright image; edge points must map back so that the
    door's left/right edges (its width) stay its width."""
    from roomscan.damage_pipeline import upright_edges
    orig_shape = (100, 200)                                  # sideways frame, H 100 x W 200
    k = 1                                                    # one CCW turn makes it upright (200 x 100)
    box_upright = (30.0, 50.0, 70.0, 150.0)                  # 40 px wide, 100 px tall in the upright image
    e = upright_edges(box_upright, orig_shape, k)
    # in the original sideways frame the door's width runs along the original y axis
    (lx, ly), (rx, ry) = e["l"], e["r"]
    (tx, ty), (bx, by) = e["t"], e["b"]
    assert abs(abs(ry - ly) - 40) <= 2 and abs(rx - lx) <= 2
    assert abs(abs(bx - tx) - 100) <= 2 and abs(by - ty) <= 2
