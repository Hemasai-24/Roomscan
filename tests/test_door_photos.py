import numpy as np

from roomscan.door_photos import see_through_width


def test_see_through_width_on_a_flat_wall_with_a_doorway():
    # wall 2 m in front of the camera, a 0.8 m doorway in it (columns where depth jumps to 5 m behind)
    H, W, f = 400, 300, 300.0
    cx, cy = W / 2, H / 2
    D = np.full((H, W), 2.0)
    half = 0.4 / 2.0 * f                      # 0.4 m at 2 m depth -> pixels
    D[50:390, int(cx - half):int(cx + half)] = 5.0
    box = (cx - half - 30, 40, cx + half + 30, 395)   # detector box includes the frame around the opening
    w = see_through_width(D, box, f, cx, cy)
    assert abs(w - 0.8) < 0.02


from tests.test_stitch_rooms import make_room          # noqa: E402
from roomscan.door_photos import add_measured_door    # noqa: E402

M = {"width": 0.85, "lo": 0.68, "hi": 1.02, "score": 0.7}


def test_measured_door_goes_on_the_wall_facing_the_hall():
    hall = make_room("H", [(0, 0), (4, 0), (4, 3), (0, 3)])
    room = make_room("B", [(4.15, 0), (6, 0), (6, 2), (4.15, 2)])     # west wall (w0) faces the hall
    add_measured_door(room, M, parent=hall)
    (d,) = room["openings"]
    assert d["type"] == "door" and d["wall_id"] == "B_w0" and d["width"]["value"] == 0.85
    assert d["source"] == "door_photo"


def test_measured_door_replaces_the_width_of_a_detected_door():
    room = make_room("B", [(0, 0), (3, 0), (3, 3), (0, 3)], doors=[(1, 0.5, 0.6)])
    add_measured_door(room, M, parent=None)
    assert len(room["openings"]) == 1 and room["openings"][0]["width"]["value"] == 0.85


def test_measured_door_ignores_a_side_wall_that_only_touches_the_hall_at_one_end():
    hall = make_room("H", [(0, 0), (4, 0), (4, 3), (0, 3)])
    # room above the hall: its bottom wall (w1) faces the hall; its west wall (w0) only touches it at one end
    room = make_room("K", [(0, 3.15), (2, 3.15), (2, 5), (0, 5)])
    add_measured_door(room, M, parent=hall)
    assert room["openings"][0]["wall_id"] == "K_w1"
