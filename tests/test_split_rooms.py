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


def test_tiny_floor_patch_is_not_a_room():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.5)
    xs, zs = np.meshgrid(np.arange(5.0, 5.7, 0.02), np.arange(1.0, 1.7, 0.02))   # 0.5 m2 seen through a door
    patch = np.c_[xs.ravel(), np.zeros(xs.size), zs.ravel()]
    rooms, _ = _split(np.concatenate([p, patch]))
    assert len(rooms) == 1


def test_walkable_path_fills_unseen_floor():
    """Video sees little floor: with only wall points, a camera path through both rooms recovers them."""
    p = two_rooms_with_door()
    c = _classes(p)
    a = manhattan_angle(c["walls"])
    fl = c["floor"].inliers
    c["floor"].inliers = fl[(fl[:, 0] > 1.8) & (fl[:, 0] < 2.2)]     # floor seen only in a thin strip
    assert len(split_rooms(c, a)[0]) < 2
    path = np.c_[np.linspace(0.8, 6.2, 400), np.full(400, 1.45)]       # walk across both rooms, through the door
    rooms, g = split_rooms(c, a, extra_free=path, extra_radius=0.9)
    assert len(rooms) == 2
