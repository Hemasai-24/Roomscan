import numpy as np

from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.photo_room import clip_mask_to_walls
from roomscan.point_cloud import estimate_normals
from roomscan.room_outline import manhattan_angle, to_plan
from roomscan.split_rooms import split_rooms
from tests.synthetic_rooms import room_points, two_rooms_with_door


def _setup(p):
    c = classify_planes(extract_planes(p, estimate_normals(p), min_inliers=500))
    a = manhattan_angle(c["walls"])
    masks, g = split_rooms(c, a)
    return c, a, np.logical_or.reduce(masks), g        # the leak: every floor region the cameras saw


def test_floor_seen_through_the_door_is_cut_off():
    # cameras stand in room A (x 0..4, z 0..3); room B (x 4..7) is only seen through the door
    c, a, mask, g = _setup(two_rooms_with_door())
    cams = to_plan(np.array([[1.5, 1.4, 1.2], [2.5, 1.4, 1.8], [2.0, 1.4, 1.0]]), a)
    clipped, _ = clip_mask_to_walls(mask, g, c, a, cams)
    area = clipped.sum() * g.cell ** 2
    assert mask.sum() * g.cell ** 2 > 18                 # before: both rooms (12 + 9)
    assert abs(area - 12.0) < 1.2                        # after: room A only


def test_single_room_is_not_shrunk():
    c, a, mask, g = _setup(room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.5))
    cams = to_plan(np.array([[1.0, 1.4, 1.0], [3.0, 1.4, 2.0]]), a)
    clipped, _ = clip_mask_to_walls(mask, g, c, a, cams)
    assert abs(clipped.sum() - mask.sum()) * g.cell ** 2 < 0.3
