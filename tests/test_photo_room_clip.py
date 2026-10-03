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


def test_wall_between_the_cameras_is_not_a_bound():
    # a 5 m room with a tall false "wall" copy at x = 1.0 (inconsistent photo depth); photos taken from both
    # ends of the room - a room's own wall cannot lie between two of its camera positions
    p = room_points([(0, 0), (5, 0), (5, 3), (0, 3)], 2.5)
    zs, ys = np.meshgrid(np.arange(0.0, 3.0, 0.02), np.arange(0.0, 2.3, 0.02))
    ghost = np.c_[np.full(zs.size, 1.0), ys.ravel(), zs.ravel()]
    c, a, mask, g = _setup(np.concatenate([p, ghost]))
    cams = to_plan(np.array([[0.4, 1.4, 0.4], [0.4, 1.4, 2.6], [4.6, 1.4, 0.4], [4.6, 1.4, 2.6], [2.5, 1.4, 1.5]]), a)
    clipped, _ = clip_mask_to_walls(mask, g, c, a, cams)
    assert clipped.sum() * g.cell ** 2 > 12.0            # whole 15 m2 room kept (minus wall band), not cut at x = 1


def test_one_doorway_photo_taken_from_the_next_room_is_ignored():
    # 4 photos inside room A + 1 doorway photo taken from inside room B (x = 4.6): A is still cut at x = 4
    c, a, mask, g = _setup(two_rooms_with_door())
    cams = to_plan(np.array([[0.5, 1.4, 0.5], [3.5, 1.4, 0.5], [0.5, 1.4, 2.5], [3.5, 1.4, 2.5],
                             [4.6, 1.4, 1.45]]), a)
    clipped, _ = clip_mask_to_walls(mask, g, c, a, cams)
    assert abs(clipped.sum() * g.cell ** 2 - 12.0) < 1.2
