import numpy as np

from roomscan.photo_sim import pick_door_view, pick_room_views


def test_one_view_per_direction_sector_preferring_far_looking_sharp_frames():
    yaw = np.radians([0, 5, 90, 95, 180, 185, 270, 275])
    sharp = np.array([10, 30, 30, 30, 30, 30, 1, 30], float)
    depth = np.array([5, 1, 3, 4, 2, 2, 9, 1], float)
    idx = np.arange(8)
    picked = pick_room_views(idx, yaw, sharp, depth, sectors=4)
    # sector 0: frame 0 is far-looking but blurry (below the room's median sharpness) -> frame 1
    # sector 3: frame 6 is blurry -> frame 7
    assert sorted(picked) == [1, 3, 4, 7]


def test_door_view_is_the_sharp_frame_aimed_at_the_door():
    cams = np.array([[0, 0], [0, 0], [0, 0], [3.9, 0]], float)
    fwd = np.array([[1, 0], [0.7, 0.7], [1, 0.05], [1, 0]], float)
    sharp = np.array([10, 50, 20, 50], float)
    k = pick_door_view(np.arange(4), cams, fwd, sharp, door_mid=np.array([3.0, 0.0]))
    # frame 1 is sharpest but aims 45 deg off; frame 3 is past the door; frame 2 beats frame 0 on sharpness
    assert k == 2


def test_no_door_view_when_nothing_aims_at_it():
    cams = np.zeros((2, 2))
    fwd = np.array([[-1, 0], [0, -1]], float)
    assert pick_door_view(np.arange(2), cams, fwd, np.ones(2), door_mid=np.array([3.0, 0.0])) is None
