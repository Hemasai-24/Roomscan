import numpy as np

from roomscan.find_surfaces import Plane
from roomscan.photo_room import wall_box


def _wall(axis, offset, lo, hi, n):
    """Vertical wall plane: axis 'u' -> x = offset spanning z in [lo, hi]; 'v' -> z = offset."""
    t = np.linspace(lo, hi, n)
    y = np.linspace(-1.4, 1.0, n)
    if axis == "u":
        pts = np.c_[np.full(n, offset), y, t]
        nrm = np.array([1.0, 0, 0])
    else:
        pts = np.c_[t, y, np.full(n, offset)]
        nrm = np.array([0, 0, 1.0])
    return Plane(nrm, -float(nrm @ pts[0]), pts, 0.005, "wall")


def test_box_between_best_supported_walls_on_each_side():
    walls = [_wall("u", 0.0, 0, 3, 3000), _wall("u", 4.0, 0, 3, 3000), _wall("v", 0.0, 0, 4, 4000),
             _wall("v", 3.0, 0, 4, 4000),
             _wall("u", 1.0, 0.0, 0.6, 300),        # wardrobe side: weakly supported, must not win
             _wall("u", 6.5, 1.0, 2.0, 200)]        # next room's wall seen through a door
    cams = np.array([[2.0, 0.0, 1.5], [2.5, 0.0, 1.2]])
    u0, v0, u1, v1 = wall_box(walls, cams, angle=0.0)
    np.testing.assert_allclose([u0, v0, u1, v1], [0.0, 0.0, 4.0, 3.0], atol=0.01)


def test_box_is_none_when_a_side_has_no_wall():
    walls = [_wall("u", 0.0, 0, 3, 3000), _wall("v", 0.0, 0, 4, 4000), _wall("v", 3.0, 0, 4, 4000)]
    assert wall_box(walls, np.array([[2.0, 0.0, 1.5]]), angle=0.0) is None
