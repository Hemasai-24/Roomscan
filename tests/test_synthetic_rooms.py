import numpy as np
from tests.synthetic_rooms import room_points


def test_box_room_extents():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.5)
    np.testing.assert_allclose(p.min(0), [0, 0, 0], atol=0.02)
    np.testing.assert_allclose(p.max(0), [4, 2.5, 3], atol=0.02)
