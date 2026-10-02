import numpy as np
from scipy.spatial.transform import Rotation
from roomscan.point_cloud import backproject, fuse_points
from roomscan.load_capture import load_stray


def K():
    return np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])


def test_backproject_identity_plane():
    pts = backproject(np.full((192, 256), 2.0, np.float32), K(), np.eye(4))
    np.testing.assert_allclose(pts[:, 2], 2.0)
    assert pts.shape == (192 * 256, 3)


def test_backproject_applies_pose_and_drops_invalid():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    T[:3, 3] = [1, 0, 0]
    d = np.full((192, 256), 2.0, np.float32)
    d[0, 0] = 0.0
    d[0, 1] = 9.0  # beyond max_depth
    pts = backproject(d, K(), T, max_depth=4.0)
    assert len(pts) == 192 * 256 - 2
    # camera z axis rotated +90 about y -> world +x; plane at x = 1 + 2
    np.testing.assert_allclose(pts[:, 0], 3.0, atol=1e-6)


def test_fuse_downsamples(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 2.0), [((0, 0, 0), (0, 0, 0, 1))] * 3)
    pts = fuse_points(load_stray(root), stride=1, voxel=0.05)
    assert 0 < len(pts) < 192 * 256
    np.testing.assert_allclose(pts[:, 2], 2.0, atol=0.03)
