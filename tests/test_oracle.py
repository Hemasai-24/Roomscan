import numpy as np
from scipy.spatial.transform import Rotation

from roomscan.oracle import frame_index, oracle_scale, upright_frame
from roomscan.point_cloud import backproject


def test_upright_frame_sees_the_same_world_points():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 3.0, (192, 256)).astype(np.float32)
    K = np.array([[210.0, 0, 127.0], [0, 212.0, 95.0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [10, 40, -5], degrees=True).as_matrix()
    T[:3, 3] = [0.3, -1.2, 2.0]
    d2, K2, T2 = upright_frame(depth, K, T)
    assert d2.shape == (256, 192)
    a = backproject(depth, K, T, max_depth=10)
    b = backproject(d2, K2, T2, max_depth=10)
    a = a[np.lexsort(a.T)]
    b = b[np.lexsort(b.T)]
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_oracle_scale_recovers_true_ratio_across_resolutions():
    rng = np.random.default_rng(1)
    lidar = rng.uniform(1.0, 4.0, (256, 192)).astype(np.float32)
    import cv2
    model = cv2.resize(lidar, (392, 518), interpolation=cv2.INTER_NEAREST) / 2.5
    conf = np.ones_like(model)
    assert abs(oracle_scale(model, conf, lidar) - 2.5) < 0.01


def test_frame_index_from_photo_name():
    assert frame_index("frame_000123.jpg") == 123
