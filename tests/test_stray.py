import cv2
import numpy as np
import pytest
from fpp.ingest.stray import is_stray, load_stray
from tests.conftest import SINGLE_ROOM, need_sample

IDENT = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def test_loads_poses_and_scales_intrinsics(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 2.0), [IDENT, ((1.0, 2.0, 3.0), (0, 0, 0, 1))])
    cap = load_stray(root)
    assert is_stray(root)
    assert len(cap.frames) == 2
    np.testing.assert_allclose(cap.frames[1].T_wc[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(cap.frames[0].K[0, 0], 1600 * 256 / 1920, rtol=1e-6)
    np.testing.assert_allclose(cap.frames[0].K[0, 2], 128.0, rtol=1e-6)
    assert cap.depth_size == (256, 192)


def test_depth_is_metres_and_low_confidence_masked(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 1.5), [IDENT])
    conf = np.full((192, 256), 2, np.uint8)
    conf[:10] = 1
    cv2.imwrite(str(root / "confidence" / "000000.png"), conf)
    d = load_stray(root).load_depth(0)
    assert d.dtype == np.float32
    assert np.all(d[:10] == 0)
    np.testing.assert_allclose(d[10:], 1.5)


def test_not_stray(tmp_path):
    assert not is_stray(tmp_path)


@pytest.mark.sample
def test_sample_single_room():
    need_sample()
    cap = load_stray(SINGLE_ROOM)
    assert len(cap.frames) == 1715
    assert cap.depth_size == (256, 192)
