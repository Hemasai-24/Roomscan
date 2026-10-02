import numpy as np
import pytest

from roomscan.find_surfaces import CaptureError
from roomscan.pipeline import detect_tier, check_video_length
from tests.test_video_frames import _make_video


def test_stray_folder_is_lidar(stray_factory):
    root = stray_factory(lambda i: np.full((192, 256), 2.0), [((0, 0, 0), (0, 0, 0, 1))])
    assert detect_tier(root) == ("lidar", root)


def test_video_file_is_video(tmp_path):
    p = tmp_path / "walk.mp4"
    p.write_bytes(b"x")
    assert detect_tier(p) == ("video", p)


def test_folder_with_only_a_video_is_video(tmp_path):
    (tmp_path / "walk.MOV").write_bytes(b"x")
    assert detect_tier(tmp_path) == ("video", tmp_path / "walk.MOV")


def test_empty_folder_explains_itself(tmp_path):
    with pytest.raises(SystemExit, match="no capture found"):
        detect_tier(tmp_path)


def test_too_short_video_is_a_clear_error(tmp_path):
    p = tmp_path / "short.mp4"
    _make_video(p, n=30, fps=30)                  # 1 second
    with pytest.raises(CaptureError, match="too short"):
        check_video_length(p)
