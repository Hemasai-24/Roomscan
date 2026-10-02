import cv2
import numpy as np
import pytest

from roomscan.video_frames import pick_sharpest, read_frames, video_info

W, H = 160, 120


def _make_video(path, n=90, fps=30, sharp_every=15):
    """Checkerboard video; one sharp frame per `sharp_every`-frame window (at offset 7), the rest blurred."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    yy, xx = np.mgrid[0:H, 0:W]
    sharp = []
    for i in range(n):
        board = ((((xx + i) // 10) + (yy // 10)) % 2 * 255).astype(np.uint8)
        img = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        img[:30, :40] = (0, 0, 255)                     # red marker top-left, to check rotation
        if i % sharp_every == 7:
            sharp.append(i)
        else:
            img = cv2.GaussianBlur(img, (15, 15), 6)
        vw.write(img)
    vw.release()
    return sharp


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "walk.mp4"
    return p, _make_video(p)


def test_video_info(video):
    p, _ = video
    n, fps = video_info(p)
    assert n == 90
    assert abs(fps - 30) < 0.5


def test_pick_sharpest_finds_unblurred_frames(video):
    p, sharp = video
    idx, imgs = pick_sharpest(p, per_sec=2, size=(W, H))
    assert idx == sharp
    assert imgs.shape == (6, H, W, 3)


def test_rotate_90_is_clockwise(video):
    p, _ = video
    _, plain = next(read_frames(p, (W, H)))
    _, rot = next(read_frames(p, (H, W), rotate=90))
    assert rot.shape == (W, H, 3)
    # clockwise: the top-left red marker moves to the top-right
    assert rot[:20, -20:, 0].mean() > 150 and rot[:20, -20:, 2].mean() < 100
    assert plain[:20, :20, 0].mean() > 150


def test_model_size_follows_video_orientation(video, tmp_path):
    from roomscan.video_frames import model_size
    p, _ = video                                            # 160 x 120: landscape
    assert model_size(p) == (518, 392)
    assert model_size(p, rotate=90) == (392, 518)
    # phone-style portrait: landscape pixels + rotation metadata
    q = tmp_path / "portrait_meta.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(q)],
                   check=True)
    assert model_size(q) == (392, 518)
