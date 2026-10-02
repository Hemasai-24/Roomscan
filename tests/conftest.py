from pathlib import Path
import cv2
import numpy as np
import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"
SINGLE_ROOM = SAMPLE / "single_room" / "c00a170fe1"

HEADER = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n"


def make_stray(root: Path, depth_fn, poses, w=256, h=192, fx_rgb=1600.0):
    """Write a minimal Stray Scanner capture. poses: list of (xyz, quat_xyzw)."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    s = 1920 / w
    K = np.array([[fx_rgb, 0, w * s / 2], [0, fx_rgb, h * s / 2], [0, 0, 1]])
    np.savetxt(root / "camera_matrix.csv", K, delimiter=", ", fmt="%.4f")
    rows = []
    for i, (t, q) in enumerate(poses):
        depth_m = depth_fn(i)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), (depth_m * 1000).astype(np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full((h, w), 2, np.uint8))
        rows.append(f"{i * 0.0167:.6f}, {i:06d}, {t[0]}, {t[1]}, {t[2]}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, "
                    f"{fx_rgb}, {fx_rgb}, {K[0, 2]}, {K[1, 2]}, , \n")
    (root / "odometry.csv").write_text(HEADER + "".join(rows))
    return root


@pytest.fixture
def stray_factory(tmp_path):
    def _make(depth_fn, poses, **kw):
        return make_stray(tmp_path / "cap", depth_fn, poses, **kw)
    return _make


def need_sample():
    if not SINGLE_ROOM.exists():
        pytest.skip("sample_data not present")
