import numpy as np
from scipy.spatial.transform import Rotation

from roomscan.load_capture import load_stray
from roomscan.video_capture import estimate_up, level, refine_up, write_capture


def _upright_cams(n=40, roll_deg=10, seed=0):
    """Cameras looking horizontally around a room, image-down = world -Y, small random roll/pitch."""
    rng = np.random.default_rng(seed)
    out = []
    base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])   # cam x=world x, cam y(down)=-Y, cam z=-Z
    for i in range(n):
        yaw = Rotation.from_euler("y", rng.uniform(0, 360), degrees=True).as_matrix()
        wob = Rotation.from_euler("xz", rng.uniform(-roll_deg, roll_deg, 2), degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = yaw @ base @ wob
        T[:3, 3] = rng.uniform(-2, 2, 3)
        out.append(T)
    return out


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_up_from_handheld_cameras():
    assert _angle(estimate_up(_upright_cams()), np.array([0, 1.0, 0])) < 3


def test_level_undoes_world_tilt():
    tilt = np.eye(4)
    tilt[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    cams = [tilt @ T for T in _upright_cams(roll_deg=0)]
    up = estimate_up(cams)
    lev = level(cams, up)
    assert _angle(estimate_up(lev), np.array([0, 1.0, 0])) < 1


def test_refine_up_snaps_to_floor_normals():
    rng = np.random.default_rng(1)
    floor_n = np.tile([0, 1.0, 0], (500, 1)) + rng.normal(0, 0.02, (500, 3))
    wall_n = np.tile([1.0, 0, 0], (500, 1)) + rng.normal(0, 0.02, (500, 3))
    normals = np.concatenate([floor_n, -floor_n[:200], wall_n])
    rough = Rotation.from_euler("z", 15, degrees=True).as_matrix() @ np.array([0, 1.0, 0])
    up = refine_up(None, normals, rough)
    assert _angle(up, np.array([0, 1.0, 0])) < 1


def test_write_capture_round_trips_through_stray_loader(tmp_path):
    cams = _upright_cams(n=3)
    K = np.array([[370.0, 0, 196], [0, 370.0, 259], [0, 0, 1]])
    depths = [np.full((518, 392), 1.0 + 0.5 * i, np.float32) for i in range(3)]
    confs = [np.full((518, 392), 2, np.uint8) for _ in range(3)]
    root = write_capture(tmp_path / "cap", depths, confs, cams, K)
    cap = load_stray(root)
    assert len(cap.frames) == 3
    for f, T in zip(cap.frames, cams):
        np.testing.assert_allclose(f.T_wc, T, atol=1e-6)
        np.testing.assert_allclose(f.K, K, rtol=1e-4)
    np.testing.assert_allclose(cap.load_depth(2), 2.0, atol=0.001)


def test_write_capture_keeps_colour_frames_when_given(tmp_path):
    import cv2
    cams = _upright_cams(n=2)
    K = np.array([[370.0, 0, 196], [0, 370.0, 259], [0, 0, 1]])
    depths = [np.full((518, 392), 1.0, np.float32) for _ in range(2)]
    confs = [np.full((518, 392), 2, np.uint8) for _ in range(2)]
    imgs = [np.full((518, 392, 3), (10 * i, 100, 200), np.uint8) for i in range(2)]
    root = write_capture(tmp_path / "cap", depths, confs, cams, K, images=imgs)
    got = cv2.cvtColor(cv2.imread(str(root / "rgb" / "000001.jpg")), cv2.COLOR_BGR2RGB)
    assert got.shape == (518, 392, 3) and abs(int(got[0, 0, 2]) - 200) < 5
