"""Video tier, step V4: find "up", level the reconstruction, and write it in the Stray Scanner layout
(depth/ + confidence/ PNGs, odometry.csv, camera_matrix.csv) so the LiDAR back end reads it unchanged."""
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

NOMINAL_RGB_WIDTH = 1920      # load_stray scales intrinsics by depth_width / 1920
HEADER = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n"


def estimate_up(T_wc):
    """People hold a phone roughly upright: the mean image-down direction (camera +y) is gravity."""
    down = np.mean([T[:3, :3] @ np.array([0.0, 1.0, 0.0]) for T in T_wc], axis=0)
    return -down / np.linalg.norm(down)


def refine_up(points, normals, up, max_deg=25):
    """Snap a rough up to the mean of surface normals within max_deg of it (floor/ceiling)."""
    n = np.asarray(normals, float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    c = n @ up
    near = np.abs(c) > np.cos(np.radians(max_deg))
    if near.sum() < 50:
        return up
    m = (n[near] * np.sign(c[near])[:, None]).mean(0)
    return m / np.linalg.norm(m)


def _rotation_to_y(up):
    up = up / np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, y)
    s, c = np.linalg.norm(axis), float(up @ y)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()


def level(T_wc, up):
    """Rotate the whole world so that `up` becomes +Y."""
    G = np.eye(4)
    G[:3, :3] = _rotation_to_y(up)
    return [G @ T for T in T_wc]


def write_capture(root, depths_m, confs, T_wc, K, fps=2.0):
    root = Path(root)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)
    h, w = depths_m[0].shape
    s = NOMINAL_RGB_WIDTH / w
    np.savetxt(root / "camera_matrix.csv", np.diag([s, s, 1.0]) @ K, delimiter=", ", fmt="%.6f")
    rows = []
    for i, (d, c, T) in enumerate(zip(depths_m, confs, T_wc)):
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.clip(d * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), c.astype(np.uint8))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        t = T[:3, 3]
        rows.append(f"{i / fps:.6f}, {i:06d}, {t[0]:.9f}, {t[1]:.9f}, {t[2]:.9f}, {q[0]:.12f}, {q[1]:.12f}, "
                    f"{q[2]:.12f}, {q[3]:.12f}, {K[0, 0] * s:.6f}, {K[1, 1] * s:.6f}, {K[0, 2] * s:.6f}, "
                    f"{K[1, 2] * s:.6f}, , \n")
    (root / "odometry.csv").write_text(HEADER + "".join(rows))
    return root
