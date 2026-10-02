"""Step 1: load a Stray Scanner export (rgb.mp4, depth/, confidence/, odometry.csv)."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Frame:
    index: int
    timestamp: float
    T_wc: np.ndarray  # camera->world; OpenCV camera axes; world +Y up
    K: np.ndarray     # intrinsics at depth resolution


@dataclass
class StrayCapture:
    root: Path
    frames: list
    depth_size: tuple  # (w, h)

    def load_depth(self, index: int, min_conf: int = 2) -> np.ndarray:
        d = cv2.imread(str(self.root / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(self.root / "confidence" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        out = d.astype(np.float32) / 1000.0
        if c is not None:          # older exports / partial copies may lack confidence maps
            out[c < min_conf] = 0.0
        return out


def is_stray(root) -> bool:
    root = Path(root)
    return (root / "odometry.csv").exists() and (root / "depth").is_dir()


def load_stray(root, rgb_width: int = 1920) -> StrayCapture:
    root = Path(root)
    od = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1, usecols=range(13))
    od = np.atleast_2d(od)
    first = cv2.imread(str(root / "depth" / f"{int(od[0, 1]):06d}.png"), cv2.IMREAD_UNCHANGED)
    h, w = first.shape
    s = w / rgb_width
    frames = []
    for row in od:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T[:3, 3] = row[2:5]
        K = np.array([[row[9] * s, 0, row[11] * s], [0, row[10] * s, row[12] * s], [0, 0, 1.0]])
        frames.append(Frame(int(row[1]), float(row[0]), T, K))
    return StrayCapture(root, frames, (w, h))
