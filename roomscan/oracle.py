"""Experiment-only helpers: substitute ground-truth-like LiDAR quantities ("oracles") into the video and
photo tiers, to measure how much of their error comes from scale, from camera poses, or from the room-shape
method. Not used by the default pipeline."""
import re

import cv2
import numpy as np

# camera of the upright (90 deg clockwise) image: x' = -y, y' = x, z' = z
_R_CW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def upright_frame(depth, K, T_wc):
    """Stray frames are landscape; the photo/video tiers see the image turned 90 deg clockwise.
    Return (depth, K, T_wc) of that upright camera, describing the same 3D points."""
    h = depth.shape[0]
    d2 = np.ascontiguousarray(np.rot90(depth, k=-1))
    K2 = np.array([[K[1, 1], 0.0, h - 1 - K[1, 2]], [0.0, K[0, 0], K[0, 2]], [0.0, 0.0, 1.0]])
    T2 = T_wc.copy()
    T2[:3, :3] = T_wc[:3, :3] @ _R_CW.T
    return d2, K2, T2


def oracle_scale(model_depth, model_conf, lidar_depth):
    """True metres-per-model-unit for one frame: median LiDAR/model depth over confident pixels."""
    h, w = lidar_depth.shape
    m = cv2.resize(np.asarray(model_depth, np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    c = cv2.resize(np.asarray(model_conf, np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    ok = (lidar_depth > 0.1) & (m > 1e-6) & (c >= np.median(c))
    return float(np.median(lidar_depth[ok] / m[ok]))


def frame_index(name):
    return int(re.search(r"(\d+)", str(name)).group(1))
