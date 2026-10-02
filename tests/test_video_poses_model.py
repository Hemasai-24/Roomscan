"""Real-model checks on the sample video (skipped unless weights and sample data are present)."""
from pathlib import Path

import numpy as np
import pytest

from roomscan.load_capture import load_stray
from roomscan.video_frames import pick_sharpest
from roomscan.video_poses import rot_angle_deg

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/hemasai/projects/floorplan-pipeline")
SINGLE = next((p / "sample_data/single_room/c00a170fe1" for p in (ROOT, MAIN)
               if (p / "sample_data/single_room/c00a170fe1").exists()), None)
# Stray stores frames sideways; upright = 90 deg clockwise; camera axes: x_up = -y_sensor, y_up = x_sensor
R_SENSOR_TO_UPRIGHT = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])

pytestmark = pytest.mark.model


def _need():
    if SINGLE is None or not (ROOT / "weights" / "vggt-1b" / "model.pt").exists():
        pytest.skip("needs sample data and weights")


def upright_truth(cap, frame_idx):
    out = []
    for i in frame_idx:
        T = cap.frames[i].T_wc
        U = np.eye(4)
        U[:3, :3] = T[:3, :3] @ R_SENSOR_TO_UPRIGHT.T
        U[:3, 3] = T[:3, 3]
        out.append(U)
    return out


def rotation_errors(est, truth):
    M = sum(g[:3, :3] @ e[:3, :3].T for e, g in zip(est, truth))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return [rot_angle_deg((R @ e[:3, :3]).T @ g[:3, :3]) for e, g in zip(est, truth)]


def test_vggt_20_frames_rotation_error_small():
    _need()
    from roomscan.video_poses import VGGTRunner
    idx, imgs = pick_sharpest(SINGLE / "rgb.mp4", per_sec=2, size=(392, 518), rotate=90)
    out = VGGTRunner(ROOT).run(imgs[:20])
    err = rotation_errors(out["T"], upright_truth(load_stray(SINGLE), idx[:20]))
    assert np.median(err) < 5
