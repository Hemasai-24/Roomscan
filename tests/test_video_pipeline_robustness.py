import numpy as np
import pytest

from roomscan import video_pipeline as vp
from roomscan.find_surfaces import CaptureError


def _fake(monkeypatch, fail_segments):
    monkeypatch.setattr(vp, "check_video_length", lambda v: (100, 30.0))
    monkeypatch.setattr(vp, "model_size", lambda v, r=0: (392, 518))
    monkeypatch.setattr(vp, "pick_sharpest", lambda *a, **k: (list(range(40)), np.zeros((40, 518, 392, 3), np.uint8)))
    monkeypatch.setattr(vp, "VGGTRunner", lambda root: None)
    monkeypatch.setattr(vp, "metric_model", lambda root: (None, 1.4))
    seg = np.array([0] * 25 + [1] * 15)
    monkeypatch.setattr(vp, "run_chunks", lambda runner, imgs: {"seg": seg, "segments": 2})

    def seg_capture(cand, merged, keep, imgs, m, b):
        if int(merged["seg"][keep[0]]) in fail_segments:
            raise CaptureError("no walls found")
        return 0.01, {"metric_scale": 1.0}
    monkeypatch.setattr(vp, "_segment_capture", seg_capture)


def test_failing_minor_segment_is_skipped(monkeypatch, tmp_path):
    _fake(monkeypatch, fail_segments={1})
    cap_dir, info, warnings = vp.video_to_capture("v.mp4", tmp_path)
    assert info["segment_used"] == 0
    assert any("segment 1" in w for w in warnings)


def test_all_segments_failing_is_a_capture_error(monkeypatch, tmp_path):
    _fake(monkeypatch, fail_segments={0, 1})
    with pytest.raises(CaptureError):
        vp.video_to_capture("v.mp4", tmp_path)
