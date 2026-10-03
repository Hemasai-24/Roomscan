import numpy as np
import pytest

from roomscan import video_scale as vs


def test_default_is_depth_anything_with_its_calibrated_bias(monkeypatch):
    monkeypatch.delenv("ROOMSCAN_METRIC_DEPTH", raising=False)
    monkeypatch.setitem(vs.METRIC_MODELS, "depth_anything", (lambda root: "DA", vs.KEY))
    model, bias = vs.metric_model()
    assert model == "DA" and bias == pytest.approx(vs.load_bias())


def test_unidepth_is_uncalibrated(monkeypatch):
    monkeypatch.setenv("ROOMSCAN_METRIC_DEPTH", "unidepth")
    monkeypatch.setitem(vs.METRIC_MODELS, "unidepth", (lambda root: "UD", None))
    model, bias = vs.metric_model()
    assert model == "UD" and bias == 1.0


def test_unknown_choice_is_rejected(monkeypatch):
    monkeypatch.setenv("ROOMSCAN_METRIC_DEPTH", "nope")
    with pytest.raises(ValueError):
        vs.metric_model()


def test_intrinsics_from_focal_centre_the_principal_point():
    K = vs.intrinsics(400.0, (518, 392))
    np.testing.assert_allclose(K, [[400, 0, 196], [0, 400, 259], [0, 0, 1]])
