import numpy as np
import pytest
from roomscan.damage_detect import CLASSES, label_to_class, postprocess, upright_turns

H, W = 480, 640


def det(label, score, box, fill=None):
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = fill or box
    m[y0:y1, x0:x1] = True
    return {"label": label, "score": score, "box": box, "mask": m}


def test_label_text_maps_to_class():
    assert label_to_class("water stain") == "water_stain"
    assert label_to_class("peeling paint") == "peeling_paint"
    assert label_to_class("a crack") == "crack"
    assert label_to_class("chair") is None
    assert set(CLASSES.values()) == {"water_stain", "crack", "mold", "peeling_paint", "hole"}


def test_below_threshold_and_unknown_labels_dropped():
    out = postprocess([det("crack", 0.2, (10, 10, 60, 60)), det("chair", 0.9, (10, 10, 60, 60))],
                      (H, W), threshold=0.3)
    assert out == []


def test_tiny_and_huge_masks_dropped():
    tiny = det("water stain", 0.9, (10, 10, 13, 13))
    huge = det("water stain", 0.9, (0, 0, W, H))
    ok = det("water stain", 0.9, (100, 100, 200, 180))
    out = postprocess([tiny, huge, ok], (H, W), threshold=0.3)
    assert [o["box"] for o in out] == [(100, 100, 200, 180)]
    assert out[0]["class"] == "water_stain"


def test_duplicate_boxes_of_same_class_merged_keeping_best():
    a = det("crack", 0.5, (100, 100, 200, 180))
    b = det("crack", 0.7, (105, 102, 205, 182))
    c = det("hole", 0.6, (105, 102, 205, 182))
    out = postprocess([a, b, c], (H, W), threshold=0.3)
    assert sorted((o["class"], o["score"]) for o in out) == [("crack", 0.7), ("hole", 0.6)]


def test_mask_clipped_to_its_box():
    d = det("water stain", 0.9, (100, 100, 200, 180), fill=(50, 50, 300, 300))
    out = postprocess([d], (H, W), threshold=0.3)
    ys, xs = np.nonzero(out[0]["mask"])
    assert xs.min() >= 100 - 3 and xs.max() < 200 + 3 and ys.min() >= 100 - 3 and ys.max() < 180 + 3


ROWS = {0: [[1, 0, 0], [0, -1, 0], [0, 0, -1]],      # world up seen pointing to the image top
        1: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],      # ... pointing right -> turn image 90 deg CCW
        2: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],       # ... pointing down
        3: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]}      # ... pointing left


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_upright_turns_from_camera_pose(k):
    T = np.eye(4)
    T[:3, :3] = np.array(ROWS[k], float)        # second row = world up in camera coordinates
    assert upright_turns(T) == k


def test_owlv2_phrases_map_to_damage_classes():
    from roomscan.damage_detect import OWL_PHRASES, label_to_class
    assert {label_to_class(p) for p in OWL_PHRASES} == {"crack", "water_stain", "mold", "peeling_paint", "hole"}


def test_each_detector_has_its_own_threshold():
    from roomscan.damage_detect import DETECTORS, THRESHOLDS
    assert set(THRESHOLDS) == set(DETECTORS) and THRESHOLDS["owlv2+sam2"] < THRESHOLDS["gdino+sam2"]
