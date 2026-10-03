import numpy as np
import pytest

from tests.conftest import SAMPLE

pytestmark = pytest.mark.model


def test_painted_stain_and_crack_are_found_with_masks():
    root = SAMPLE / "single_scan_with_ceiling" / "c7d28f72c6"
    if not root.exists():
        pytest.skip("sample_data not present")
    from roomscan.damage_detect import detect_damage, release_detectors, upright_turns
    from roomscan.load_capture import load_stray
    from roomscan.video_frames import read_frames
    from scripts.review_damage import paint_damage
    cap = load_stray(root)
    i = 6663                                     # a plain wall + ceiling corner frame (review script)
    img = next(im for j, im in read_frames(root / "rgb.mp4", (640, 480)) if j == i)
    img = np.ascontiguousarray(np.rot90(img, upright_turns(cap.frames[i].T_wc)))
    painted, truth = paint_damage(img)
    try:
        found = {d["class"]: d for d in detect_damage(painted)}
    finally:
        release_detectors()
    assert {"water_stain", "crack"} <= set(found)
    x0, y0, x1, y1 = truth["stain_box"]
    ys, xs = np.nonzero(found["water_stain"]["mask"])
    assert x0 - 10 <= xs.mean() <= x1 + 10 and y0 - 10 <= ys.mean() <= y1 + 10
