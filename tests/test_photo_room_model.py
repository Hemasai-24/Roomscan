import numpy as np
import pytest

from tests.conftest import SINGLE_ROOM, need_sample

pytestmark = [pytest.mark.model, pytest.mark.sample]


def test_real_photos_of_sample_room_give_a_room(tmp_path):
    need_sample()
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if not (root / "weights" / "vggt-1b" / "model.pt").exists():
        pytest.skip("weights not fetched")
    from roomscan.photo_room import measure_photo_room, photos_to_capture
    from roomscan.video_frames import read_frames
    from roomscan.video_poses import VGGTRunner
    from roomscan.video_scale import MetricDepth, load_bias
    want = {100, 400, 700, 1000, 1300, 1600}
    imgs = np.stack([img for i, img in read_frames(SINGLE_ROOM / "rgb.mp4", (392, 518), rotate=90) if i in want])
    cap_dir, info = photos_to_capture(imgs, None, tmp_path / "cap", VGGTRunner(root), MetricDepth(root), load_bias())
    out = measure_photo_room(cap_dir, "room")
    assert info["n_photos"] == 6
    assert len(out["room"]["polygon"]) >= 4
    floor_h = out["room"]["ceiling_height"]          # schema-complete record
    assert floor_h["hi"] > floor_h["lo"]
