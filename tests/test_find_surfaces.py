import pytest
import numpy as np
from roomscan.find_surfaces import classify_planes, extract_planes
from tests.synthetic_rooms import room_points, table_points


def test_box_room_planes():
    p = np.concatenate([room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6),
                        table_points(1, 1, 2, 1.8, top=0.9)])
    c = classify_planes(extract_planes(p))
    assert abs(c["floor"].height - 0.0) < 0.01
    assert abs(c["ceiling"].height - 2.6) < 0.01
    assert len(c["walls"]) == 4
    assert any(abs(o.height - 0.9) < 0.02 for o in c["other"])  # table is not floor/ceiling
    for w in c["walls"]:
        assert abs(w.normal[1]) < 0.05


def test_no_ceiling_returns_none():
    p = room_points([(0, 0), (3, 0), (3, 3), (0, 3)], 2.5, ceiling=False)
    c = classify_planes(extract_planes(p))
    assert c["ceiling"] is None


def test_deterministic():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6)
    a = extract_planes(p)
    b = extract_planes(p)
    assert [round(x.d, 6) for x in a] == [round(x.d, 6) for x in b]


def test_near_coplanar_floor_slabs_merge():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6)
    p[(p[:, 1] < 0.02) & (p[:, 0] > 2.0), 1] += 0.03     # half the floor 3 cm higher (drift/tilt)
    c = classify_planes(extract_planes(p))
    n_floor = int(((p[:, 1] < 0.05) & (p[:, 0] > 0.05) & (p[:, 0] < 3.95)).sum())
    assert len(c["floor"].inliers) >= 0.9 * n_floor
    assert c["floor_spread"] >= 0.01


def test_no_floor_raises_capture_error():
    import pytest
    from roomscan.find_surfaces import CaptureError
    walls_only = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6, ceiling=False)
    walls_only = walls_only[walls_only[:, 1] > 0.3]
    with pytest.raises(CaptureError):
        classify_planes(extract_planes(walls_only))


def test_no_fake_horizontal_slices_through_walls():
    from roomscan.point_cloud import estimate_normals
    # tall narrow room: lots of wall area, little floor -> a plain RANSAC slice at mid-height
    # through all four walls collects more points than a real small plane
    p = room_points([(0, 0), (1.2, 0), (1.2, 6), (0, 6)], 3.0, ceiling=False)
    planes = extract_planes(p, estimate_normals(p), min_inliers=300)
    horiz = [pl for pl in planes if abs(pl.normal[1]) > 0.95]
    assert all(abs(pl.height) < 0.03 for pl in horiz)        # only the floor


@pytest.mark.sample
def test_sample_floor_only_finds_many_walls():
    from tests.conftest import SAMPLE
    from roomscan.load_capture import load_stray
    from roomscan.point_cloud import fuse_points, estimate_normals
    root = SAMPLE / "single_scan_floor_only" / "1a8384c3f6"
    if not root.exists():
        pytest.skip("sample_data not present")
    pts = fuse_points(load_stray(root), stride=10)
    c = classify_planes(extract_planes(pts, estimate_normals(pts)))
    assert len(c["walls"]) >= 15
    fh = c["floor"].height
    fake = [p for p in c["horizontal"] if 0.2 < p.height - fh < 0.7 and len(p.inliers) > 15000]
    assert fake == []
