import numpy as np
from shapely.geometry import Polygon
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.room_outline import footprint
from tests.synthetic_rooms import room_points


def _rot(p, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return p @ R.T


def test_rotated_box_lengths_and_area():
    p = _rot(room_points([(0, 0), (4.2, 0), (4.2, 3.1), (0, 3.1)], 2.5), 27)
    poly, edges, _ = footprint(classify_planes(extract_planes(p)))
    lengths = sorted(round(abs(e.end - e.start), 2) for e in edges)
    np.testing.assert_allclose(lengths, [3.1, 3.1, 4.2, 4.2], atol=0.01)
    assert abs(Polygon(poly).area - 4.2 * 3.1) < 0.05


def test_l_shape_keeps_notch():
    fp = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 4), (0, 4)]
    p = room_points(fp, 2.5)
    poly, edges, _ = footprint(classify_planes(extract_planes(p, min_inliers=800)))
    assert len(edges) == 6
    assert abs(Polygon(poly).area - (5 * 2 + 3 * 2)) < 0.08


def test_edges_connect_consecutive_vertices():
    p = _rot(room_points([(0, 0), (5, 0), (5, 2), (3, 2), (3, 4), (0, 4)], 2.5), 40)
    poly, edges, _ = footprint(classify_planes(extract_planes(p, min_inliers=800)))
    for i, e in enumerate(edges):
        a, b = poly[i - 1], poly[i]
        k, j = (0, 1) if e.axis == "u" else (1, 0)
        assert abs(a[k] - e.offset) < 1e-9 and abs(b[k] - e.offset) < 1e-9
        assert abs(a[j] - e.start) < 1e-9 and abs(b[j] - e.end) < 1e-9


def test_small_jog_gives_no_zero_length_walls():
    fp = [(0, 0), (4, 0), (4, 1.5), (4.1, 1.5), (4.1, 3), (0, 3)]
    p = room_points(fp, 2.5)
    poly, edges, _ = footprint(classify_planes(extract_planes(p, min_inliers=800)))
    assert min(abs(e.end - e.start) for e in edges) >= 0.05


def test_non_manhattan_room_does_not_crash():
    p = room_points([(0, 0), (4, 0), (0, 3)], 2.5)
    poly, edges, _ = footprint(classify_planes(extract_planes(p, min_inliers=800)))
    assert len(poly) >= 4


def test_parallel_walls_12cm_apart_keep_both():
    fp = [(0, 0), (4, 0), (4, 1.5), (4.12, 1.5), (4.12, 3), (0, 3)]
    p = room_points(fp, 2.5)
    poly, edges, _ = footprint(classify_planes(extract_planes(p, min_inliers=800)))
    assert min(abs(e.end - e.start) for e in edges) >= 0.05
