import numpy as np
from shapely.geometry import Polygon
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.point_cloud import estimate_normals
from roomscan.room_outline import manhattan_angle, outline_from_mask
from roomscan.room_surfaces import surfaces_for_room
from roomscan.split_rooms import split_rooms
from tests.synthetic_rooms import two_rooms_with_door


def _setup():
    p = two_rooms_with_door()
    c = classify_planes(extract_planes(p, estimate_normals(p), min_inliers=500))
    a = manhattan_angle(c["walls"])
    masks, g = split_rooms(c, a)
    return c, a, masks, g


def test_each_room_outline_matches_its_walls():
    c, a, masks, g = _setup()
    areas = []
    for m in masks:
        verts, edges = outline_from_mask(m, g, c["walls"], a)
        areas.append(Polygon(verts).area)
        assert len(edges) == 4
    np.testing.assert_allclose(sorted(areas), [9.0, 12.0], rtol=0.03)


def test_rooms_do_not_overlap():
    c, a, masks, g = _setup()
    polys = [Polygon(outline_from_mask(m, g, c["walls"], a)[0]) for m in masks]
    assert polys[0].intersection(polys[1]).area < 0.05


def test_per_room_ceiling_found():
    c, a, masks, g = _setup()
    for m in masks:
        s = surfaces_for_room(c, m, g, a)
        assert s["ceiling"] is not None and abs(s["ceiling"].height - 2.5) < 0.02
