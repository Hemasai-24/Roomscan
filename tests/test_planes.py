import numpy as np
from fpp.geometry.planes import classify_planes, extract_planes
from tests.synth import room_points, table_points


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
