import numpy as np
from fpp.geometry.planes import classify_planes, extract_planes
from fpp.geometry.room import footprint
from fpp.measure import measure_room
from fpp.uncertainty import interval
from tests.synth import room_points


def test_interval_symmetric_95():
    m = interval(2.0, 0.01)
    assert m.lo < 2.0 < m.hi
    assert abs((m.hi - m.lo) / 2 - 0.0196) < 1e-9


def _room(ceiling=True):
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6, ceiling=ceiling)
    c = classify_planes(extract_planes(p))
    v, e, a = footprint(c)
    return measure_room(c, v, e, a, [])


def test_room_measures_contain_truth():
    r = _room()
    ch = r["ceiling_height"]
    assert ch["lo"] <= 2.6 <= ch["hi"]
    assert r["ceiling_observed"] is True
    fa = r["floor_area"]
    assert fa["lo"] <= 12.0 <= fa["hi"]
    for w in r["walls"]:
        L = w["length"]
        assert any(L["lo"] <= t <= L["hi"] for t in (3.0, 4.0))


def test_missing_ceiling_is_honest():
    r = _room(ceiling=False)
    assert r["ceiling_observed"] is False
    ch = r["ceiling_height"]
    assert ch["hi"] - ch["lo"] >= 0.3
    assert any("ceiling" in w for w in r["warnings"])


def test_unseen_ceiling_interval_respects_lower_bound():
    r = _room(ceiling=False)          # walls observed up to 2.6 m
    ch = r["ceiling_height"]
    assert ch["lo"] >= 2.6 - 0.02     # never below the highest observed wall point
    assert ch["hi"] >= 2.6 + 0.3


def test_perimeter_interval_accounts_for_shared_offsets():
    r = _room()
    half = (r["perimeter"]["hi"] - r["perimeter"]["lo"]) / 2
    # rectangle: each offset moves the perimeter by 2x -> sigma_P = 2 * sigma * sqrt(n) = 4 sigma
    assert half >= 1.96 * 4 * 0.008 * 0.999
