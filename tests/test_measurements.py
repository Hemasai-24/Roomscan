import numpy as np
from roomscan.find_surfaces import classify_planes, extract_planes
from roomscan.room_outline import footprint
from roomscan.measurements import measure_room
from roomscan.measurements import interval
from tests.synthetic_rooms import room_points


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


def test_video_ranges_reflect_measured_video_error():
    """Sample single_room video vs LiDAR: wall errors of 20-100% (docs/plans/plan3-video-tier.md, Task 6).
    Until re-fitted on tape-measured videos, a 2 m video wall must carry a 95% range of at least +-40%."""
    from roomscan.measurements import interval, TIER_REL, TIER_SIGMA_FLOOR
    L = 2.0
    sig = np.sqrt(2 * TIER_SIGMA_FLOOR["video"] ** 2 + (TIER_REL["video"] * L) ** 2)
    m = interval(L, sig)
    assert (m.hi - m.lo) / 2 >= 0.4 * L


def test_ranges_never_negative():
    assert interval(0.1, 1.0).lo == 0.0


def test_unseen_ceiling_in_video_tier_includes_scale_uncertainty():
    p = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], 2.6, ceiling=False)
    c = classify_planes(extract_planes(p))
    v, e, a = footprint(c)
    lidar = measure_room(c, v, e, a, [], tier="lidar")["ceiling_height"]
    video = measure_room(c, v, e, a, [], tier="video")["ceiling_height"]
    assert video["lo"] < lidar["lo"] - 0.3          # walls seen up to 2.6 m in a video whose scale may be 25% off
    assert video["hi"] > lidar["hi"]
