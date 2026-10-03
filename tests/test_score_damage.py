import pytest

from roomscan.score_damage import score_damage


def M(v, h=0.03, unit="m"):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": unit}


GT = {"room1": {"damage_water_stain_width": {"S1": 0.30}, "damage_water_stain_height": {"S1": 0.21},
                "damage_crack_length": {"K1": 0.45}}}


def plan(damage):
    return {"damage": damage}


def d(cls, room, w, h, area=None):
    return {"class": cls, "room_id": room, "width": M(w), "height": M(h),
            "area": M(area if area is not None else w * h, 0.01, "m2"), "surface_id": f"{room}_w0"}


def test_found_stain_and_crack_with_errors():
    s = score_damage(plan([d("water_stain", "room_2", 0.32, 0.20), d("crack", "room_2", 0.02, 0.43),
                           d("hole", "room_2", 0.3, 0.3)]), GT, {"room1": "room_2"})
    st, cr = s["water_stain"], s["crack"]
    assert st["detected"] and st["width_err_m"] == pytest.approx(0.02) and st["height_err_m"] == pytest.approx(0.01)
    assert st["width_in_range"] and st["height_in_range"]
    assert cr["detected"] and cr["length_err_m"] == pytest.approx(0.02)
    assert s["extra_detections_in_room"] == 1


def test_missed_damage_and_other_room_ignored():
    s = score_damage(plan([d("water_stain", "room_5", 0.3, 0.2)]), GT, {"room1": "room_2"})
    assert s["water_stain"]["detected"] is False and s["crack"]["detected"] is False
    assert s["extra_detections_in_room"] == 0


def test_largest_matching_region_is_scored():
    s = score_damage(plan([d("water_stain", "room_2", 0.05, 0.05), d("water_stain", "room_2", 0.29, 0.22)]),
                     GT, {"room1": "room_2"})
    assert s["water_stain"]["width_err_m"] == pytest.approx(0.01)
    assert s["extra_detections_in_room"] == 1
