import jsonschema
import pytest

from roomscan.save_plan import build_plan, validate


def M(v, unit="m"):
    return {"value": v, "lo": v - 0.01, "hi": v + 0.01, "unit": unit}


def _plan():
    p = build_plan("x", "lidar", [], {})
    p["damage"] = [{"id": "dmg_0", "class": "water_stain", "surface_id": "room_0_w1", "room_id": "room_0",
                    "area": M(0.06, "m2"), "width": M(0.3), "height": M(0.2), "bottom_above_floor": M(1.0),
                    "n_views": 2, "score": 0.5, "near_opening": False}]
    p["concealed_damage_flags"] = [{"rule_id": "R2", "description": "d", "severity": "medium",
                                    "damage_ids": ["dmg_0"], "surface_ids": ["room_0_w1"]}]
    p["scope_items"] = [{"id": "s0", "surface_id": "room_0_w1", "action": "inspect",
                         "quantity": {"value": 1, "lo": 1, "hi": 1, "unit": "each"}, "unit": "each",
                         "reason": "R2"}]
    return p


def test_plan_with_damage_validates():
    validate(_plan())


@pytest.mark.parametrize("key,field", [("damage", "surface_id"), ("damage", "class"), ("damage", "area"),
                                       ("concealed_damage_flags", "rule_id"),
                                       ("concealed_damage_flags", "damage_ids"),
                                       ("scope_items", "surface_id"), ("scope_items", "quantity")])
def test_missing_required_field_rejected(key, field):
    p = _plan()
    del p[key][0][field]
    with pytest.raises(jsonschema.ValidationError):
        validate(p)


def test_unknown_damage_class_rejected():
    p = _plan()
    p["damage"][0]["class"] = "graffiti"
    with pytest.raises(jsonschema.ValidationError):
        validate(p)
