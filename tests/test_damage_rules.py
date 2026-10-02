from roomscan.damage_rules import RULES, concealed_flags


def M(v, h=0.01, unit="m"):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": unit}


def dmg(cls, surface, bottom=1.0, size=(0.3, 0.2), did="d0", near_opening=False):
    return {"id": did, "class": cls, "surface_id": surface, "room_id": surface.rsplit("_", 1)[0],
            "width": M(size[0]), "height": M(size[1]), "area": M(size[0] * size[1], 0.005, "m2"),
            "bottom_above_floor": M(bottom), "n_views": 2, "score": 0.6, "near_opening": near_opening}


def room(rid, floor_level=0.0, openings=()):
    return {"id": rid, "floor_level": floor_level, "walls": [{"id": f"{rid}_w{i}"} for i in range(4)],
            "openings": list(openings)}


def fired(flags):
    return sorted(f["rule_id"] for f in flags)


def test_rules_have_ids_and_descriptions():
    assert {r["id"] for r in RULES} == {"R1", "R2", "R3", "R4", "R5"}
    assert all(r["description"] for r in RULES)


def test_ceiling_stain_fires_r1_only():
    assert fired(concealed_flags([dmg("water_stain", "room_0_ceiling", bottom=2.5)], [room("room_0")], [])) == ["R1"]


def test_low_wall_stain_fires_r2():
    assert fired(concealed_flags([dmg("water_stain", "room_0_w1", bottom=0.1)], [room("room_0")], [])) == ["R2"]


def test_high_wall_stain_fires_nothing():
    assert concealed_flags([dmg("water_stain", "room_0_w1", bottom=1.2)], [room("room_0")], []) == []


def test_mold_always_fires_r3():
    assert "R3" in fired(concealed_flags([dmg("mold", "room_0_w1", bottom=1.2)], [room("room_0")], []))


def test_long_crack_or_crack_at_opening_fires_r4():
    long_crack = dmg("crack", "room_0_w1", size=(1.3, 0.02))
    short_at_door = dmg("crack", "room_0_w2", size=(0.3, 0.02), did="d1", near_opening=True)
    short = dmg("crack", "room_0_w3", size=(0.3, 0.02), did="d2")
    flags = concealed_flags([long_crack, short_at_door, short], [room("room_0")], [])
    assert fired(flags) == ["R4", "R4"]
    assert {f["damage_ids"][0] for f in flags} == {"d0", "d1"}


def test_stain_on_wall_towards_lower_wet_room_fires_r5():
    a = room("room_0", 0.0, [{"id": "room_0_o0", "wall_id": "room_0_w1", "type": "door"}])
    b = room("room_1", -0.05)
    adj = [{"room_a": "room_0", "room_b": "room_1", "via": "room_0_o0"}]
    flags = concealed_flags([dmg("water_stain", "room_0_w1", bottom=1.0)], [a, b], adj)
    assert fired(flags) == ["R5"]
    assert flags[0]["surface_ids"] == ["room_0_w1"]


def test_flags_name_their_evidence():
    f = concealed_flags([dmg("water_stain", "room_0_ceiling", bottom=2.5)], [room("room_0")], [])[0]
    assert f["damage_ids"] == ["d0"] and f["surface_ids"] == ["room_0_ceiling"] and f["severity"]
