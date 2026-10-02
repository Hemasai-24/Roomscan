import pytest
from roomscan.repair_scope import scope_items
from tests.test_damage_rules import M, dmg


def room():
    walls = [{"id": "room_0_w0", "length": M(4.0)}, {"id": "room_0_w1", "length": M(3.0)}]
    return {"id": "room_0", "walls": walls, "ceiling_height": M(2.5), "floor_area": M(12.0, 0.1, "m2"),
            "openings": []}


def test_stain_repaints_whole_wall_with_range():
    items = scope_items([dmg("water_stain", "room_0_w0")], [], [room()])
    paint = [i for i in items if i["action"] == "stain_block_and_repaint"]
    assert len(paint) == 1 and paint[0]["surface_id"] == "room_0_w0"
    q = paint[0]["quantity"]
    assert q["value"] == pytest.approx(10.0) and q["unit"] == "m2" and q["lo"] < 10.0 < q["hi"]


def test_crack_fill_by_length_plus_repaint():
    items = scope_items([dmg("crack", "room_0_w1", size=(0.5, 0.02))], [], [room()])
    acts = sorted(i["action"] for i in items)
    assert acts == ["fill_crack", "repaint"]
    fill = next(i for i in items if i["action"] == "fill_crack")
    assert fill["quantity"]["value"] == pytest.approx(0.5) and fill["unit"] == "m"


def test_mold_remediation_has_margin_and_minimum():
    items = scope_items([dmg("mold", "room_0_w1", size=(0.2, 0.2))], [], [room()])
    rem = next(i for i in items if i["action"] == "mold_remediation")
    assert rem["quantity"]["value"] == pytest.approx(1.0)        # 0.04 * 1.5 < 1 m2 minimum


def test_ceiling_damage_uses_floor_area():
    items = scope_items([dmg("water_stain", "room_0_ceiling", bottom=2.5)], [], [room()])
    assert next(i for i in items if i["action"] == "stain_block_and_repaint")["quantity"]["value"] == pytest.approx(12.0)


def test_each_flag_adds_an_inspection():
    flag = {"rule_id": "R1", "description": "x", "damage_ids": ["d0"], "surface_ids": ["room_0_ceiling"],
            "severity": "high"}
    items = scope_items([], [flag], [room()])
    assert [(i["action"], i["reason"]) for i in items] == [("inspect", "R1")]


def test_items_have_unique_ids_and_reasons():
    items = scope_items([dmg("water_stain", "room_0_w0"), dmg("crack", "room_0_w1", did="d1")], [], [room()])
    assert len({i["id"] for i in items}) == len(items)
    assert all(i["reason"] for i in items)
