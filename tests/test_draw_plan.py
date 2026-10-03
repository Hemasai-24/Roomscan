import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from roomscan.draw_plan import setup_axes


def test_plan_axes_not_mirrored():
    fig, ax = plt.subplots()
    setup_axes(ax)
    assert ax.yaxis_inverted()      # screen-up is -z when looking down from +y
    plt.close(fig)


def test_damage_and_flags_drawn(tmp_path):
    from roomscan.draw_plan import render
    M = lambda v, u="m": {"value": v, "lo": v - 0.01, "hi": v + 0.01, "unit": u}
    walls = [{"id": f"room_0_w{i}", "start": list(a), "end": list(b), "length": M(1.0)}
             for i, (a, b) in enumerate([((0, 3), (0, 0)), ((0, 0), (4, 0)), ((4, 0), (4, 3)), ((4, 3), (0, 3))])]
    room = {"id": "room_0", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]], "walls": walls, "openings": [],
            "floor_area": M(12.0, "m2"), "ceiling_height": M(2.5)}
    dmg = {"id": "dmg_0", "class": "water_stain", "surface_id": "room_0_w1", "room_id": "room_0",
           "area": M(0.06, "m2"), "offset_along_wall": 1.5}
    ceil = {"id": "dmg_1", "class": "mold", "surface_id": "room_0_ceiling", "room_id": "room_0",
            "area": M(0.2, "m2")}
    plan = {"capture": {"id": "t", "tier": "lidar"}, "rooms": [room], "adjacency": [], "damage": [dmg, ceil],
            "concealed_damage_flags": [{"rule_id": "R1", "description": "Leak from above?", "damage_ids": ["dmg_1"],
                                        "surface_ids": ["room_0_ceiling"], "severity": "high"}]}
    render(plan, tmp_path / "p")
    svg = (tmp_path / "p.svg").read_text()
    assert "dmg_0" in svg and "dmg_1" in svg and "R1" in svg


def M(v, h, unit="m"):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": unit}


def test_short_walls_get_no_label_and_long_ones_show_their_range():
    from roomscan.draw_plan import wall_label
    assert wall_label({"length": M(0.3, 0.02)}) is None
    assert wall_label({"length": M(3.42, 0.06)}) == "3.42 m ±6 cm"


def test_room_label_says_when_the_ceiling_was_not_seen():
    from roomscan.draw_plan import room_label
    room = {"name": "kitchen", "floor_area": M(4.07, 1.2, "m2"), "ceiling_height": M(2.9, 0.6), "ceiling_observed": False}
    assert room_label(room) == "kitchen\n4.1 m² ±1.2\nceiling not seen"
    room["ceiling_observed"] = True
    assert room_label(room).endswith("ceiling 2.90 m")
