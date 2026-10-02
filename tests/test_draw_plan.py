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
