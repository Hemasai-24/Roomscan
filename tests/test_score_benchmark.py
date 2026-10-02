import pytest
from roomscan.score_benchmark import load_ground_truth, match_room_walls, score_plan


def M(v, h):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": "m"}


def plan(lengths, half=0.02, ceiling=(2.50, 0.01), doors=(0.80,)):
    walls = [{"id": f"room_0_w{i}", "length": M(L, half)} for i, L in enumerate(lengths)]
    ops = [{"id": f"room_0_o{i}", "wall_id": "room_0_w0", "type": "door", "width": M(w, 0.01)}
           for i, w in enumerate(doors)]
    return {"capture": {"tier": "lidar"}, "rooms": [{"id": "room_0", "walls": walls, "openings": ops,
            "ceiling_height": M(*ceiling), "ceiling_observed": True,
            "floor_area": {"value": lengths[0] * lengths[1], "lo": 0, "hi": 99, "unit": "m2"}}]}


GT = {"room1": {"wall": {"W1": 4.00, "W2": 3.00, "W3": 4.00, "W4": 3.00},
                "ceiling_height": {"C1": 2.495, "C2": 2.505}, "door_width": {"D1": 0.81}}}


def test_load_ground_truth_csv(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("room,item,id,value_cm,notes\nroom1,wall,W1,400,\nroom1,wall,W2,,\nroom1,door_width,D1,81,\n")
    gt = load_ground_truth(p)
    assert gt == {"room1": {"wall": {"W1": 4.0}, "door_width": {"D1": 0.81}}}


def test_walls_matched_by_cyclic_order_either_direction():
    pred = [3.01, 4.02, 2.99, 3.98]                     # starts at a different wall
    pairs = match_room_walls(GT["room1"]["wall"], pred)
    assert [round(p, 2) for _, p in pairs] == [4.02, 2.99, 3.98, 3.01]


def test_score_reports_gates_and_coverage():
    s = score_plan(plan([4.01, 3.00, 3.99, 3.02]), GT, {"room1": "room_0"})
    assert s["walls"]["n"] == 4
    assert s["walls"]["max_abs_err_m"] == pytest.approx(0.02, abs=1e-9)
    assert s["walls"]["coverage"] == 1.0                   # every truth inside the +-2 cm range
    assert s["ceiling"]["abs_err_m"] == pytest.approx(0.0, abs=1e-9)
    assert s["openings"]["within_2cm"] == 1 and s["openings"]["missed"] == 0
    assert s["gates"]["opening_widths"] is True


def test_missed_and_phantom_openings_count_as_misses():
    s = score_plan(plan([4, 3, 4, 3], doors=(0.80, 0.90)), {"room1": dict(GT["room1"], door_width={})},
                   {"room1": "room_0"})
    assert s["openings"]["phantom"] == 2 and s["gates"]["opening_widths"] is False
