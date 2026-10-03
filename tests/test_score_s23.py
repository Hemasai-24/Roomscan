import json

from scripts.score_s23 import score_all


def M(v, h=0.02):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": "m"}


def _plan(path, tier, lengths):
    walls = [{"id": f"r0_w{i}", "start": [0, 0], "end": [L, 0], "length": M(L)} for i, L in enumerate(lengths)]
    plan = {"capture": {"tier": tier}, "rooms": [{"id": "r0", "walls": walls, "openings": [],
            "ceiling_height": M(2.5), "ceiling_observed": True, "floor_area": M(12.0)}]}
    path.mkdir(parents=True)
    (path / "plan.json").write_text(json.dumps(plan))


def test_scores_each_run_with_its_room_map(tmp_path):
    runs = tmp_path / "s23"
    _plan(runs / "photo", "photo", [4.1, 3.0, 4.1, 3.0])
    _plan(runs / "video_walkthrough", "video", [4.0, 3.0, 4.0, 3.0])
    gt = tmp_path / "gt.csv"
    gt.write_text("room,item,id,value_cm,notes\n" + "".join(
        f"room1,wall,W{i + 1},{v},\n" for i, v in enumerate([400, 300, 400, 300])))
    rmap = tmp_path / "map.json"
    rmap.write_text(json.dumps({"photo": {"room1": "r0"}, "video_walkthrough": {"room1": "r0"}}))
    out = score_all(runs, gt, rmap)
    assert set(out["runs"]) == {"photo", "video_walkthrough"}
    assert abs(out["runs"]["photo"]["walls"]["max_abs_err_m"] - 0.1) < 1e-9
    assert out["runs"]["video_walkthrough"]["gates"]["wall_lengths"] is True
