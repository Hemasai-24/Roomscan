"""Score our own benchmark (Samsung Galaxy M53, tape-measured) and write every reported number.

usage: python scripts/score_m53.py [photo_plan.json] [before_photo_plan.json]
Defaults: outputs/fix_loop/after_photo_v2/plan.json and outputs/fix_loop/before_photo/plan.json, plus the video
runs in outputs/m53/video_<name>/plan.json. Writes outputs/m53_benchmark.json and prints the tables used in
docs/benchmark_report.md and docs/fix_loop.md."""
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.compare_plans import align_plans, match_walls   # noqa: E402
from roomscan.score_benchmark import load_ground_truth, score_plan   # noqa: E402

GT = "data/ground_truth/own_rooms.csv"
PHOTO_MAP = json.loads(Path(__file__).resolve().parents[1].joinpath("data/ground_truth/room_map.json").read_text())["photo"]


def true_areas(gt):
    out = {}
    for room, items in gt.items():
        w = sorted(items.get("wall", {}).items())
        if len(w) == 4:
            out[room] = w[0][1] * w[1][1]
    return out


def photo_rows(plan, gt, areas):
    rooms = {r["id"]: r for r in plan["rooms"]}
    rows = {}
    for g, rid in PHOTO_MAP.items():
        a = rooms[rid]["floor_area"]
        rows[g] = {"area": round(a["value"], 2), "truth": round(areas[g], 2),
                   "err_pct": round((a["value"] / areas[g] - 1) * 100, 1), "in_range": a["lo"] <= areas[g] <= a["hi"],
                   "range": [a["lo"], a["hi"]]}
    tot = sum(rooms[r]["floor_area"]["value"] for r in PHOTO_MAP.values())
    s = score_plan(plan, gt, PHOTO_MAP)
    return {"rooms": rows, "total": round(tot, 2), "total_truth": round(sum(areas.values()), 2),
            "total_err_pct": round((tot / sum(areas.values()) - 1) * 100, 1), "score": s,
            "adjacency": [(a["room_a"], a["room_b"]) for a in plan["adjacency"]],
            "unplaced": sum("drawn beside" in w for w in plan["warnings"]), "damage": len(plan["damage"]),
            "runtime_s": plan["meta"]["runtime_s"]}


def video_rows(areas):
    out = {}
    for n in ["walkthrough", "bedroom_take1", "bedroom_take2", "bathroom_lowlight"]:
        p = Path(f"outputs/m53/video_{n}/plan.json")
        if not p.exists():
            continue
        q = json.loads(p.read_text())
        out[n] = {"rooms": [(r["id"], round(r["floor_area"]["value"], 2)) for r in q["rooms"]],
                  "total": round(sum(r["floor_area"]["value"] for r in q["rooms"]), 2),
                  "ceiling_lower_bounds": [round(r["ceiling_height"]["lo"], 2) for r in q["rooms"]],
                  "damage": len(q["damage"]), "runtime_s": q["meta"]["runtime_s"]}
    a, b = Path("outputs/m53/video_bedroom_take1/plan.json"), Path("outputs/m53/video_bedroom_take2/plan.json")
    if a.exists() and b.exists():
        pa, pb = json.loads(a.read_text()), json.loads(b.read_text())
        R2, t = align_plans(pa, pb)
        rows = match_walls(pa, pb, R2, t)
        out["repeatability_bedroom"] = {"walls_matched": len(rows), "within_gate": sum(r["pass"] for r in rows),
                                        "diffs_cm": sorted(round(r["diff_m"] * 100, 1) for r in rows)}
    return out


def main():
    after = sys.argv[1] if len(sys.argv) > 1 else "outputs/fix_loop/after_photo_v2/plan.json"
    before = sys.argv[2] if len(sys.argv) > 2 else "outputs/fix_loop/before_photo/plan.json"
    gt = load_ground_truth(GT)
    areas = true_areas(gt)
    res = {"truth_areas_m2": {k: round(v, 2) for k, v in areas.items()}}
    for name, path in (("photo_before_fix", before), ("photo_after_fix", after)):
        if Path(path).exists():
            res[name] = photo_rows(json.loads(Path(path).read_text()), gt, areas)
    res["video"] = video_rows(areas)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/m53_benchmark.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
