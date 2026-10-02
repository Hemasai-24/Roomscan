"""Photo tier vs LiDAR tier on folders made by scripts/make_photo_folders.py (the LiDAR plan is the
reference; the sample data has no tape ground truth).

Usage: python scripts/eval_photo_vs_lidar.py <out_dir_of_make_photo_folders>
Runs the photo tier on <out_dir>/rooms, writes <out_dir>/photo/plan.{json,svg,png} and <out_dir>/eval.json,
prints per-room and whole-property tables."""
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.compare_plans import align_partial      # noqa: E402
from roomscan.draw_plan import render                 # noqa: E402
from roomscan.photo_pipeline import run_photos        # noqa: E402
from roomscan.save_plan import validate               # noqa: E402


def _pairs(plan):
    return {tuple(sorted((a["room_a"], a["room_b"]))) for a in plan["adjacency"]}


def _top_walls(room, k=4):
    return sorted((w["length"]["value"] for w in room["walls"]), reverse=True)[:k]


def _top_wall_ranges(room, k=4):
    ws = sorted(room["walls"], key=lambda w: -w["length"]["value"])[:k]
    return [(w["length"]["lo"], w["length"]["hi"]) for w in ws]


def main():
    out = Path(sys.argv[1])
    ref = json.loads((out / "lidar" / "plan.json").read_text())
    photo_path = out / "photo" / "plan.json"
    if photo_path.exists() and "--rerun" not in sys.argv:
        plan = json.loads(photo_path.read_text())
    else:
        plan = run_photos(out / "rooms", out / "photo")
        validate(plan)
        photo_path.parent.mkdir(parents=True, exist_ok=True)
        photo_path.write_text(json.dumps(plan, indent=2))
        render(plan, out / "photo" / "plan")
    lid = {r["id"]: r for r in ref["rooms"]}
    rows, covered, n_cov, wall_err, wall_in = [], 0, 0, [], []
    for r in plan["rooms"]:
        L = lid.get(r["id"])
        if L is None:
            continue
        a, la = r["floor_area"], L["floor_area"]["value"]
        inside = a["lo"] <= la <= a["hi"]
        covered += inside
        n_cov += 1
        pw, lw = _top_walls(r), _top_walls(L)
        k = min(len(pw), len(lw))
        errs = [abs(p / q - 1) for p, q in zip(pw[:k], lw[:k])]
        wall_err += errs
        wall_in += [lo <= q <= hi for (lo, hi), q in zip(_top_wall_ranges(r)[:k], lw[:k])]
        rows.append({"room": r["id"], "photo_area": round(a["value"], 2), "range": [a["lo"], a["hi"]],
                     "lidar_area": round(la, 2), "area_err_pct": round((a["value"] / la - 1) * 100, 1),
                     "lidar_in_range": bool(inside), "top_wall_err_pct": [round(e * 100) for e in errs],
                     "warnings": [w for w in plan["warnings"] if w.startswith(r["id"] + ":")]})
    pa = unary_union([Polygon(r["polygon"]).buffer(0) for r in plan["rooms"]])
    la_ = unary_union([Polygon(r["polygon"]).buffer(0) for r in ref["rooms"] if r["id"] in {x["room"] for x in rows}])
    ps = [Polygon(r["polygon"]).buffer(0) for r in plan["rooms"]]
    overlap = max([ps[i].intersection(ps[j]).area for i in range(len(ps)) for j in range(i + 1, len(ps))] + [0.0])
    R2, t = align_partial(ref, plan)
    from shapely import affinity
    pa_al = affinity.affine_transform(pa, [R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1], t[0], t[1]])
    iou = pa_al.intersection(la_).area / max(pa_al.union(la_).area, 1e-9)
    truth, found = _pairs(ref), _pairs(plan)
    tp = len(truth & found)
    summary = {"rooms_photo": len(plan["rooms"]), "rooms_lidar": len(ref["rooms"]),
               "footprint_photo_m2": round(sum(r["floor_area"]["value"] for r in plan["rooms"]), 2),
               "footprint_lidar_m2": round(sum(lid[x["room"]]["floor_area"]["value"] for x in rows), 2),
               "footprint_iou_after_alignment": round(float(iou), 3),
               "adjacency_found": sorted(found), "adjacency_lidar": sorted(truth),
               "adjacency_precision": round(tp / len(found), 2) if found else None,
               "adjacency_recall": round(tp / len(truth), 2) if truth else None,
               "max_overlap_m2": round(float(overlap), 3),
               "area_range_coverage": f"{covered}/{n_cov}",
               "wall_range_coverage": f"{sum(wall_in)}/{len(wall_in)}",
               "median_top_wall_err_pct": round(float(np.median(wall_err)) * 100, 1) if wall_err else None,
               "runtime_s": plan["meta"]["runtime_s"]}
    summary["footprint_err_pct"] = round((summary["footprint_photo_m2"] / summary["footprint_lidar_m2"] - 1) * 100, 1)
    (out / "eval.json").write_text(json.dumps({"rooms": rows, "summary": summary}, indent=2))
    print(f"{'room':>8} {'photo m2':>9} {'lidar m2':>9} {'err %':>7} {'in range':>9}  top-wall err %")
    for x in rows:
        print(f"{x['room']:>8} {x['photo_area']:>9.2f} {x['lidar_area']:>9.2f} {x['area_err_pct']:>7.1f} "
              f"{str(x['lidar_in_range']):>9}  {x['top_wall_err_pct']}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
