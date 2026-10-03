"""Experiment: how much of the photo tier's whole-property error is the stitching (placing rooms)?
Rooms of an oracle "all" run live in the LiDAR world frame, so placing each at its TRUE position (no stitching)
gives the footprint the room-shape method alone would produce. Compared with the stitched plan of the same run.

usage: python scripts/exp_photo_stitch_oracle.py <photo_out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.photo_room import measure_photo_room   # noqa: E402


def plan_to_world(poly, angle):
    """Inverse of room_outline.to_plan for 2D points (u, v) -> world (x, z)."""
    c, s = np.cos(angle), np.sin(angle)
    P = np.asarray(poly, float)
    return np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]]


def main():
    out = Path(sys.argv[1])
    ref = json.loads((out / "lidar" / "plan.json").read_text())
    g = ref["meta"]["manhattan_angle_rad"]
    lid = unary_union([Polygon(r["polygon"]).buffer(0) for r in ref["rooms"]])
    polys = []
    for d in sorted((out / "oracle_all" / "photo_rooms").iterdir()):
        m = measure_photo_room(d, d.name)
        world = plan_to_world(m["room"]["polygon"], m["angle"])
        polys.append(Polygon(plan_to_world(world, -g)).buffer(0))      # world -> LiDAR plan frame
    true_pos = unary_union(polys)
    stitched = json.loads((out / "oracle_all" / "eval.json").read_text())["summary"]
    res = {"rooms": len(polys), "footprint_true_positions_m2": round(true_pos.area, 2),
           "footprint_lidar_m2": round(lid.area, 2),
           "iou_true_positions": round(true_pos.intersection(lid).area / true_pos.union(lid).area, 3),
           "iou_stitched_after_alignment": stitched["footprint_iou_after_alignment"],
           "overlap_sum_true_positions_m2": round(sum(p.area for p in polys) - true_pos.area, 2)}
    print(json.dumps(res))
    (out / "stitch_oracle.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
