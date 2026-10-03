"""Experiment: room-shape method alone. Re-measure rooms already reconstructed in an oracle run (default: oracle
"all" = LiDAR depth + poses of the photos) with each room-extent method, and compare per-room area and longest
walls with the LiDAR reference rooms (same ids).

usage: python scripts/exp_photo_extent.py <photo_out_dir> [oracle_mode]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.photo_room import measure_photo_room   # noqa: E402


def top_walls(room, k=4):
    return sorted((w["length"]["value"] for w in room["walls"]), reverse=True)[:k]


def main():
    out = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    ref = {r["id"]: r for r in json.loads((out / "lidar" / "plan.json").read_text())["rooms"]}
    res = {}
    for ext in sys.argv[3:] or ("floor", "walls", "core"):
        area_err, wall_err = [], []
        for d in sorted((out / f"oracle_{mode}" / "photo_rooms").iterdir()):
            if d.name not in ref:
                continue
            r = measure_photo_room(d, d.name, extent=ext)["room"]
            L = ref[d.name]
            area_err.append(r["floor_area"]["value"] / L["floor_area"]["value"] - 1)
            pw, lw = top_walls(r), top_walls(L)
            k = min(len(pw), len(lw))
            wall_err += [abs(p / q - 1) for p, q in zip(pw[:k], lw[:k])]
        res[ext] = {"rooms": len(area_err), "median_abs_area_err_pct": round(100 * float(np.median(np.abs(area_err))), 1),
                    "median_area_err_pct": round(100 * float(np.median(area_err)), 1),
                    "median_top_wall_err_pct": round(100 * float(np.median(wall_err)), 1)}
        print(mode, ext, json.dumps(res[ext]), flush=True)
    (out / f"extent_{mode}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
