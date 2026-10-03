"""Print every experiment table found under outputs/exp as markdown (for docs/experiments/tier_accuracy.md)."""
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "outputs" / "exp"


def photo_rows(name):
    rows = []
    for d in sorted(E.glob(f"{name}/oracle_*/eval.json")):
        s = json.loads(d.read_text())["summary"]
        rows.append((d.parent.name.replace("oracle_", ""), s["median_top_wall_err_pct"], s["footprint_err_pct"],
                     s["footprint_iou_after_alignment"], s["adjacency_precision"], s["adjacency_recall"],
                     s["max_overlap_m2"], s["wall_range_coverage"]))
    return rows


def video_rows(name):
    rows = []
    for d in sorted(E.glob(f"{name}/oracle_*/eval.json")):
        s = json.loads(d.read_text())["summary"]
        v = json.loads((d.parent / "plan.json").read_text())["meta"].get("video", {})
        rows.append((d.parent.name.replace("oracle_", ""), s["median_abs_err_pct_ge_1m"], s["within_3pct_ge_1m"],
                     s["matched_walls_ge_1m"], s["video_area_m2"], s["lidar_area_m2"],
                     s["video_footprint_iou_with_lidar"], v.get("frames_used"), v.get("da_scale_err_pct")))
    return rows


def main():
    for name in ("photo_floor_only", "photo_with_ceiling", "photo_floor_only_door"):
        print(f"\n### {name}\n| mode | median wall err % | footprint err % | footprint IoU | adj. precision | adj. recall "
              f"| max overlap m2 | wall range coverage |\n|---|---|---|---|---|---|---|---|")
        for r in photo_rows(name):
            print("| " + " | ".join(str(x) for x in r) + " |")
    for name in ("video_single_room", "video_floor_only"):
        print(f"\n### {name}\n| mode | median wall err % (>=1 m) | walls within 3 % | matched walls | video m2 | lidar m2 "
              f"| footprint IoU | frames used | DA scale err % |\n|---|---|---|---|---|---|---|---|---|")
        for r in video_rows(name):
            print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
