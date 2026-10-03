"""Video tier vs LiDAR tier on the same Stray Scanner capture (the LiDAR plan is the reference here, since
the sample data has no tape ground truth).

Usage: python scripts/eval_video_vs_lidar.py <stray_capture_dir> [--rotate 90]
Writes outputs/eval_video/<name>/{lidar,video}/plan.json and outputs/eval_video/<name>.json, prints a table."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.compare_plans import align_partial, match_walls   # noqa: E402
from roomscan.draw_plan import render                          # noqa: E402
from roomscan.pipeline import run_lidar                        # noqa: E402
from roomscan.video_pipeline import run_video                  # noqa: E402


def evaluate(L, V, name):
    """Video plan V vs LiDAR reference plan L: (summary, matched wall rows)."""
    R2, t = align_partial(L, V)
    pairs = match_walls(V, L, np.linalg.inv(R2), -t @ np.linalg.inv(R2).T, max_mid=0.4)
    vw = {w["id"]: w for r in V["rooms"] for w in r["walls"]}
    rows = []
    for m in pairs:
        w = vw[m["a"]]
        lv, ll = m["len_a"], m["len_b"]
        rows.append({"video_wall": m["a"], "lidar_wall": m["b"], "video_m": round(lv, 3), "lidar_m": round(ll, 3),
                     "err_pct": round(100 * (lv - ll) / ll, 1),
                     "lidar_inside_video_range": bool(w["length"]["lo"] <= ll <= w["length"]["hi"])})
    vpoly = unary_union([Polygon(r["polygon"]) for r in V["rooms"]])
    vpoly_in_l = Polygon(np.array(vpoly.exterior.coords) @ R2.T + t) if vpoly.geom_type == "Polygon" else None
    lpoly = unary_union([Polygon(r["polygon"]) for r in L["rooms"]])
    iou = None if vpoly_in_l is None else round(vpoly_in_l.intersection(lpoly).area / vpoly_in_l.union(lpoly).area, 3)
    long_rows = [r for r in rows if r["lidar_m"] >= 1.0]
    summary = {"capture": name, "video_rooms": len(V["rooms"]), "lidar_rooms": len(L["rooms"]),
               "video_area_m2": round(vpoly.area, 2), "lidar_area_m2": round(lpoly.area, 2),
               "matched_walls": len(rows), "matched_walls_ge_1m": len(long_rows),
               "median_abs_err_pct_ge_1m": round(float(np.median([abs(r["err_pct"]) for r in long_rows])), 1) if long_rows else None,
               "within_3pct_ge_1m": sum(abs(r["err_pct"]) <= 3 for r in long_rows),
               "lidar_inside_video_range": f"{sum(r['lidar_inside_video_range'] for r in rows)}/{len(rows)}",
               "video_footprint_iou_with_lidar": iou, "video_meta": V["meta"].get("video"), "walls": rows}
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--rotate", type=int, default=90)
    a = ap.parse_args()
    name = a.capture.parent.name
    out = Path("outputs/eval_video") / name
    plans = {}
    for tier in ("lidar", "video"):
        d = out / tier
        d.mkdir(parents=True, exist_ok=True)
        p = run_lidar(a.capture) if tier == "lidar" else run_video(a.capture / "rgb.mp4", d, rotate=a.rotate)
        (d / "plan.json").write_text(json.dumps(p, indent=2))
        render(p, d / "plan")
        plans[tier] = p
    L, V = plans["lidar"], plans["video"]
    summary, rows = evaluate(L, V, name)
    (out.parent / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"{'video wall':>12} {'lidar wall':>12} {'video m':>8} {'lidar m':>8} {'err %':>6} in-range")
    for r in sorted(rows, key=lambda r: -r["lidar_m"]):
        print(f"{r['video_wall']:>12} {r['lidar_wall']:>12} {r['video_m']:8.3f} {r['lidar_m']:8.3f} {r['err_pct']:6.1f} "
              f"{'yes' if r['lidar_inside_video_range'] else 'NO'}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("walls", "video_meta")}, indent=1))


if __name__ == "__main__":
    main()
