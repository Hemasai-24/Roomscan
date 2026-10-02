"""Drift correction on vs off on one capture: plan pictures + table (brief: drift accountability)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.drift_correction import correct_drift, wall_sharpness   # noqa: E402
from roomscan.draw_plan import render                                  # noqa: E402
from roomscan.find_surfaces import classify_planes, extract_planes     # noqa: E402
from roomscan.load_capture import load_stray                           # noqa: E402
from roomscan.pipeline import run_lidar                                # noqa: E402
from roomscan.point_cloud import estimate_normals, fuse_points         # noqa: E402


def sharpness(cap):
    pts = fuse_points(cap)
    nrm = estimate_normals(pts)
    return wall_sharpness(pts, nrm, classify_planes(extract_planes(pts, nrm))["walls"])


def main():
    cap_dir = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"outputs/ablation/{cap_dir.name}")
    rows = {}
    for mode in ("off", "on"):
        plan = run_lidar(cap_dir, drift_correction=(mode == "on"))
        (out / mode).mkdir(parents=True, exist_ok=True)
        (out / mode / "plan.json").write_text(json.dumps(plan, indent=2))
        render(plan, out / mode / "plan")
        cap = load_stray(cap_dir)
        if mode == "on":
            cap, _ = correct_drift(cap)
        rows[mode] = {"wall_sharpness_mm": round(sharpness(cap) * 1000, 2),
                      "total_floor_area_m2": round(sum(r["floor_area"]["value"] for r in plan["rooms"]), 2),
                      "rooms": len(plan["rooms"]), "drift": plan["meta"].get("drift")}
    (out / "ablation.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
