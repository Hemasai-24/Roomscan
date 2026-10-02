"""One command per capture: python run.py <Stray folder | video file> [--out DIR] [--tier auto|lidar|video]"""
import argparse
import json
from pathlib import Path

from roomscan.save_plan import validate
from roomscan.pipeline import detect_tier, run_lidar
from roomscan.draw_plan import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path, help="Stray Scanner export folder, or a walkthrough video")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drift-correction", choices=["on", "off"], default="off",
                    help="re-align the walk so walls seen twice coincide (default off: see docs/TRADEOFFS.md)")
    ap.add_argument("--tier", choices=["auto", "lidar", "video"], default="auto")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="video only: clockwise turn to make frames upright (Stray Scanner rgb.mp4 needs 90)")
    a = ap.parse_args()
    tier, path = detect_tier(a.capture_dir)
    if a.tier == "video" and tier == "lidar":
        tier, path = "video", path / "rgb.mp4"          # use only the colour video of a Stray capture
    out = a.out or Path("outputs") / (path.stem if tier == "video" else path.name)
    out.mkdir(parents=True, exist_ok=True)
    if tier == "video":
        from roomscan.video_pipeline import run_video
        plan = run_video(path, out, rotate=a.rotate)
    else:
        plan = run_lidar(path, drift_correction=(a.drift_correction == "on"))
    validate(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    render(plan, out / "plan")
    for r in plan["rooms"]:
        print(f"{r['id']}: area {r['floor_area']['value']:.2f} m², ceiling {r['ceiling_height']['value']:.3f} m, "
              f"{len(r['walls'])} walls, {len(r['openings'])} openings")
    print(f"wrote {out}/plan.json, plan.svg, plan.png in {plan['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
