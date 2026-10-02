"""One command per capture: python run.py <capture_dir> [--out DIR]"""
import argparse
import json
from pathlib import Path

from roomscan.load_capture import is_stray
from roomscan.save_plan import validate
from roomscan.pipeline import run_lidar
from roomscan.draw_plan import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path("outputs") / a.capture_dir.name
    out.mkdir(parents=True, exist_ok=True)
    if not is_stray(a.capture_dir):
        raise SystemExit(f"{a.capture_dir}: not a Stray Scanner capture (video/photo tiers: Plans 3-4)")
    plan = run_lidar(a.capture_dir)
    validate(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    render(plan, out / "plan")
    for r in plan["rooms"]:
        print(f"{r['id']}: area {r['floor_area']['value']:.2f} m², ceiling {r['ceiling_height']['value']:.3f} m, "
              f"{len(r['walls'])} walls, {len(r['openings'])} openings")
    print(f"wrote {out}/plan.json, plan.svg, plan.png in {plan['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
