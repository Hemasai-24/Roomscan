"""Experiment: video-tier back-end settings on a capture whose poses and depth are perfect (oracle "all").
Isolates what the back end itself costs: the walkable-camera-path widening and the frame stride.

usage: python scripts/exp_video_backend.py <oracle_all_capture_dir> <lidar_plan.json> <name>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_video_vs_lidar import evaluate      # noqa: E402
from roomscan.pipeline import run_lidar       # noqa: E402

KEYS = ("video_rooms", "video_area_m2", "lidar_area_m2", "matched_walls_ge_1m", "median_abs_err_pct_ge_1m",
        "within_3pct_ge_1m", "video_footprint_iou_with_lidar")


def main():
    cap, L, name = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), sys.argv[3]
    out = {}
    for wp in (True, False):
        V = run_lidar(cap, tier="video", stride=1, walkable_path=wp)
        s, _ = evaluate(L, V, name)
        out[f"walkable_path={wp}"] = {k: s[k] for k in KEYS}
        print(f"walkable_path={wp}", json.dumps(out[f"walkable_path={wp}"]), flush=True)
    (cap.parent / "backend_ablation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
