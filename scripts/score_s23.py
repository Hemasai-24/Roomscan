"""Score every run of our own S23 benchmark against the tape measurements.

usage: python scripts/score_s23.py <runs_dir> data/ground_truth/own_rooms.csv data/ground_truth/room_map.json
runs_dir holds one folder per run (photo/, video_<name>/) each with plan.json; room_map.json maps, per run,
ground-truth room names to plan room ids (read off each rendered plan), e.g.
{"photo": {"room1": "room1"}, "video_walkthrough": {"room1": "room_2", "room2": "room_0"}}.
Also compares video_room1_take1 vs video_room1_take2 (repeatability) when both exist.
Writes <runs_dir>/score.json and prints a table."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.compare_plans import align_plans, match_walls   # noqa: E402
from roomscan.score_benchmark import load_ground_truth, score_plan   # noqa: E402


def score_all(runs_dir, gt_csv, room_map_json):
    runs_dir = Path(runs_dir)
    gt = load_ground_truth(gt_csv)
    rmap = json.loads(Path(room_map_json).read_text())
    out = {"runs": {}}
    for name, mapping in rmap.items():
        p = runs_dir / name / "plan.json"
        if p.exists():
            out["runs"][name] = score_plan(json.loads(p.read_text()), gt, mapping)
    a, b = runs_dir / "video_room1_take1" / "plan.json", runs_dir / "video_room1_take2" / "plan.json"
    if a.exists() and b.exists():
        pa, pb = json.loads(a.read_text()), json.loads(b.read_text())
        R2, t = align_plans(pa, pb)
        rows = match_walls(pa, pb, R2, t)
        out["repeatability_video_room1"] = {"walls_matched": len(rows), "within_gate": sum(r["pass"] for r in rows),
                                            "pairs": rows}
    return out


def main():
    res = score_all(*sys.argv[1:4])
    (Path(sys.argv[1]) / "score.json").write_text(json.dumps(res, indent=2))
    print(f"{'run':24} {'tier':6} {'walls':>5} {'max err':>8} {'mean err':>8} {'coverage':>8} {'ceiling err':>11} {'openings':>9}")
    for name, s in res["runs"].items():
        w, c, o = s["walls"], s["ceiling"], s["openings"]
        f = lambda v, k=100: "-" if v is None else f"{v * k:.1f}"
        print(f"{name:24} {s['tier']:6} {w['n']:>5} {f(w['max_abs_err_m']):>7}cm {f(w['mean_abs_err_m']):>7}cm "
              f"{f(w['coverage'], 100):>7}% {f(c['abs_err_m']):>10}cm {o['within_2cm']}/{o['n_truth'] + o['phantom']:>7}")
    if "repeatability_video_room1" in res:
        r = res["repeatability_video_room1"]
        print(f"repeatability (video, room 1 twice): {r['within_gate']}/{r['walls_matched']} walls within 1 cm / 0.5%")


if __name__ == "__main__":
    main()
