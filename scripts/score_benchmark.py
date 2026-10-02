"""Score a plan.json against tape ground truth.

usage: python scripts/score_benchmark.py plan.json data/ground_truth/own_rooms.csv room1=room_0 room2=room_3 ...
(room mapping: ground-truth room name = plan room id, read off the rendered plan)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.score_benchmark import load_ground_truth, score_plan   # noqa: E402


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    gt = load_ground_truth(sys.argv[2])
    mapping = dict(a.split("=", 1) for a in sys.argv[3:])
    print(json.dumps(score_plan(plan, gt, mapping), indent=2))


if __name__ == "__main__":
    main()
