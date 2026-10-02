"""Detected damage vs the tape-measured staged damage.

usage: python scripts/eval_damage.py plan.json data/ground_truth/own_rooms.csv room1=<plan room id>
(the plan room holding the staged damage, read off the rendered plan)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.score_benchmark import load_ground_truth   # noqa: E402
from roomscan.score_damage import score_damage            # noqa: E402


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    gt = load_ground_truth(sys.argv[2])
    mapping = dict(a.split("=", 1) for a in sys.argv[3:])
    print(json.dumps(score_damage(plan, gt, mapping), indent=2))


if __name__ == "__main__":
    main()
