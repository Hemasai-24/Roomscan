"""Damage detector on public photos of real damage (Wikimedia Commons; detection only - no depth, no size).

usage: python scripts/fetch_public_damage.sh && python scripts/eval_public_damage.py
Each image has an expected set of acceptable classes; reports detections, hits and misses."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.damage_detect import detect_damage   # noqa: E402

DIR = Path(__file__).resolve().parents[1] / "data" / "public_damage"
EXPECTED = {   # image -> acceptable classes (what a human would call it)
    "01_crack_in_wall.jpg": {"crack"}, "03_crack_in_wall.jpg": {"crack"}, "04_crack_in_wall.jpg": {"crack"},
    "05_water_damage_ceiling.jpg": {"water_stain", "peeling_paint"}, "06_water_damage_ceiling.jpg": {"water_stain"},
    "07_water_damage_ceiling.jpg": {"crack", "water_stain"}, "09_peeling_paint_wall.jpg": {"peeling_paint"},
    "10_peeling_paint_wall.jpg": {"peeling_paint"}, "11_peeling_paint_wall.jpg": {"hole", "peeling_paint"},
    "12_peeling_paint_wall.jpg": {"hole", "peeling_paint"},
}


def main():
    rows, hits = [], 0
    for name, want in EXPECTED.items():
        img = np.asarray(Image.open(DIR / name).convert("RGB"))
        found = detect_damage(img)
        classes = sorted({d["class"] for d in found})
        hit = bool(want & set(classes))
        hits += hit
        rows.append({"image": name, "expected": sorted(want), "detected": classes,
                     "scores": [round(d["score"], 2) for d in found], "hit": hit})
        print(f"{name:32} expected {sorted(want)!s:28} detected {classes!s:40} {'HIT' if hit else 'MISS'}")
    print(f"recall: {hits}/{len(EXPECTED)} images")
    (DIR / "eval.json").write_text(json.dumps({"recall": f"{hits}/{len(EXPECTED)}", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
