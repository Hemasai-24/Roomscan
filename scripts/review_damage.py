"""Damage detector review on frames of a capture with NO real damage (false alarms), plus a synthetic
positive (a brown stain and a dark jagged line painted onto a wall frame).

usage: python scripts/review_damage.py <stray capture> [n_frames] [out_dir]
Writes a contact sheet PNG with every detection drawn and prints counts per score threshold."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.damage_detect import get_detector, postprocess, upright_turns   # noqa: E402
from roomscan.load_capture import load_stray                                  # noqa: E402
from roomscan.video_frames import read_frames                                 # noqa: E402

THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
COLORS = {"water_stain": (0, 140, 255), "crack": (0, 0, 255), "mold": (0, 200, 0), "peeling_paint": (255, 0, 255),
          "hole": (255, 255, 0)}


def paint_damage(img, seed=0):
    """Synthetic positive: brown stain blob upper left, dark jagged crack on the right."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    stain = np.zeros((h, w), np.float32)
    cx, cy = int(w * 0.3), int(h * 0.35)
    for _ in range(12):
        c = (cx + int(rng.normal(0, w * 0.03)), cy + int(rng.normal(0, h * 0.03)))
        cv2.circle(stain, c, int(rng.uniform(w * 0.03, w * 0.06)), 1.0, -1)
    stain = cv2.GaussianBlur(stain, (31, 31), 0)[..., None]
    brown = np.array([110, 80, 45], np.float32)            # RGB
    out = (out * (1 - 0.6 * stain) + brown * 0.6 * stain).astype(np.uint8)
    x, y = int(w * 0.7), int(h * 0.2)
    pts = [(x, y)]
    for _ in range(14):
        x += int(rng.integers(-8, 9))
        y += int(h * 0.04)
        pts.append((x, y))
    cv2.polylines(out, [np.array(pts, np.int32)], False, (30, 30, 30), 3)
    return out, {"stain_box": (int(w * 0.18), int(h * 0.22), int(w * 0.42), int(h * 0.48)),
                 "crack_box": (min(p[0] for p in pts), pts[0][1], max(p[0] for p in pts), pts[-1][1])}


def draw(img, dets):
    vis = img.copy()
    for d in dets:
        col = COLORS[d["class"]]
        vis[d["mask"]] = (0.5 * vis[d["mask"]] + 0.5 * np.array(col)).astype(np.uint8)
        x0, y0, x1, y1 = d["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, f"{d['class']} {d['score']:.2f}", (x0, max(12, y0 - 3)), 0, 0.4, col, 1)
    return vis


def main():
    cap_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "outputs/damage_review")
    out.mkdir(parents=True, exist_ok=True)
    cap = load_stray(cap_dir)
    want = set(np.linspace(10, len(cap.frames) - 10, n).astype(int).tolist())
    frames = {i: img for i, img in read_frames(cap_dir / "rgb.mp4", (640, 480)) if i in want}
    det = get_detector()
    raw_all, tiles, t_sum = [], [], 0.0
    for i in sorted(frames):
        k = upright_turns(cap.frames[i].T_wc)
        img = np.ascontiguousarray(np.rot90(frames[i], k))
        t = time.time()
        raw = det(img)
        t_sum += time.time() - t
        raw_all.append((img, raw, i))
    counts = {th: sum(len(postprocess(r, im.shape, th)) for im, r, _ in raw_all) for th in THRESHOLDS}
    by_class = {}
    for im, r, _ in raw_all:
        for d in postprocess(r, im.shape, 0.35):
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
    for im, r, _ in raw_all:
        tiles.append(cv2.resize(draw(im, postprocess(r, im.shape, THRESHOLDS[0])), (240, 320)
                                if im.shape[0] > im.shape[1] else (320, 240)))
    # synthetic positive on a wall-ish frame (the frame with the fewest raw detections)
    im0, _, frame0 = min(raw_all, key=lambda x: len(x[1]))
    painted, truth = paint_damage(im0)
    pos = det(painted)
    pos_counts = {th: [(d["class"], round(d["score"], 2), d["box"]) for d in postprocess(pos, painted.shape, th)]
                  for th in THRESHOLDS}
    cv2.imwrite(str(out / "synthetic_positive.png"), cv2.cvtColor(draw(painted, postprocess(pos, painted.shape,
                                                                                              THRESHOLDS[0])),
                                                                   cv2.COLOR_RGB2BGR))
    rows = [np.hstack([cv2.resize(t, (240, 320)) for t in tiles[j:j + 5]]) for j in range(0, len(tiles) - 4, 5)]
    if rows:
        cv2.imwrite(str(out / f"false_alarms_{cap_dir.name}.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    report = {"capture": str(cap_dir), "frames": len(raw_all), "seconds_per_frame": round(t_sum / len(raw_all), 2),
              "false_alarms_by_threshold": counts, "false_alarms_by_class_at_0.35": by_class,
              "synthetic_frame": int(frame0), "synthetic_truth": truth, "synthetic_detections": pos_counts}
    (out / f"review_{cap_dir.name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
