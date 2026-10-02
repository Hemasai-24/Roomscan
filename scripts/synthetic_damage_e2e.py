"""End-to-end damage check with KNOWN truth on a real capture: paint a 0.40 x 0.30 m stain and a 0.60 m crack
onto a real wall in 3D, project them into every view the pipeline will use (true pose, LiDAR-depth occlusion
test), then run the full LiDAR pipeline with damage on and compare what it measures with the truth.

usage: python scripts/synthetic_damage_e2e.py <stray capture> [out_dir]"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.damage_pipeline import pick_view_indices                   # noqa: E402
from roomscan.find_surfaces import classify_planes, extract_planes       # noqa: E402
from roomscan.load_capture import load_stray                             # noqa: E402
from roomscan.pipeline import run_lidar                                  # noqa: E402
from roomscan.point_cloud import estimate_normals, fuse_points           # noqa: E402
from roomscan.video_frames import read_frames                            # noqa: E402

STAIN = (0.40, 0.30)       # m, width x height
CRACK = 0.60               # m, vertical
SIZE = (1280, 960)         # same as roomscan.damage_pipeline.DETECT_WIDTH
UP = np.array([0.0, 1.0, 0.0])


def plan_to_world(q, angle, y):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * q[0] - s * q[1], y, s * q[0] + c * q[1]])   # inverse of room_outline.to_plan


HEIGHT = 0.9               # m above the floor (the sample scans look mostly down)


def candidates(walls, floor_h):
    """Spots on real fitted wall planes: every 0.5 m along each wall >= 1.6 m wide, at 0.6-1.5 m height;
    stain at the spot, crack 1 m further along."""
    for k, w in enumerate(walls):
        n = np.asarray(w.normal, float)
        t = np.cross(UP, n)
        t /= np.linalg.norm(t)
        along = w.inliers @ t
        if along.max() - along.min() < 1.6:
            continue
        base = w.inliers.mean(0)
        base = base - n * (base @ n + w.d)                        # exactly on the plane
        for s in np.arange(along.min() + 0.3, along.max() - 1.3, 0.5):
            for hgt in (0.6, 0.9, 1.2, 1.5):
                p = base + t * (s - base @ t)
                p[1] = floor_h + hgt
                yield k, t, p, p + t * 1.0


def pick_wall(walls, floor_h, cap, idx):
    """The wall where the stain and crack positions are each visible (depth agrees) in the most views."""
    frames = {f.index: f for f in cap.frames}
    depths = {i: cap.load_depth(i, min_conf=0) for i in idx}
    best, best_n = None, -1
    for cand in candidates(walls, floor_h):
        _, t, sc, cc = cand
        ns = [0, 0]
        for i in idx:
            f, d = frames[i], depths[i]
            for j, p in enumerate((sc, cc)):
                uv, z = project(p[None], f.T_wc, f.K)
                ns[j] += int(visible(uv, z, d, f.K, 1.0).all())
        if min(ns) > best_n:
            best, best_n = cand, min(ns)
    return best


def project(P, T, K):
    pc = (P - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    uv = (pc[:, :2] / np.maximum(z[:, None], 1e-6)) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def visible(uv, z, depth, K_d, scale):
    ok = []
    for (u, v), zz in zip(uv, z):
        ud, vd = int(u / scale), int(v / scale)
        if zz <= 0.2 or not (0 <= ud < depth.shape[1] and 0 <= vd < depth.shape[0]):
            ok.append(False)
            continue
        d = depth[vd, ud]
        ok.append(d == 0 or abs(d - zz) < 0.10)
    return np.array(ok)


def paint(img, frame, depth, t, stain_c, crack_c, rng):
    scale = SIZE[0] / depth.shape[1]
    K = frame.K.copy()
    K[:2] *= scale
    T = frame.T_wc
    hit = {"stain": False, "crack": False}
    w, h = STAIN
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    blob = np.array([stain_c + t * (w / 2) * np.cos(a) * (1 + 0.08 * np.sin(3 * a)) + UP * (h / 2) * np.sin(a)
                     for a in ang])
    uv, z = project(blob, T, K)
    vis = visible(uv, z, depth, frame.K, scale)
    if vis.mean() > 0.8:
        m = np.zeros(img.shape[:2], np.float32)
        cv2.fillPoly(m, [uv.astype(np.int32)], 1.0)
        m = cv2.GaussianBlur(m, (9, 9), 0)[..., None]
        img = (img * (1 - 0.65 * m) + np.array([105, 75, 40]) * 0.65 * m).astype(np.uint8)
        hit["stain"] = True
    ys = np.linspace(-CRACK / 2, CRACK / 2, 25)
    zig = np.random.default_rng(7).uniform(-0.025, 0.025, len(ys)) * np.resize([1, -1], len(ys))
    line = np.array([crack_c + UP * y + t * z for y, z in zip(ys, zig)])       # jagged, like a real crack
    uv, z = project(line, T, K)
    vis = visible(uv, z, depth, frame.K, scale)
    if vis.mean() > 0.8:
        px = max(2, int(round(0.006 * K[0, 0] / max(float(np.median(z)), 0.3))))
        cv2.polylines(img, [uv.astype(np.int32)], False, (35, 35, 35), px)
        hit["crack"] = True
    return img, hit


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"outputs/synthetic_damage/{src.name}")
    cap_copy = out / "capture"
    if cap_copy.exists():
        shutil.rmtree(cap_copy)
    cap_copy.mkdir(parents=True)
    for name in ("depth", "confidence"):
        (cap_copy / name).symlink_to((src / name).resolve())
    for name in ("odometry.csv", "camera_matrix.csv"):
        shutil.copy(src / name, cap_copy / name)
    cap = load_stray(src)
    base = run_lidar(src, damage=False)
    pts = fuse_points(cap)
    classes = classify_planes(extract_planes(pts, estimate_normals(pts)))
    floor_h = classes["floor"].height
    idx = pick_view_indices(cap)
    wall_k, t, stain_c, crack_c = pick_wall(classes["walls"], floor_h, cap, idx)
    room_id, wall_id = "(any)", f"fitted wall plane #{wall_k}"
    frames = {f.index: f for f in cap.frames}
    (cap_copy / "rgb").mkdir()
    imgs = {i: im for i, im in read_frames(src / "rgb.mp4", SIZE) if i in set(idx)}
    rng = np.random.default_rng(0)
    seen = {"stain": 0, "crack": 0}
    for i in idx:
        img, hit = paint(imgs[i].copy(), frames[i], cap.load_depth(i, min_conf=0), t, stain_c, crack_c, rng)
        for k in seen:
            seen[k] += hit[k]
        cv2.imwrite(str(cap_copy / "rgb" / f"{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plan = run_lidar(cap_copy, damage=True, capture_id=src.name + "_painted")
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    from roomscan.draw_plan import render
    render(plan, out / "plan")
    report = {"truth": {"room": room_id, "wall": wall_id, "stain_m": STAIN, "stain_area_m2": round(np.pi / 4 * STAIN[0] * STAIN[1], 4),
                        "crack_m": CRACK, "views_showing": seen, "views_total": len(idx)},
              "found": [{k: d[k] for k in ("class", "surface_id", "n_views", "score", "area", "width", "height",
                                           "bottom_above_floor")} for d in plan["damage"]],
              "candidates": plan["meta"].get("damage_candidates"),
              "flags": [f["rule_id"] for f in plan["concealed_damage_flags"]],
              "scope": [(s["action"], s["surface_id"]) for s in plan["scope_items"]]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
