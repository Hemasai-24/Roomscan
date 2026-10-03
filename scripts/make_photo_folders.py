"""Simulated photo-tier input from a LiDAR sample capture (no tape ground truth exists for the sample).

The LiDAR pipeline splits the capture into rooms; for each room the "photos" are frames of the colour video
taken from inside it, one per viewing direction (looking across the room, sharp), plus one facing each
detected door — what the capture protocol asks a person to do. Only JPEGs go into the folders (no depth, no
poses); the true lens focal length is written to EXIF the way a phone does. The LiDAR plan is saved next to
the folders as the reference.

Usage: python scripts/make_photo_folders.py <stray_capture_dir> <out_dir>
Writes <out_dir>/rooms/<room_id>/*.jpg and <out_dir>/lidar/plan.json."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.load_capture import load_stray                     # noqa: E402
from roomscan.photo_folders import EXIF_IFD, FOCAL_35MM, FULL_FRAME_DIAGONAL_MM   # noqa: E402
from roomscan.photo_sim import pick_door_view, pick_doorway_view, pick_room_views   # noqa: E402
from roomscan.pipeline import run_lidar                          # noqa: E402
from roomscan.room_outline import to_plan                        # noqa: E402
from roomscan.save_plan import validate                          # noqa: E402
from roomscan.stitch_rooms import door_frames                    # noqa: E402
from roomscan.video_frames import read_frames, sharpness         # noqa: E402

W, H = 720, 960          # upright half-resolution photos
MAX_PHOTOS = 8


def main():
    cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    ref_path = out / "lidar" / "plan.json"
    if ref_path.exists():
        plan = json.loads(ref_path.read_text())
    else:
        plan = run_lidar(cap_dir)
        validate(plan)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(plan, indent=2))
    ang = plan["meta"]["manhattan_angle_rad"]
    cap = load_stray(cap_dir)
    cams = to_plan(np.array([f.T_wc[:3, 3] for f in cap.frames]), ang)
    fwd = to_plan(np.array([f.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0]) for f in cap.frames]), ang)
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    sharp = np.zeros(len(cap.frames))
    for i, img in read_frames(cap_dir / "rgb.mp4", (W, H), rotate=90):
        if i < len(sharp):
            sharp[i] = sharpness(img)
    depth = np.zeros(len(cap.frames))
    for k, f in enumerate(cap.frames):
        d = cap.load_depth(f.index)
        v = d[d > 0]
        depth[k] = np.median(v) if len(v) > 100 else 0.0
    picks = {}
    for room in plan["rooms"]:
        poly = Polygon(room["polygon"])
        idx = np.array([k for k, p in enumerate(cams) if poly.contains(Point(p))])
        if len(idx) < 2:
            print(f"{room['id']}: camera never inside ({len(idx)} frames) - no folder")
            continue
        sel = pick_room_views(idx, yaw, sharp, depth)
        for door in door_frames(room):
            k = pick_door_view(idx, cams, fwd, sharp, door["mid"])
            if k is not None and k not in sel:
                sel.append(k)
        picks[room["id"]] = sorted(sel)[:MAX_PHOTOS] if len(sel) > MAX_PHOTOS else sorted(sel)
        print(f"{room['id']}: {len(idx)} frames inside, {len(picks[room['id']])} photos")
    if "--doorway-shared" in sys.argv:          # experimental protocol: one doorway photo in both rooms' folders
        rooms = {r["id"]: r for r in plan["rooms"]}
        shared = {rid: [] for rid in picks}
        for a in plan["adjacency"]:
            if a["room_a"] not in picks or a["room_b"] not in picks:
                continue
            door = next((d for d in door_frames(rooms[a["room_a"]]) if d["id"] == a["via"]), None)
            if door is None:
                continue
            k = pick_doorway_view(np.arange(len(cams)), cams, fwd, sharp, door["mid"])
            if k is None:
                print(f"{a['room_a']}-{a['room_b']}: no frame in the doorway")
                continue
            for rid in (a["room_a"], a["room_b"]):
                shared[rid].append(k)
            print(f"{a['room_a']}-{a['room_b']}: doorway photo {k} shared")
        for rid, ks in shared.items():                  # doorway photos first; room views fill the rest
            ks = sorted(set(ks))[:MAX_PHOTOS]
            picks[rid] = ks + [k for k in picks[rid] if k not in ks][:MAX_PHOTOS - len(ks)]
    fx_rgb = cap.frames[0].K[0, 0] * 1920 / cap.depth_size[0]           # focal at 1920 px (landscape width)
    f35 = fx_rgb * (W / 1440) * FULL_FRAME_DIAGONAL_MM / np.hypot(W, H)  # upright 1440-wide frame -> W
    need = {}
    for rid, ks in picks.items():
        for k in ks:
            need.setdefault(k, []).append(rid)        # a shared doorway photo goes to two rooms
    for i, img in read_frames(cap_dir / "rgb.mp4", (W, H), rotate=90):
        for rid in need.get(i, []):
            d = out / "rooms" / rid
            d.mkdir(parents=True, exist_ok=True)
            ex = Image.Exif()
            ex.get_ifd(EXIF_IFD)[FOCAL_35MM] = int(round(f35))
            Image.fromarray(img).save(d / f"frame_{i:06d}.jpg", quality=92, exif=ex)
    (out / "selection.json").write_text(json.dumps({"focal_35mm": round(float(f35), 2), "photos": picks}, indent=2))


if __name__ == "__main__":
    main()
