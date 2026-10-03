"""Photo tier end to end: per-room photo folders -> each room reconstructed and measured on its own
(roomscan/photo_room.py) -> rooms linked by photos that see each other (photo_links.py) -> stitched into one
whole-property plan at shared doors (stitch_rooms.py), with tier="photo" ranges plus metric-scale uncertainty."""
import json
import time
from pathlib import Path

import numpy as np

from roomscan.find_surfaces import CaptureError
from roomscan.photo_folders import load_room_images, room_folders
from roomscan.photo_links import room_links
from roomscan.save_plan import build_plan
from roomscan.stitch_rooms import stitch
from roomscan.video_scale import CALIBRATION, KEY

MIN_PHOTOS, MAX_PHOTOS = 2, 8          # VGGT fits 8 photos at 392x518 in 8 GB
ROOT = Path(__file__).resolve().parents[1]


def scale_rel_sigma(per_frame_spread, n_photos, path=CALIBRATION):
    """Relative std of a room's metric scale: calibration error between captures (leave-one-out) combined
    with this room's own per-photo disagreement."""
    try:
        loo = np.array(list(json.loads(Path(path).read_text())[KEY]["leave_one_out_error"].values()), float)
        cal = float(np.sqrt(np.mean(loo ** 2)))
    except (OSError, KeyError, ValueError):
        cal = 0.1
    return float(np.hypot(cal, per_frame_spread / np.sqrt(max(n_photos, 1))))


def _widen(m, rel):
    half = (m["hi"] - m["lo"]) / 2
    h = float(np.hypot(half, 1.96 * rel * m["value"]))
    m["lo"], m["hi"] = round(max(m["value"] - h, 0.0), 4), round(m["value"] + h, 4)   # lengths/areas >= 0


def widen_room(room, rel):
    """Add a relative scale uncertainty `rel` (std) to every measurement; areas scale with rel twice."""
    for w in room["walls"]:
        _widen(w["length"], rel)
    _widen(room["floor_area"], 2 * rel)
    if "perimeter" in room:
        _widen(room["perimeter"], rel)
    _widen(room["ceiling_height"], rel)
    ch = room["ceiling_height"]
    room["warnings"] = [w.split("; reported range")[0] + f"; reported range {ch['lo']:.2f}-{ch['hi']:.2f} m "
                        "(includes the photo tier's scale uncertainty)" if w.startswith("ceiling not observed") else w
                        for w in room.get("warnings", [])]
    for op in room.get("openings", []):
        for k in ("width", "height", "sill_height"):
            if k in op:
                _widen(op[k], rel)


def _model_reconstruct():
    from roomscan.photo_room import measure_photo_room, photos_to_capture
    from roomscan.video_poses import VGGTRunner
    from roomscan.video_scale import MetricDepth, load_bias
    runner, metric, bias = VGGTRunner(ROOT), MetricDepth(ROOT), load_bias()

    def reconstruct(imgs, fx, out_dir, rid):
        cap_dir, info = photos_to_capture(imgs, fx, out_dir, runner, metric, bias)
        return measure_photo_room(cap_dir, rid, tier="photo"), info
    return reconstruct


def _room_damage(m, detect):
    """Damage for one photo room, measured in that room's own frame (before stitching moves it)."""
    from roomscan.damage_pipeline import annotate, damage_for_capture, filter_damage, room_surface_list
    from roomscan.load_capture import load_stray
    g, rid = m["geometry"], m["room"]["id"]
    surfaces = room_surface_list(rid, g["s"], g["edges"], g["angle"], g["verts"])
    found = filter_damage(damage_for_capture(load_stray(m["cap_dir"]), m["cap_dir"], surfaces, "photo",
                                             detect=detect), "photo")
    return annotate(found, {rid: m["room"]}, {rid: g["angle"]})


def run_photos(path, out_dir, reconstruct=None, damage=False, detect=None):
    t0 = time.time()
    path, out_dir = Path(path), Path(out_dir)
    folders = room_folders(path)
    if not folders:
        raise CaptureError(f"{path}: no room folders with photos")
    reconstruct = reconstruct or _model_reconstruct()
    infos, images, warnings, per_room = [], [], [], {}
    for f in folders:
        imgs, fx, names, dropped = load_room_images(f)
        rid = f.name
        if dropped:
            warnings.append(f"{rid}: {len(dropped)} photo(s) in the other orientation not used: {', '.join(dropped)}")
        if len(imgs) < MIN_PHOTOS:
            warnings.append(f"{rid}: {len(imgs)} photo(s); a room needs at least {MIN_PHOTOS} photos - skipped")
            continue
        if len(imgs) > MAX_PHOTOS:
            keep = np.linspace(0, len(imgs) - 1, MAX_PHOTOS).round().astype(int)
            warnings.append(f"{rid}: {len(imgs)} photos; using {MAX_PHOTOS} spread over the set")
            imgs = imgs[keep]
        try:
            m, info = reconstruct(imgs, fx, out_dir / "photo_rooms" / rid, rid)
        except Exception as e:                       # one bad room must not sink the property
            warnings.append(f"{rid}: could not be reconstructed ({e}) - skipped")
            continue
        rel = scale_rel_sigma(info.get("metric_scale_spread", 0.0), len(imgs))
        widen_room(m["room"], rel)
        info["scale_rel_sigma"] = round(rel, 4)
        infos.append(m)
        images.append(imgs)
        per_room[rid] = info
    if not infos:
        raise CaptureError("no room could be reconstructed from the photos")
    found, t_dmg = [], time.time()
    if damage:
        from roomscan.damage_pipeline import free_gpu
        reconstruct = None                          # drop the 3D models before loading the damage models
        free_gpu()
        for m in infos:
            if "geometry" in m:
                found += _room_damage(m, detect)
            else:
                warnings.append(f"{m['room']['id']}: no fitted surfaces (fallback room) - damage not checked")
    links = room_links(images)
    rooms, adjacency, stitch_warnings = stitch(infos, links)
    meta = {"n_rooms": len(rooms), "rooms": per_room,
            "links": [{"room_a": infos[i]["room"]["id"], "room_b": infos[j]["room"]["id"], **v}
                      for (i, j), v in sorted(links.items())],
            "runtime_s": round(time.time() - t0, 1)}
    plan = build_plan(path.name, "photo", rooms, meta, adjacency)
    plan["warnings"] = warnings + stitch_warnings + plan["warnings"]
    if damage:
        from roomscan.damage_pipeline import finish_plan
        finish_plan(plan, found)
        plan["meta"]["damage_s"] = round(time.time() - t_dmg, 1)
    return plan
