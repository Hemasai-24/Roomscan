"""Photo tier: measure a door from one straight-on photo of it (the `doors/` subfolder of a room folder).

The detector finds the door frame ("door frame."). Depth from the same photo gives the wall around the frame
as a plane. The opening is the part of the frame you can see through: there the depth jumps well beyond the
wall plane. Its left and right ends, projected onto the wall plane, give the width. These photos are not
used for the room's shape (they are usually taken from outside the room, through the doorway)."""
import numpy as np

FRAME_PROMPT = "door frame."
FRAME_THRESHOLD = 0.3
STRIP = 0.12              # wall strip each side of the frame used to fit the wall plane (fraction of box width)
BEYOND = 0.25             # m: a pixel this far behind the wall plane is seen through the opening
SCALE_REL_SIGMA = 0.10    # single-photo metric depth: about +10 % on the Galaxy M53 doors (4 doors, tape)


def _ray(u, v, f, cx, cy):
    return np.array([(u - cx) / f, (v - cy) / f, 1.0])


def wall_plane(D, box, f, cx, cy):
    """(point, normal) of the wall fitted to the depth strips left and right of the box."""
    x0, y0, x1, y1 = box
    H, W = D.shape
    m = max(2, int(STRIP * (x1 - x0)))
    pts = []
    for xa, xb in ((max(0, int(x0) - m), int(x0)), (int(x1), min(W, int(x1) + m))):
        for v in range(max(0, int(y0)), min(H, int(y1)), 4):
            for u in range(xa, xb, 2):
                pts.append(_ray(u, v, f, cx, cy) * D[v, u])
    P = np.array(pts)
    c = P.mean(0)
    return c, np.linalg.svd(P - c)[2][-1]


def see_through_width(D, box, f, cx, cy):
    """Width (m) of the see-through opening inside the door-frame box, median over rows at 30-70 % height;
    None if no row sees through."""
    c, n = wall_plane(D, box, f, cx, cy)
    x0, y0, x1, y1 = box
    hit = lambda u, v: _ray(u, v, f, cx, cy) * ((n @ c) / (n @ _ray(u, v, f, cx, cy)))
    us = np.arange(int(np.ceil(x0)), int(x1))
    widths = []
    for v in np.linspace(y0 + 0.3 * (y1 - y0), y0 + 0.7 * (y1 - y0), 15).astype(int):
        wall_z = np.array([hit(u, v)[2] for u in us])
        idx = np.where(D[v, us] > wall_z + BEYOND)[0]
        if len(idx) < 5:
            continue
        runs = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        r = max(runs, key=len)
        widths.append(np.linalg.norm(hit(us[r[0]] - 0.5, v) - hit(us[r[-1]] + 0.5, v)))
    return float(np.median(widths)) if widths else None


def measure_door_photo(img, f, detector, depth_model, bias):
    """{"width", "lo", "hi", "score"} for the most confident door frame in the photo, or None."""
    boxes = detector.boxes(np.ascontiguousarray(img), FRAME_PROMPT, FRAME_THRESHOLD)
    if not boxes:
        return None
    b = max(boxes, key=lambda x: x["score"])
    D = depth_model.predict([img])[0] / bias
    H, W = D.shape
    w = see_through_width(D, b["box"], f, W / 2, H / 2)
    if w is None or not 0.5 <= w <= 1.5:
        return None
    s = 1.96 * SCALE_REL_SIGMA * w
    return {"width": w, "lo": max(0.0, w - s), "hi": w + s, "score": float(b["score"])}


def _measure(m):
    return {"value": round(m["width"], 4), "lo": round(m["lo"], 4), "hi": round(m["hi"], 4), "unit": "m"}


def add_measured_door(room, m, parent=None):
    """Put a door measured from its own photo into the room. If the room already has a detected door, only its
    width is replaced (its position stays). Otherwise the door goes on the room's wall nearest to the room it
    connects to (`parent`, e.g. the hall), centred at the closest point; with no parent, on the longest wall."""
    from shapely.geometry import LineString, Polygon
    doors = [o for o in room["openings"] if o["type"] == "door"]
    if doors:
        doors[0].update(width=_measure(m), source="door_photo", n_views=1)
        doors[0].pop("height", None)
        return doors[0]
    walls = room["walls"]
    if parent is not None:
        pp = Polygon(parent["polygon"]).buffer(0)
        fits = [x for x in walls if np.linalg.norm(np.subtract(x["end"], x["start"])) >= m["width"]] or walls
        mid = lambda x: LineString([x["start"], x["end"]]).interpolate(0.5, normalized=True)
        w = min(fits, key=lambda x: mid(x).distance(pp))
        line = LineString([w["start"], w["end"]])
        along = line.project(pp.centroid)
    else:
        w = max(walls, key=lambda x: np.linalg.norm(np.subtract(x["end"], x["start"])))
        line = LineString([w["start"], w["end"]])
        along = line.length / 2
    off = float(np.clip(along - m["width"] / 2, 0.0, max(0.0, line.length - m["width"])))
    door = {"id": f"{room['id']}_dp{len(room['openings'])}", "wall_id": w["id"], "type": "door",
            "offset_along_wall": round(off, 4), "width": _measure(m), "source": "door_photo", "n_views": 1}
    room["openings"].append(door)
    return door


def load_door_photos(room_folder, height=1008):
    """[(image, focal_px)] for the photos in `<room_folder>/doors/`, upright, scaled to `height` pixels."""
    from PIL import Image, ImageOps
    from roomscan.photo_folders import EXIF_IFD, FOCAL_35MM, _images, focal_px_from_35mm
    d = room_folder / "doors"
    out = []
    for p in (_images(d) if d.is_dir() else []):
        im = Image.open(p)
        f35 = im.getexif().get_ifd(EXIF_IFD).get(FOCAL_35MM)
        im = ImageOps.exif_transpose(im).convert("RGB")
        if not f35:                     # no lens data: the size cannot be measured from one photo
            continue
        W0, H0 = im.size
        s = height / H0
        out.append((np.asarray(im.resize((round(W0 * s), height))), focal_px_from_35mm(f35, W0, H0) * s))
    return out


def doors_from_photos(folders, rooms, adjacency, warnings):
    """Measure every `doors/` photo and add the doors to the stitched rooms (in place)."""
    todo = {f.name: load_door_photos(f) for f in folders}
    if not any(todo.values()):
        return
    from roomscan.damage_detect import get_detector
    from roomscan.video_scale import MetricDepth, load_bias
    det, depth, bias = get_detector(), MetricDepth(), load_bias()
    by_id = {r["id"]: r for r in rooms}
    parent = {}
    for a in adjacency:
        parent.setdefault(a["room_b"], by_id.get(a["room_a"]))
    for rid, photos in todo.items():
        if rid not in by_id:
            continue
        for img, f in photos:
            m = measure_door_photo(img, f, det, depth, bias)
            if m is None:
                warnings.append(f"{rid}: no door could be measured in a doors/ photo")
                continue
            add_measured_door(by_id[rid], m, parent.get(rid))
