"""Photo tier, step P5: put separately reconstructed rooms together into one whole-property plan.

Rooms are puzzle pieces joined at doors. Two rooms are linked when photos of one show the inside of the
other (feature matches, roomscan/photo_links.py). Room B is turned and moved so that one of its doors lands
on a door of room A, facing it, one wall thickness apart. Greedy: the strongest links are placed first. A
placement may never overlap rooms already placed: the new room is pushed away from the door until it does
not. A room that cannot be linked is still drawn, in a row beside the plan, with a warning."""
import numpy as np
from shapely.geometry import Polygon

WALL_THICKNESS = 0.15
MIN_MATCHES = 25
MAX_OVERLAP = 0.1          # m2
AIM_MAX_DEG = 60.0         # a photo "looks at" a door if the door is within this angle of its view direction
PUSH_STEP, PUSH_MAX = 0.05, 4.0
VIRTUAL_DOOR_WIDTH = 0.8


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _wall_frame(room, wall_id):
    w = next(x for x in room["walls"] if x["id"] == wall_id)
    a, b = np.array(w["start"], float), np.array(w["end"], float)
    u = _unit(b - a)
    return a, u, np.array([u[1], -u[0]])          # CCW polygon: interior on the left, outward on the right


def door_frames(room):
    out = []
    for op in room.get("openings", []):
        if op["type"] != "door":
            continue
        a, u, n = _wall_frame(room, op["wall_id"])
        w = op["width"]["value"]
        out.append({"id": op["id"], "mid": a + u * (op["offset_along_wall"] + w / 2), "normal_out": n,
                    "width": w, "virtual": False})
    return out


def virtual_door(room, camera, forward, width=VIRTUAL_DOOR_WIDTH):
    """A door on the wall that a photo looks at (first wall hit by the view ray)."""
    c, f = np.asarray(camera, float), _unit(forward)
    best = None
    for w in room["walls"]:
        a, b = np.array(w["start"], float), np.array(w["end"], float)
        d = b - a
        M = np.array([f, -d]).T
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        t, s = np.linalg.solve(M, a - c)
        if t > 1e-6 and 0 <= s <= 1 and (best is None or t < best[0]):
            u = _unit(d)
            best = (t, c + t * f, np.array([u[1], -u[0]]))
    if best is None:
        return None
    return {"id": f"{room['id']}_vdoor", "mid": best[1], "normal_out": best[2], "width": width, "virtual": True}


def place(door_b, door_a, wall_thickness=WALL_THICKNESS):
    """(R2, t) moving room B so door_b faces door_a (normals opposite), one wall thickness apart."""
    na, nb = _unit(door_a["normal_out"]), _unit(door_b["normal_out"])
    ang = np.arctan2(-na[1], -na[0]) - np.arctan2(nb[1], nb[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    t = door_a["mid"] + na * wall_thickness - R @ door_b["mid"]
    return R, t


def transform_room(room, R, t):
    out = dict(room)
    tf = lambda p: [round(float(x), 4) for x in (R @ np.asarray(p, float) + t)]
    out["polygon"] = [tf(p) for p in room["polygon"]]
    out["walls"] = [dict(w, start=tf(w["start"]), end=tf(w["end"])) for w in room["walls"]]
    return out


def _poly(room):
    return Polygon(room["polygon"]).buffer(0)


def _overlap(room, placed):
    p = _poly(room)
    return sum(p.intersection(_poly(q)).area for q in placed)


def _aim(info, k, door):
    """cos of the angle between photo k's view direction and the direction to the door (or -1)."""
    if k is None or k >= len(info["cameras_plan"]):
        return -1.0
    return float(_unit(info["forward_plan"][k]) @ _unit(door["mid"] - info["cameras_plan"][k]))


def _doors(info, k):
    ds = door_frames(info["room"])
    if k is not None and k < len(info["cameras_plan"]):
        v = virtual_door(info["room"], info["cameras_plan"][k], info["forward_plan"][k])
        if v is not None:
            ds.append(v)
    return ds


def _candidates(infos, rooms, placed_idx, i, j, link, used):
    """All (score, j, R, t, door_i, door_j) for attaching unplaced room j to placed room i."""
    ki, kj = (link["photo_i"], link["photo_j"]) if i < j else (link["photo_j"], link["photo_i"])
    cos_lim = np.cos(np.radians(AIM_MAX_DEG))
    out = []
    for da in _doors(infos[i], ki):
        if da["id"] in used:
            continue
        for db in _doors(infos[j], kj):
            if db["id"] in used:
                continue
            aim = max(_aim(infos[i], ki, da), _aim(infos[j], kj, db))
            if (da["virtual"] or db["virtual"]) and aim < cos_lim:
                continue
            R, t = place(db, da)
            cand = transform_room(infos[j]["room"], R, t)
            ov = _overlap(cand, [rooms[p] for p in placed_idx])
            width = 1 - abs(da["width"] - db["width"]) / max(da["width"], db["width"])
            score = np.log1p(link["matches"]) + width + max(aim, 0) - 0.5 * (da["virtual"] + db["virtual"]) - 2 * ov
            out.append((score, R, t, da, db, ov))
    return out


def _push(room, placed, normal):
    """Move room along `normal` until it overlaps placed rooms by <= MAX_OVERLAP."""
    for k in range(int(PUSH_MAX / PUSH_STEP) + 1):
        moved = transform_room(room, np.eye(2), normal * PUSH_STEP * k)
        if _overlap(moved, placed) <= MAX_OVERLAP:
            return moved, PUSH_STEP * k
    return moved, PUSH_MAX


def stitch(infos, links):
    """infos: per room {"room", "cameras_plan", "forward_plan"} in the room's own plan coordinates;
    links: {(i, j): {"matches", "photo_i", "photo_j"}} with i < j. Returns (rooms, adjacency, warnings)."""
    n = len(infos)
    rooms = [info["room"] for info in infos]
    strength = np.zeros(n)
    for (i, j), l in links.items():
        if l["matches"] >= MIN_MATCHES:
            strength[i] += l["matches"]
            strength[j] += l["matches"]
    root = int(np.argmax(strength + 1e-6 * np.array([r["floor_area"]["value"] for r in rooms])))
    placed, used, adjacency, warnings = [root], set(), [], []
    while True:
        best = None
        for (i, j), l in links.items():
            if l["matches"] < MIN_MATCHES:
                continue
            for a, b in ((i, j), (j, i)):
                if a in placed and b not in placed:
                    for c in _candidates(infos, rooms, placed, a, b, l, used):
                        if best is None or c[0] > best[0][0]:
                            best = (c, a, b, l)
        if best is None:
            break
        (score, R, t, da, db, ov), a, b, l = best
        room = transform_room(infos[b]["room"], R, t)
        pushed = 0.0
        if ov > MAX_OVERLAP:
            room, pushed = _push(room, [rooms[p] for p in placed], _unit(da["normal_out"]))
            warnings.append(f"{room['id']}: pushed {pushed:.2f} m away from {rooms[a]['id']} to avoid overlap")
        rooms[b] = room
        placed.append(b)
        used.update({da["id"], db["id"]})
        conf = min(1.0, l["matches"] / 100) * (0.5 if (da["virtual"] or db["virtual"]) else 1.0) \
            * (0.5 if pushed > 0 else 1.0)
        adjacency.append({"room_a": rooms[a]["id"], "room_b": room["id"], "via": da["id"], "via_b": db["id"],
                          "evidence_matches": int(l["matches"]), "confidence": round(float(conf), 2)})
    rest = [k for k in range(n) if k not in placed]
    if rest:
        x = max(_poly(rooms[p]).bounds[2] for p in placed) + 1.0
        y0 = min(_poly(rooms[p]).bounds[1] for p in placed)
        for k in rest:
            minx, miny, maxx, _ = _poly(rooms[k]).bounds
            rooms[k] = transform_room(rooms[k], np.eye(2), np.array([x - minx, y0 - miny]))
            x += (maxx - minx) + 1.0
            warnings.append(f"{rooms[k]['id']}: not connected to the other rooms (no photo shows it from "
                            f"another room); drawn beside the plan")
    return rooms, adjacency, warnings
