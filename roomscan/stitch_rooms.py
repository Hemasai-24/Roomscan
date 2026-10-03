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
PUSH_STEP = 0.05
VIRTUAL_DOOR_WIDTH = 0.8
MAX_NEIGHBOUR_GAP = 1.0    # m: rooms placed by a shared photo this close to another room are slid against it


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


def same_photo_pose(info_i, ki, info_j, kj):
    """(R2, t) moving room j so that photo kj lands on photo ki, looking the same way. Used when the very same
    photo is in both rooms' folders: it was taken from one spot, so its camera must coincide in both rooms."""
    fi, fj = _unit(info_i["forward_plan"][ki]), _unit(info_j["forward_plan"][kj])
    ang = np.arctan2(fi[1], fi[0]) - np.arctan2(fj[1], fj[0])
    ang = round(ang / (np.pi / 2)) * (np.pi / 2)    # room outlines are square to their walls: turn by right angles
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return R, np.asarray(info_i["cameras_plan"][ki], float) - R @ np.asarray(info_j["cameras_plan"][kj], float)


def _candidates(infos, rooms, placed_idx, i, j, link, used):
    """All (score, j, R, t, door_i, door_j) for attaching unplaced room j to placed room i."""
    ki, kj = (link["photo_i"], link["photo_j"]) if i < j else (link["photo_j"], link["photo_i"])
    cos_lim = np.cos(np.radians(AIM_MAX_DEG))
    out = []
    if link.get("same_photo"):
        # strongest evidence: the shared photo's camera is one point in both rooms
        R, t = same_photo_pose(infos[i], ki, infos[j], kj)
        cand = transform_room(infos[j]["room"], R, t)
        ov = _overlap(cand, [rooms[p] for p in placed_idx])
        cam = {"id": f"{infos[j]['room']['id']}_shared_photo", "mid": np.asarray(infos[i]["cameras_plan"][ki], float),
               "normal_out": _unit(infos[i]["forward_plan"][ki]), "width": VIRTUAL_DOOR_WIDTH, "virtual": True}
        out.append((np.log1p(link["matches"]) + 3.0 - 2 * ov, R, t, cam, cam, ov))
        return out
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


def resolve_overlap(room, placed, normal):
    """Smallest move (away from the door first, then sideways, then back) after which the room overlaps the
    placed rooms by <= MAX_OVERLAP. Always succeeds: far enough away nothing overlaps."""
    n = _unit(normal)
    dirs = [n, np.array([-n[1], n[0]]), np.array([n[1], -n[0]]), -n]
    k = 0
    while True:
        for d in dirs:
            moved = transform_room(room, np.eye(2), d * PUSH_STEP * k)
            if _overlap(moved, placed) <= MAX_OVERLAP:
                return moved, PUSH_STEP * k
        k += 1


def slide_to(room, target, placed, gap=WALL_THICKNESS):
    """Move room straight towards `target` until they are `gap` apart, stopping early if it would overlap a
    placed room. A shared photo fixes a room's turn well but its position only to about a metre."""
    from shapely.ops import nearest_points
    p, q = nearest_points(_poly(room), _poly(target))
    d = p.distance(q)
    if d <= gap + PUSH_STEP:
        return room, 0.0
    u = _unit([q.x - p.x, q.y - p.y])
    moved, done = room, 0.0
    for dist in list(np.arange(PUSH_STEP, d - gap, PUSH_STEP)) + [d - gap]:
        nxt = transform_room(room, np.eye(2), u * dist)
        if _overlap(nxt, placed) > MAX_OVERLAP:
            break
        moved, done = nxt, float(dist)
    return moved, done


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
    parent_of, by_photo = {}, []
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
            room, pushed = resolve_overlap(room, [rooms[p] for p in placed], _unit(da["normal_out"]))
            warnings.append(f"{room['id']}: pushed {pushed:.2f} m away from {rooms[a]['id']} to avoid overlap")
        if l.get("same_photo"):
            room, slid = slide_to(room, rooms[a], [rooms[p] for p in placed])
            if slid > 0:
                warnings.append(f"{room['id']}: moved {slid:.2f} m towards {rooms[a]['id']} to close the gap")
            by_photo.append(b)
        parent_of[b] = a
        rooms[b] = room
        placed.append(b)
        # a detected door joins exactly two rooms; a virtual door is only "the wall this photo looks at",
        # which may lead to several rooms (overlaps are still prevented by pushing)
        used.update(d["id"] for d in (da, db) if not d["virtual"])
        conf = min(1.0, l["matches"] / 100) * (0.5 if (da["virtual"] or db["virtual"]) else 1.0) \
            * (0.5 if pushed > 0 else 1.0)
        adjacency.append({"room_a": rooms[a]["id"], "room_b": room["id"], "via": da["id"], "via_b": db["id"],
                          "evidence_matches": int(l["matches"]), "confidence": round(float(conf), 2)})
    # a shared photo fixes a room's position to about a metre: close small gaps to neighbouring rooms too
    for b in by_photo:
        others = [p for p in placed if p not in (b, parent_of[b])]
        if not others:
            continue
        near = min(others, key=lambda p: _poly(rooms[b]).distance(_poly(rooms[p])))
        if _poly(rooms[b]).distance(_poly(rooms[near])) <= MAX_NEIGHBOUR_GAP:
            rooms[b], slid = slide_to(rooms[b], rooms[near], [rooms[p] for p in placed if p != b])
            if slid > 0:
                warnings.append(f"{rooms[b]['id']}: moved {slid:.2f} m towards {rooms[near]['id']} to close the gap")
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
