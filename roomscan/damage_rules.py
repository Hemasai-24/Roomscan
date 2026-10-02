"""Concealed-damage rules: plain if-then checks that flag damage you cannot see, each with an id.

Each flag names the rule that fired and its evidence (damage ids, surface ids), as the brief asks."""

WET = ("water_stain", "mold")
LOW_ON_WALL = 0.3          # m: stain bottom this close to the floor -> rising damp / floor-level leak
LONG_CRACK = 1.0           # m
WET_ROOM_STEP = 0.03       # m: a neighbour floor this much lower is treated as a wet room

RULES = [
    {"id": "R1", "severity": "high",
     "description": "Water stain or mould on a ceiling: possible leak from above (roof, bathroom, pipe)."},
    {"id": "R2", "severity": "medium",
     "description": "Water stain or mould low on a wall: possible rising damp or a leak at floor level."},
    {"id": "R3", "severity": "high",
     "description": "Mould: hidden moisture behind the surface is likely; inspect the cavity."},
    {"id": "R4", "severity": "high",
     "description": "Crack longer than 1 m or starting at a door/window corner: possible structural movement."},
    {"id": "R5", "severity": "medium",
     "description": "Water stain or mould on the wall towards a lower-floored (wet) room: possible pipe leak in the wall."},
]
_BY_ID = {r["id"]: r for r in RULES}


def _surface_kind(surface_id):
    tail = surface_id.rsplit("_", 1)[-1]
    return tail if tail in ("ceiling", "floor") else "wall"


def _flag(rule_id, d):
    r = _BY_ID[rule_id]
    return {"rule_id": rule_id, "description": r["description"], "severity": r["severity"],
            "damage_ids": [d["id"]], "surface_ids": [d["surface_id"]]}


def _wet_neighbour_walls(rooms, adjacency):
    """Wall ids whose door leads into a neighbouring room with a floor >= 3 cm lower."""
    by_id = {r["id"]: r for r in rooms}
    out = set()
    for a in adjacency:
        for here, there in ((a["room_a"], a["room_b"]), (a["room_b"], a["room_a"])):
            r, o = by_id.get(here), by_id.get(there)
            if r is None or o is None or r.get("floor_level") is None or o.get("floor_level") is None:
                continue
            if r["floor_level"] - o["floor_level"] < WET_ROOM_STEP:
                continue
            for op in r.get("openings", []):
                if op["id"] == a["via"]:
                    out.add(op["wall_id"])
    return out


def concealed_flags(damage, rooms, adjacency):
    wet_walls = _wet_neighbour_walls(rooms, adjacency)
    flags = []
    for d in damage:
        kind = _surface_kind(d["surface_id"])
        cls = d["class"]
        if cls in WET and kind == "ceiling":
            flags.append(_flag("R1", d))
        if cls in WET and kind == "wall" and d["bottom_above_floor"]["value"] < LOW_ON_WALL:
            flags.append(_flag("R2", d))
        if cls == "mold":
            flags.append(_flag("R3", d))
        if cls == "crack":
            length = max(d["width"]["value"], d["height"]["value"])
            if length > LONG_CRACK or d.get("near_opening"):
                flags.append(_flag("R4", d))
        if cls in WET and kind == "wall" and d["surface_id"] in wet_walls:
            flags.append(_flag("R5", d))
    return flags
