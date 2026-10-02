"""Repair scope: turn damage records and concealed-damage flags into line items keyed to surfaces.

Quantities carry ranges: a wall's area is its length x the room's ceiling height, with the two
relative uncertainties combined."""
import numpy as np

MOLD_MARGIN = 1.5          # treat 50% more than the visible mould area
MOLD_MIN_M2 = 1.0


def _m(value, half, unit):
    return {"value": round(value, 4), "lo": round(value - half, 4), "hi": round(value + half, 4), "unit": unit}


def _half(m):
    return (m["hi"] - m["lo"]) / 2


def _product(a, b, unit):
    v = a["value"] * b["value"]
    rel = np.hypot(_half(a) / a["value"], _half(b) / b["value"])
    return _m(v, v * rel, unit)


def _scaled(m, k, unit):
    return _m(m["value"] * k, _half(m) * k, unit)


def _surface_area(surface_id, rooms):
    room_id, tail = surface_id.rsplit("_", 1)
    room = next(r for r in rooms if r["id"] == room_id)
    if tail in ("ceiling", "floor"):
        return dict(room["floor_area"], unit="m2")
    wall = next(w for w in room["walls"] if w["id"] == surface_id)
    return _product(wall["length"], room["ceiling_height"], "m2")


def scope_items(damage, flags, rooms):
    items = []

    def add(surface_id, action, quantity, reason):
        items.append({"id": f"s{len(items)}", "surface_id": surface_id, "action": action,
                      "quantity": quantity, "unit": quantity["unit"], "reason": reason})

    for d in damage:
        s, cls = d["surface_id"], d["class"]
        if cls == "water_stain":
            add(s, "stain_block_and_repaint", _surface_area(s, rooms), d["id"])
        elif cls == "mold":
            area = _scaled(d["area"], MOLD_MARGIN, "m2")
            if area["value"] < MOLD_MIN_M2:
                area = _m(MOLD_MIN_M2, 0.0, "m2")
            add(s, "mold_remediation", area, d["id"])
            add(s, "repaint", _surface_area(s, rooms), d["id"])
        elif cls == "crack":
            length = d["width"] if d["width"]["value"] >= d["height"]["value"] else d["height"]
            add(s, "fill_crack", dict(length, unit="m"), d["id"])
            add(s, "repaint", _surface_area(s, rooms), d["id"])
        elif cls == "hole":
            add(s, "patch", dict(d["area"], unit="m2"), d["id"])
            add(s, "repaint", _surface_area(s, rooms), d["id"])
        elif cls == "peeling_paint":
            add(s, "scrape_prime_repaint", _surface_area(s, rooms), d["id"])
    for f in flags:
        for s in f["surface_ids"]:
            add(s, "inspect", _m(1, 0, "each"), f["rule_id"])
    return items
