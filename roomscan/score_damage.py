"""Score detected damage against the tape-measured staged damage (ground-truth rows damage_*).

Per staged item: detected or not (a region of that class in the mapped room), size error of the largest such
region, and whether the truth lies inside the reported range. Other regions in that room are counted as extras
(false alarms or split pieces)."""


def _largest(found, cls):
    c = [d for d in found if d["class"] == cls]
    return max(c, key=lambda d: d["area"]["value"]) if c else None


def _err(m, truth):
    return round(abs(m["value"] - truth), 4), bool(m["lo"] <= truth <= m["hi"])


def score_damage(plan, gt, room_map):
    """room_map: ground-truth room name -> plan room id (the room holding the staged damage)."""
    out = {"water_stain": {"detected": False}, "crack": {"detected": False}, "extra_detections_in_room": 0}
    for name, rid in room_map.items():
        g = gt.get(name, {})
        found = [d for d in plan["damage"] if d["room_id"] == rid]
        used = []
        if "damage_water_stain_width" in g or "damage_water_stain_height" in g:
            s = _largest(found, "water_stain")
            if s is not None:
                used.append(s)
                r = {"detected": True, "n_views": s.get("n_views"), "area_m2": s["area"]["value"]}
                for key, m in (("width", s["width"]), ("height", s["height"])):
                    truth = next(iter(g.get(f"damage_water_stain_{key}", {}).values()), None)
                    if truth is not None:
                        r[f"{key}_err_m"], r[f"{key}_in_range"] = _err(m, truth)
                out["water_stain"] = r
        if "damage_crack_length" in g:
            c = _largest(found, "crack")
            if c is not None:
                used.append(c)
                truth = next(iter(g["damage_crack_length"].values()))
                length = c["width"] if c["width"]["value"] >= c["height"]["value"] else c["height"]
                err, ok = _err(length, truth)
                out["crack"] = {"detected": True, "n_views": c.get("n_views"), "length_err_m": err,
                                "length_in_range": ok}
        out["extra_detections_in_room"] += len([d for d in found if not any(d is u for u in used)])
    return out
