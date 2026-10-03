from roomscan.photo_pipeline import widen_room


def M(v, h):
    return {"value": v, "lo": v - h, "hi": v + h, "unit": "m"}


def test_widened_unseen_ceiling_warning_shows_the_final_range():
    room = {"walls": [], "floor_area": dict(M(10.0, 1.0), unit="m2"), "ceiling_height": M(2.6, 0.3), "openings": [],
            "ceiling_observed": False,
            "warnings": ["ceiling not observed: highest wall point seen at 2.50 m; reported range 2.30-2.90 m (x)"]}
    widen_room(room, 0.6)
    ch = room["ceiling_height"]
    w = room["warnings"][0]
    assert f"{ch['lo']:.2f}-{ch['hi']:.2f}" in w
