from roomscan.damage_pipeline import filter_damage


def M(v, unit="m"):
    return {"value": v, "lo": v * 0.8, "hi": v * 1.2, "unit": unit}


def dmg(cls="water_stain", n_views=2, area=0.02, score=0.5, surface="r_w0"):
    return {"id": "d", "class": cls, "surface_id": surface, "room_id": "r", "n_views": n_views, "score": score,
            "area": M(area, "m2"), "width": M(0.2), "height": M(0.1), "bottom_above_floor": M(1.0)}


def test_photo_damage_needs_two_photos():
    assert filter_damage([dmg(n_views=1)], "photo") == []
    assert len(filter_damage([dmg(n_views=2)], "photo")) == 1


def test_tiny_regions_are_noise():
    assert filter_damage([dmg(area=0.001)], "lidar") == []


def test_peeling_paint_needs_higher_confidence():
    assert filter_damage([dmg(cls="peeling_paint", score=0.40)], "lidar") == []
    assert len(filter_damage([dmg(cls="peeling_paint", score=0.50)], "lidar")) == 1
