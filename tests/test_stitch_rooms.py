import numpy as np
from shapely.geometry import Polygon

from roomscan.stitch_rooms import door_frames, place, stitch, transform_room, virtual_door


def make_room(rid, poly, doors=()):
    """Room record like measure_room's: CCW polygon, wall i from poly[i-1] to poly[i]; doors (wall, offset, width)."""
    walls = [{"id": f"{rid}_w{i}", "start": list(poly[i - 1]), "end": list(poly[i]),
              "length": {"value": float(np.linalg.norm(np.subtract(poly[i], poly[i - 1])))}} for i in range(len(poly))]
    ops = [{"id": f"{rid}_o{k}", "wall_id": f"{rid}_w{w}", "type": "door", "offset_along_wall": off,
            "width": {"value": wd}} for k, (w, off, wd) in enumerate(doors)]
    return {"id": rid, "polygon": [list(p) for p in poly], "walls": walls, "openings": ops,
            "floor_area": {"value": Polygon(poly).area}}


def rot(poly, deg, shift):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return [tuple(R @ np.array(p) + shift) for p in poly]


def info(room, cams, fwds):
    return {"room": room, "cameras_plan": np.array(cams, float), "forward_plan": np.array(fwds, float)}


# A: 4 x 3, door on its east wall (wall 2 runs (4,0)->(4,3)) at z 1.0..1.9
A = make_room("A", [(0, 0), (4, 0), (4, 3), (0, 3)], doors=[(2, 1.0, 0.9)])
# B: 3 x 3 in its own frame; door on its west wall (wall 0 runs (0,3)->(0,0)) at 1.2..2.1 from (0,3)
B_local = make_room("B", rot([(0, 0), (3, 0), (3, 3), (0, 3)], 0, (0, 0)), doors=[(0, 1.2, 0.9)])


def _overlaps(rooms):
    ps = [Polygon(r["polygon"]).buffer(0) for r in rooms]
    return max([ps[i].intersection(ps[j]).area for i in range(len(ps)) for j in range(i + 1, len(ps))] + [0])


def test_door_frame_points_out_of_room():
    d = door_frames(A)[0]
    np.testing.assert_allclose(d["mid"], [4.0, 1.45])
    np.testing.assert_allclose(d["normal_out"], [1.0, 0.0], atol=1e-9)


def test_place_puts_b_door_facing_a_door_one_wall_apart():
    da, db = door_frames(A)[0], door_frames(B_local)[0]
    R, t = place(db, da, wall_thickness=0.15)
    b = transform_room(B_local, R, t)
    db2 = door_frames(b)[0]
    np.testing.assert_allclose(db2["mid"], da["mid"] + 0.15 * da["normal_out"], atol=1e-9)
    np.testing.assert_allclose(db2["normal_out"], -da["normal_out"], atol=1e-9)
    assert Polygon(b["polygon"]).intersection(Polygon(A["polygon"])).area < 1e-9


def test_two_rooms_stitched_through_their_door():
    b = make_room("B", rot(B_local["polygon"], 90, (7, -2)), doors=[(0, 1.2, 0.9)])
    rooms, adj, warns = stitch([info(A, [(2, 1.5)], [(1, 0)]), info(b, [(5.5, -0.5)], [(0, 1)])],
                               {(0, 1): {"matches": 80, "photo_i": 0, "photo_j": 0}})
    pa, pb = Polygon(rooms[0]["polygon"]), Polygon(rooms[1]["polygon"])
    assert pa.intersection(pb).area < 0.1
    assert pb.bounds[0] > 3.9                                     # B is east of A's door wall
    assert [(e["room_a"], e["room_b"]) for e in adj] == [("A", "B")]
    assert adj[0]["via"] == "A_o0"


def test_three_rooms_in_a_row():
    a = make_room("A", [(0, 0), (4, 0), (4, 3), (0, 3)], doors=[(2, 1.0, 0.9)])
    b = make_room("B", [(0, 0), (3, 0), (3, 3), (0, 3)], doors=[(0, 1.2, 0.9), (2, 1.0, 0.8)])
    c = make_room("C", rot([(0, 0), (2, 0), (2, 2), (0, 2)], 180, (10, 10)), doors=[(0, 0.5, 0.8)])
    infos = [info(a, [(2, 1.5)], [(1, 0)]), info(b, [(1.5, 1.5)], [(1, 0)]), info(c, [(9, 9)], [(0, 1)])]
    rooms, adj, warns = stitch(infos, {(0, 1): {"matches": 60, "photo_i": 0, "photo_j": 0},
                                       (1, 2): {"matches": 50, "photo_i": 0, "photo_j": 0}})
    assert len(adj) == 2 and _overlaps(rooms) < 0.1
    xs = [Polygon(r["polygon"]).centroid.x for r in rooms]
    assert xs[0] < xs[1] < xs[2]


def test_room_without_doors_linked_by_photo_gets_virtual_door():
    b = make_room("B", [(0, 0), (3, 0), (3, 3), (0, 3)])            # no detected door
    rooms, adj, warns = stitch([info(A, [(2, 1.5)], [(1, 0)]), info(b, [(1.5, 1.5)], [(-1, 0)])],
                               {(0, 1): {"matches": 70, "photo_i": 0, "photo_j": 0}})
    assert len(adj) == 1 and adj[0]["via_b"].endswith("vdoor")
    assert Polygon(rooms[1]["polygon"]).bounds[0] > 3.9
    vd = virtual_door(b, np.array([1.5, 1.5]), np.array([-1.0, 0.0]))
    np.testing.assert_allclose(vd["mid"], [0.0, 1.5])


def test_overlap_resolved_even_when_pushing_along_the_door_is_not_enough():
    a = make_room("A", [(0, 0), (4, 0), (4, 3), (0, 3)], doors=[(2, 1.0, 0.9), (3, 0.3, 0.9)])
    b = make_room("B", [(0, 0), (3, 0), (3, 4), (0, 4)], doors=[(0, 2.5, 0.9)])
    huge = make_room("C", [(0, 0), (12, 0), (12, 12), (0, 12)], doors=[(1, 5.55, 0.9)])   # north of A, x up to 9.25
    rooms, adj, warns = stitch([info(a, [(2, 1.5)], [(1, 0)]), info(b, [(1.5, 2)], [(-1, 0)]),
                                info(huge, [(6, 6)], [(0, -1)])],
                               {(0, 1): {"matches": 60, "photo_i": 0, "photo_j": 0},
                                (0, 2): {"matches": 95, "photo_i": 0, "photo_j": 0}})
    assert _overlaps(rooms) <= 0.1


def test_resolve_overlap_moves_far_enough_in_some_direction():
    from roomscan.stitch_rooms import resolve_overlap
    big = make_room("P", [(0, 0), (20, 0), (20, 14), (0, 14)])
    small = make_room("S", [(9, 6), (11, 6), (11, 8), (9, 8)])
    moved, dist = resolve_overlap(small, [big], np.array([1.0, 0.0]))
    assert Polygon(moved["polygon"]).intersection(Polygon(big["polygon"])).area <= 0.1
    assert dist < 8.5          # nearest free side is north/south (~7.95 m), not east (~11 m)


def test_room_without_doors_can_link_to_several_rooms():
    hub = make_room("H", [(0, 0), (4, 0), (4, 4), (0, 4)])                 # no detected doors
    b = make_room("B", [(0, 0), (3, 0), (3, 3), (0, 3)], doors=[(0, 1.2, 0.9)])
    c = make_room("C", [(0, 0), (3, 0), (3, 3), (0, 3)], doors=[(0, 1.2, 0.9)])
    infos = [info(hub, [(2, 2), (2, 2)], [(1, 0), (0, 1)]), info(b, [(1.5, 1.5)], [(-1, 0)]),
             info(c, [(1.5, 1.5)], [(-1, 0)])]
    rooms, adj, warns = stitch(infos, {(0, 1): {"matches": 90, "photo_i": 0, "photo_j": 0},
                                       (0, 2): {"matches": 80, "photo_i": 0, "photo_j": 0}})
    assert len(adj) == 2
    assert _overlaps(rooms) <= 0.1


def test_unlinked_room_placed_beside_with_warning():
    b = make_room("B", [(0, 0), (3, 0), (3, 3), (0, 3)], doors=[(0, 1.2, 0.9)])
    lone = make_room("L", [(0, 0), (2, 0), (2, 2), (0, 2)])
    rooms, adj, warns = stitch([info(A, [(2, 1.5)], [(1, 0)]), info(b, [(1.5, 1.5)], [(-1, 0)]),
                                info(lone, [(1, 1)], [(1, 0)])],
                               {(0, 1): {"matches": 70, "photo_i": 0, "photo_j": 0}})
    assert len(rooms) == 3 and _overlaps(rooms) < 0.1
    assert any("L" in w and "not connected" in w for w in warns)


def test_conflicting_placement_pushed_apart():
    a = make_room("A", [(0, 0), (4, 0), (4, 3), (0, 3)], doors=[(2, 1.0, 0.9), (3, 0.3, 0.9)])
    # B is 3 x 4 with its door low on its west wall, so placed east of A it reaches up to y = 4.4,
    # into the spot where C (north of A, y >= 3.15, x 2.3..8.3) goes
    b = make_room("B", [(0, 0), (3, 0), (3, 4), (0, 4)], doors=[(0, 2.5, 0.9)])
    big = make_room("C", [(0, 0), (6, 0), (6, 6), (0, 6)], doors=[(1, 0.5, 0.9)])
    rooms, adj, warns = stitch([info(a, [(2, 1.5)], [(1, 0)]), info(b, [(1.5, 1.5)], [(-1, 0)]),
                                info(big, [(3, 3)], [(0, -1)])],
                               {(0, 1): {"matches": 90, "photo_i": 0, "photo_j": 0},
                                (0, 2): {"matches": 40, "photo_i": 0, "photo_j": 0}})
    assert _overlaps(rooms) <= 0.1
    assert any("pushed" in w for w in warns)


def test_registration_overrides_door_guessing():
    # B's door would put it EAST of A, but registration says B truly sits NORTH of A
    b = make_room("B", rot(B_local["polygon"], 90, (7, -2)), doors=[(0, 1.2, 0.9)])
    R_true = np.array([[0.0, 1.0], [-1.0, 0.0]])            # undo the 90 deg turn
    turned = np.array(transform_room(b, R_true, np.zeros(2))["polygon"])
    t_true = np.array([0.0, 3.15]) - turned.min(0)

    def register(i, j):
        return (R_true, t_true, 40) if (i, j) == (0, 1) else None
    rooms, adj, warns = stitch([info(A, [(2, 1.5)], [(1, 0)]), info(b, [(5.5, -0.5)], [(0, 1)])],
                               {(0, 1): {"matches": 80, "photo_i": 0, "photo_j": 0}}, register=register)
    pb = Polygon(rooms[1]["polygon"])
    np.testing.assert_allclose(pb.bounds[:2], [0.0, 3.15], atol=1e-6)    # north of A
    assert _overlaps(rooms) < 0.1
    assert adj[0]["placement"] == "registered"


def test_registration_composes_with_moved_rooms_and_inverts():
    # A-B-C-D in a row, each registered only as (left, right) = "right is 2.2 m east of left".
    # B is the root (strongest links); A is placed by the INVERTED registration; D is placed against C,
    # which itself was moved - so D's placement must compose with C's.
    sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
    rs = [make_room(n, sq) for n in "ABCD"]
    east = (np.eye(2), np.array([2.2, 0.0]), 50)

    def register(i, j):
        return east if j == i + 1 else None
    links = {(0, 1): {"matches": 90, "photo_i": 0, "photo_j": 0}, (1, 2): {"matches": 60, "photo_i": 0, "photo_j": 0},
             (2, 3): {"matches": 30, "photo_i": 0, "photo_j": 0}}
    rooms, adj, _ = stitch([info(r, [(1, 1)], [(1, 0)]) for r in rs], links, register=register)
    xs = [Polygon(r["polygon"]).bounds[0] for r in rooms]
    np.testing.assert_allclose(np.subtract(xs, xs[0]), [0.0, 2.2, 4.4, 6.6], atol=1e-6)
    assert all(e["placement"] == "registered" for e in adj)
