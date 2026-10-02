import numpy as np
from roomscan.connect_rooms import adjacency
from roomscan.room_outline import Grid


def _room(rid, poly, doors):
    P = np.array(poly, float)
    walls = [{"id": f"{rid}_w{i}", "start": list(P[i - 1]), "end": list(P[i])} for i in range(len(P))]
    ops = [{"id": f"{rid}_o{k}", "wall_id": f"{rid}_w{w}", "type": "door", "offset_along_wall": off,
            "width": {"value": 0.9}} for k, (w, off) in enumerate(doors)]
    return {"id": rid, "walls": walls, "openings": ops}


def _mask(g, u0, u1, v0, v1):
    m = np.zeros(g.shape, bool)
    m[int((v0 - g.lo[1]) / g.cell):int((v1 - g.lo[1]) / g.cell), int((u0 - g.lo[0]) / g.cell):int((u1 - g.lo[0]) / g.cell)] = True
    return m


def test_door_on_shared_wall_connects_rooms():
    g = Grid(np.array([-1.0, -1.0]), 0.05, (120, 200))
    # room A [0,4]x[0,3] CCW: w1 is the edge (4,0)->(4,3); door 1.0 m along it
    a = _room("a", [(0, 0), (4, 0), (4, 3), (0, 3)], doors=[(2, 1.0)])
    b = _room("b", [(4.1, 0), (7, 0), (7, 3), (4.1, 3)], doors=[])
    c = _room("c", [(0, 4), (4, 4), (4, 6), (0, 6)], doors=[])        # touches A only via a wall
    masks = [_mask(g, 0, 4, 0, 3), _mask(g, 4.1, 7, 0, 3), _mask(g, 0, 4, 4, 6)]
    adj = adjacency([a, b, c], masks, g)
    assert adj == [{"room_a": "a", "room_b": "b", "via": "a_o0"}]
