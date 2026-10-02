import numpy as np
from fpp.geometry.openings import find_openings
from fpp.geometry.room import Edge


def _rays_through(wall_v, hole, behind=1.0, cam=(2.0, 1.4, 1.0), seed=0):
    """Camera inside room (v < wall_v), rays to points behind wall line v = wall_v.
    Points inside `hole` (u0, u1, y0, y1) are visible beyond the wall."""
    rng = np.random.default_rng(seed)
    u0, u1, y0, y1 = hole
    u = rng.uniform(u0, u1, 20000)
    y = rng.uniform(y0, y1, 20000)
    c = np.array(cam)
    hit = np.c_[u, y, np.full_like(u, wall_v)]
    dirn = hit - c
    t = (wall_v + behind - c[2]) / dirn[:, 2]
    return [(c, c + dirn * t[:, None])]


def test_door_width():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)    # wall v=3 from u=0..4, plan(x,z)=(u,v)
    ops = find_openings(_rays_through(3.0, (1.0, 1.9, 0.0, 2.05)), [edge], 0.0, 0.0, 2.5)
    assert len(ops) == 1
    assert ops[0].kind == "door"
    assert abs(ops[0].width - 0.9) < 0.02
    assert abs(ops[0].s0 - 1.0) < 0.02


def test_window_not_door():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)
    ops = find_openings(_rays_through(3.0, (2.5, 3.7, 0.9, 2.1)), [edge], 0.0, 0.0, 2.5)
    assert [o.kind for o in ops] == ["window"]
    assert abs(ops[0].width - 1.2) < 0.02


def test_unobserved_wall_is_not_opening():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)
    assert find_openings([], [edge], 0.0, 0.0, 2.5) == []


def test_narrow_gap_rejected():
    edge = Edge("v", 3.0, 0.0, 4.0, 10000, 0.005)
    assert find_openings(_rays_through(3.0, (1.0, 1.3, 0.0, 2.0)), [edge], 0.0, 0.0, 2.5) == []


def test_unsupported_edge_has_no_openings():
    edge = Edge("v", 3.0, 0.0, 4.0, 0, 0.02)          # raster-only edge, no wall plane
    assert find_openings(_rays_through(3.0, (1.0, 1.9, 0.0, 2.05)), [edge], 0.0, 0.0, 2.5) == []


def test_opening_spanning_whole_wall_rejected():
    edge = Edge("v", 3.0, 0.0, 0.6, 10000, 0.005)      # 0.6 m wall, rays through all of it
    assert find_openings(_rays_through(3.0, (0.0, 0.6, 0.0, 2.0), cam=(0.3, 1.4, 1.0)),
                         [edge], 0.0, 0.0, 2.5) == []
