"""Synthetic room point clouds for geometry tests. Plan coords (x, z); y is up."""
import numpy as np
from matplotlib.path import Path as MPath


def _grid(a0, a1, b0, b1, step):
    a = np.arange(a0, a1 + 1e-9, step)
    b = np.arange(b0, b1 + 1e-9, step)
    return np.meshgrid(a, b)


def room_points(footprint, height, step=0.02, ceiling=True, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    fp = np.asarray(footprint, float)
    xs, zs = _grid(fp[:, 0].min(), fp[:, 0].max(), fp[:, 1].min(), fp[:, 1].max(), step)
    inside = MPath(fp).contains_points(np.c_[xs.ravel(), zs.ravel()], radius=1e-9)
    fx, fz = xs.ravel()[inside], zs.ravel()[inside]
    parts = [np.c_[fx, np.zeros_like(fx), fz]]
    if ceiling:
        parts.append(np.c_[fx, np.full_like(fx, height), fz])
    for (x0, z0), (x1, z1) in zip(fp, np.roll(fp, -1, axis=0)):
        L = np.hypot(x1 - x0, z1 - z0)
        t, y = _grid(0, L, 0, height, step)
        t, y = t.ravel() / L, y.ravel()
        parts.append(np.c_[x0 + t * (x1 - x0), y, z0 + t * (z1 - z0)])
    p = np.concatenate(parts)
    return p + rng.normal(0, noise, p.shape)


def table_points(x0, z0, x1, z1, top=0.9, step=0.02):
    xs, zs = _grid(x0, x1, z0, z1, step)
    return np.c_[xs.ravel(), np.full(xs.size, top), zs.ravel()]


def two_rooms_with_door(door=(1.0, 1.9), door_h=2.05, h=2.5, step=0.02):
    """Room A x in [0,4], room B x in [4,7]; both z in [0,3]. Shared wall x=4 has a door at z in `door`."""
    a = room_points([(0, 0), (4, 0), (4, 3), (0, 3)], h, step=step)
    b = room_points([(4, 0), (7, 0), (7, 3), (4, 3)], h, step=step, seed=1)
    p = np.concatenate([a, b])
    hole = (np.abs(p[:, 0] - 4) < 0.03) & (p[:, 2] > door[0]) & (p[:, 2] < door[1]) & (p[:, 1] < door_h) \
        & (p[:, 1] > 0.02)
    return p[~hole]


def render_box_depth(K, T_wc, lo, hi, w, h):
    """Depth (metres, along camera z) seen from inside an axis-aligned box room lo..hi."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_w = d_cam @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    t = np.full(u.shape, np.inf)
    for ax in range(3):
        for b in (lo[ax], hi[ax]):
            with np.errstate(divide="ignore", invalid="ignore"):
                ti = (b - o[ax]) / d_w[..., ax]
            t = np.where((ti > 1e-6) & (ti < t), ti, t)
    return t      # d_cam has z = 1, so ray parameter t equals depth


def rooms_with_doors(rects, doors, h=2.5, step=0.02, door_h=2.05):
    """Axis-aligned rooms rects=[(x0,z0,x1,z1)], doors=[(x_wall, z0, z1)] holes in walls at x = x_wall."""
    p = np.concatenate([room_points([(a, b), (c, b), (c, d), (a, d)], h, step=step, seed=i)
                        for i, (a, b, c, d) in enumerate(rects)])
    for xw, z0, z1 in doors:
        hole = (np.abs(p[:, 0] - xw) < 0.03) & (p[:, 2] > z0) & (p[:, 2] < z1) & (p[:, 1] < door_h) & (p[:, 1] > 0.02)
        p = p[~hole]
    return p
