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
