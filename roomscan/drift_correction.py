"""Drift correction: re-align the walk chunk by chunk so walls seen again land on walls seen before.

Each chunk gets a small turn about the vertical axis and a shift, solved together so its wall and
floor points land on the already-placed ones (point-to-plane ICP, Huber-weighted, damped). Corrections are
blended linearly between chunk centres. 'Poses used as-is' is an automatic fail in the brief."""
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from roomscan.point_cloud import backproject, estimate_normals

MATCH_DIST = 0.20
ITERS = 5
MAX_STEP_YAW = np.radians(2.0)     # a chunk may differ from the previous one by at most this much;
MAX_STEP_SHIFT = 0.10             # beyond it the matches are wrong (e.g. a newly seen room): keep previous
HUBER = 0.01
DAMPING = 0.05 ** 2   # directions the walls barely pin down (sliding along a lone wall) stay put


def _chunk_points(cap, frames, stride, max_depth=4.0):
    pts = [backproject(cap.load_depth(f.index), f.K, f.T_wc, max_depth) for f in frames[::stride]]
    p = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(p) > 60000:
        p = p[np.random.default_rng(0).choice(len(p), 60000, replace=False)]
    return p


def _rigid(yaw, shift, center):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center + shift
    return T


def _apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def _icp_step(p, n, yaw, shift, center, ref, tree):
    """One point-to-plane step for (yaw, shift): linearised least squares with Huber weights."""
    T = _rigid(yaw, shift, center)
    q = _apply(T, p)
    qn = n @ T[:3, :3].T
    dist, idx = tree.query(q, distance_upper_bound=MATCH_DIST)
    ok = np.isfinite(dist)
    ok[ok] &= np.abs(np.einsum("ij,ij->i", qn[ok], ref[1][idx[ok]])) > 0.9
    if ok.sum() < 200:
        return None
    rn, rq, qq = ref[1][idx[ok]], ref[0][idx[ok]], q[ok]
    gap = np.einsum("ij,ij->i", rq - qq, rn)
    rel = qq - (center + shift)
    # d/d(yaw) of a point: x += yaw * z_rel, z -= yaw * x_rel
    J = np.c_[rn[:, 0] * rel[:, 2] - rn[:, 2] * rel[:, 0], rn]
    w = np.ones(len(gap))
    for _ in range(3):
        A = J * w[:, None]
        x = np.linalg.solve(J.T @ A + DAMPING * w.sum() * np.eye(4), A.T @ gap)
        r = np.abs(gap - J @ x)
        w = np.where(r < HUBER, 1.0, HUBER / np.maximum(r, 1e-9))
    return x


def correct_drift(cap, chunk=300, stride=5):
    frames = cap.frames
    chunks = [frames[i:i + chunk] for i in range(0, len(frames), chunk)]
    ref = None            # placed points + normals so far
    params = []           # (yaw, shift, center) per chunk
    prev = (0.0, np.zeros(3))
    rejected = 0
    for ch in chunks:
        center = np.mean([f.T_wc[:3, 3] for f in ch], 0)
        p = _chunk_points(cap, ch, stride)
        if len(p) < 500:
            params.append((prev[0], prev[1].copy(), center))
            continue
        yaw, shift = prev[0], prev[1].copy()
        n = estimate_normals(p)
        if ref is None:
            yaw, shift = 0.0, np.zeros(3)
        else:
            tree = cKDTree(ref[0])
            for _ in range(ITERS):
                step = _icp_step(p, n, yaw, shift, center, ref, tree)
                if step is None:
                    break
                yaw += step[0]
                shift += step[1:]
            if abs(yaw - prev[0]) > MAX_STEP_YAW or np.linalg.norm(shift - prev[1]) > MAX_STEP_SHIFT:
                yaw, shift = prev[0], prev[1].copy()
                rejected += 1
        T = _rigid(yaw, shift, center)
        placed = (_apply(T, p), n @ T[:3, :3].T)
        ref = placed if ref is None else (np.concatenate([ref[0], placed[0]]), np.concatenate([ref[1], placed[1]]))
        if len(ref[0]) > 400000:
            keep = np.random.default_rng(0).choice(len(ref[0]), 400000, replace=False)
            ref = (ref[0][keep], ref[1][keep])
        params.append((yaw, shift.copy(), center))
        prev = (yaw, shift.copy())
    centres = np.array([(i + 0.5) * chunk for i in range(len(params))])
    yaws = np.array([q[0] for q in params])
    shifts = np.array([q[1] for q in params])
    cents = np.array([q[2] for q in params])
    out = deepcopy(cap)
    for k, f in enumerate(out.frames):
        y = np.interp(k, centres, yaws)
        s = np.array([np.interp(k, centres, shifts[:, a]) for a in range(3)])
        c = np.array([np.interp(k, centres, cents[:, a]) for a in range(3)])
        f.T_wc = _rigid(y, s, c) @ f.T_wc
    report = {"chunks": len(params), "max_yaw_deg": round(float(np.degrees(np.max(np.abs(yaws)))), 3),
              "max_shift_m": round(float(np.max(np.linalg.norm(shifts, axis=1))), 4),
              "rejected_chunks": rejected}
    return out, report


def wall_sharpness(points, normals, walls):
    """Median over walls of the robust thickness (1.4826*MAD, m) of points within 10 cm of the wall."""
    vals = []
    vert = np.abs(normals[:, 1]) < 0.3
    pv = points[vert]
    for w in walls:
        d = pv @ w.normal + w.d
        lo, hi = w.inliers.min(0), w.inliers.max(0)
        inb = np.all((pv >= lo - 0.1) & (pv <= hi + 0.1), axis=1)
        dd = d[(np.abs(d) < 0.10) & inb]
        if len(dd) > 100:
            vals.append(1.4826 * np.median(np.abs(dd - np.median(dd))))
    return float(np.median(vals)) if vals else float("nan")
