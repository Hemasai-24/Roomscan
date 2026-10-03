"""Photo tier, experimental stitching: place room B relative to room A from the photos that see both.

A feature matched between a photo of A and a photo of B has depth in both rooms' reconstructions, so it is one
3D point known in both rooms' coordinates. Both rooms are levelled (gravity = +Y), so the rooms differ by a turn
about the vertical axis plus a shift: a 2D rigid transform of the floor plan, fitted robustly (RANSAC) to
those points. Unlike door-to-door guessing, this uses where things actually are."""
import cv2
import numpy as np

from roomscan.load_capture import load_stray
from roomscan.point_cloud import backproject   # noqa: F401  (same camera model as the captures)
from roomscan.room_outline import to_plan

RATIO = 0.75
INLIER_M = 0.15
MIN_INLIERS = 12


def rigid2d_ransac(a, b, thresh=INLIER_M, iters=1000, seed=0):
    """(R, t, inlier mask) with a ~= b @ R.T + t, or None. Seeded: same input, same answer."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        db, da = b[j] - b[i], a[j] - a[i]
        if np.linalg.norm(db) < 0.2:
            continue
        ang = np.arctan2(da[1], da[0]) - np.arctan2(db[1], db[0])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        t = a[i] - R @ b[i]
        inl = np.linalg.norm(b @ R.T + t - a, axis=1) < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (R, t, inl)
    if best is None or best[2].sum() < 3:
        return None
    inl = best[2]
    ca, cb = a[inl].mean(0), b[inl].mean(0)
    H = (b[inl] - cb).T @ (a[inl] - ca)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    t = ca - R @ cb
    inl = np.linalg.norm(b @ R.T + t - a, axis=1) < thresh
    return R, t, inl


def _frame_points(cap, k, pix, img_shape):
    """3D points (capture frame) of pixels `pix` (x, y at image resolution) of frame k; NaN where no depth."""
    f = cap.frames[k]
    d = cap.load_depth(f.index, min_conf=0)
    sy, sx = d.shape[0] / img_shape[0], d.shape[1] / img_shape[1]
    u = np.clip((pix[:, 0] * sx).astype(int), 0, d.shape[1] - 1)
    v = np.clip((pix[:, 1] * sy).astype(int), 0, d.shape[0] - 1)
    z = d[v, u].astype(float)
    x = (u + 0.5 - f.K[0, 2]) / f.K[0, 0] * z
    y = (v + 0.5 - f.K[1, 2]) / f.K[1, 1] * z
    p = np.c_[x, y, z] @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]
    p[z <= 0.1] = np.nan
    return p


def _matches(img_a, img_b):
    sift = cv2.SIFT_create(nfeatures=3000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    return (np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 2),
            np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 2))


def register_rooms(info_a, imgs_a, info_b, imgs_b):
    """(R2, t, n_inliers) placing room B's plan coordinates into room A's, from all photo pairs; or None."""
    if "cap_dir" not in info_a or "cap_dir" not in info_b:
        return None
    ca, cb = load_stray(info_a["cap_dir"]), load_stray(info_b["cap_dir"])
    A, B = [], []
    for p, ia in enumerate(imgs_a):
        for q, ib in enumerate(imgs_b):
            xa, xb = _matches(ia, ib)
            if len(xa) < 8:
                continue
            pa = _frame_points(ca, p, xa, ia.shape)
            pb = _frame_points(cb, q, xb, ib.shape)
            ok = ~(np.isnan(pa).any(1) | np.isnan(pb).any(1))
            A.append(to_plan(pa[ok], info_a["angle"]))
            B.append(to_plan(pb[ok], info_b["angle"]))
    if not A:
        return None
    A, B = np.concatenate(A), np.concatenate(B)
    r = rigid2d_ransac(A, B)
    if r is None or r[2].sum() < MIN_INLIERS:
        return None
    return r[0], r[1], int(r[2].sum())
