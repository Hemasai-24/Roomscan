"""Compare two plans of the same place (repeatability): align them, pair walls, diff lengths."""
import numpy as np


def _walls(plan):
    for r in plan["rooms"]:
        for w in r["walls"]:
            yield r["id"], w


def _mid(w, R2=np.eye(2), t=np.zeros(2)):
    return ((np.array(w["start"]) + np.array(w["end"])) / 2) @ R2.T + t


def align_plans(plan_a, plan_b):
    """Try the 4 right-angle turns; for each, shift centroids together, refine by nearest wall midpoints."""
    ma = np.array([_mid(w) for _, w in _walls(plan_a)])
    mb = np.array([_mid(w) for _, w in _walls(plan_b)])
    best = None
    for k in range(4):
        a = k * np.pi / 2
        R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).round(12)
        t = ma.mean(0) - (mb @ R2.T).mean(0)
        for _ in range(5):
            mb2 = mb @ R2.T + t
            d = np.linalg.norm(ma[:, None] - mb2[None], axis=2)
            j = d.argmin(1)
            ok = d[np.arange(len(ma)), j] < 0.5
            if ok.sum() == 0:
                break
            t = t + np.median(ma[ok] - mb2[j[ok]], axis=0)
        cost = np.median(np.linalg.norm(ma[:, None] - (mb @ R2.T + t)[None], axis=2).min(1))
        if best is None or cost < best[0]:
            best = (cost, R2, t)
    return best[1], best[2]


def match_walls(plan_a, plan_b, R2, t, max_mid=0.3):
    bs = list(_walls(plan_b))
    out = []
    for _, wa in _walls(plan_a):
        da = np.array(wa["end"]) - np.array(wa["start"])
        best = None
        for _, wb in bs:
            db = (np.array(wb["end"]) - np.array(wb["start"])) @ R2.T
            parallel = abs(abs(da @ db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9) - 1) < 0.01
            dm = np.linalg.norm(_mid(wa) - _mid(wb, R2, t))
            if parallel and dm < max_mid and (best is None or dm < best[0]):
                best = (dm, wb)
        if best:
            la, lb = wa["length"]["value"], best[1]["length"]["value"]
            diff = abs(la - lb)
            out.append({"a": wa["id"], "b": best[1]["id"], "len_a": la, "len_b": lb, "diff_m": round(diff, 4),
                        "pass": bool(diff <= 0.01 or diff <= 0.005 * max(la, lb))})
    return out


def _segs(plan):
    out = []
    for _, w in _walls(plan):
        a, b = np.array(w["start"], float), np.array(w["end"], float)
        out.append(((a + b) / 2, b - a))
    return out


def align_partial(plan_a, plan_b, tol=0.3):
    """Align a plan that covers only part of plan_a (e.g. video vs LiDAR): for each right-angle turn, try the
    shift that puts each wall of B on each parallel wall of A; keep the one landing most B walls on A walls."""
    A, B = _segs(plan_a), _segs(plan_b)
    ma = np.array([m for m, _ in A])
    da = np.array([d / (np.linalg.norm(d) + 1e-9) for _, d in A])
    best = None
    for k in range(4):
        ang = k * np.pi / 2
        R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]).round(12)
        mb = np.array([m for m, _ in B]) @ R2.T
        db = np.array([d / (np.linalg.norm(d) + 1e-9) for _, d in B]) @ R2.T
        par = np.abs(db @ da.T) > 0.99                       # (nb, na) parallel pairs
        for i, j in zip(*np.nonzero(par)):
            t = ma[j] - mb[i]
            d = np.linalg.norm((mb + t)[:, None] - ma[None], axis=2)
            d[~par] = np.inf
            near = d.min(1)
            score = (int((near < tol).sum()), -float(near[near < tol].sum()))
            if best is None or score > best[0]:
                best = (score, R2, t)
    _, R2, t = best
    mb = np.array([m for m, _ in B]) @ R2.T + t
    d = np.linalg.norm(mb[:, None] - ma[None], axis=2)
    j = d.argmin(1)
    ok = d[np.arange(len(mb)), j] < tol
    if ok.any():
        t = t + np.median(ma[j[ok]] - mb[ok], axis=0)
    return R2, t
