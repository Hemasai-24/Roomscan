"""Score a plan against tape/laser ground truth: per-gate numbers and interval coverage.

Ground truth CSV (data/ground_truth/*.csv): room,item,id,value_cm,notes. Walls are named W1..Wn
clockwise; predicted walls are paired by the best cyclic order (either direction) when the counts
agree, otherwise by nearest length (and the mismatch is reported)."""
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

WALL_TOL = {"lidar": ("abs", 0.02), "video": ("rel", 0.03), "photo": ("rel", 0.08)}   # lidar: our assumption
OPENING_TOL = 0.02
OPENING_RATE = 0.85
CEILING_TOL = 0.015
MAX_OPENING_PAIR = 0.15


def load_ground_truth(path):
    gt = {}
    with open(Path(path)) as f:
        for row in csv.DictReader(f):
            if not row["value_cm"].strip():
                continue
            gt.setdefault(row["room"], {}).setdefault(row["item"], {})[row["id"]] = float(row["value_cm"]) / 100
    return gt


def _match_indices(gt_walls, pred_lengths):
    """[(ground-truth wall name, predicted wall index)]."""
    names = sorted(gt_walls, key=lambda k: int(k[1:]))
    truth = np.array([gt_walls[k] for k in names])
    pred = np.asarray(pred_lengths, float)
    if len(pred) == len(truth):
        idx = np.arange(len(pred))
        best = None
        for order in (idx, idx[::-1]):
            for s in range(len(order)):
                cand = np.roll(order, -s)
                err = np.abs(pred[cand] - truth).sum()
                if best is None or err < best[0]:
                    best = (err, cand)
        return list(zip(names, best[1].tolist()))
    r, c = linear_sum_assignment(np.abs(truth[:, None] - pred[None]))
    return [(names[i], int(j)) for i, j in zip(r, c)]


def match_room_walls(gt_walls, pred_lengths):
    pred = list(pred_lengths)
    return [(n, float(pred[j])) for n, j in _match_indices(gt_walls, pred)]


def _wall_ok(err, truth, tier):
    kind, tol = WALL_TOL[tier]
    return err <= (tol if kind == "abs" else tol * truth)


def score_plan(plan, gt, room_map):
    """room_map: ground-truth room name -> plan room id."""
    tier = plan["capture"]["tier"]
    rooms = {r["id"]: r for r in plan["rooms"]}
    w_err, w_cov, w_ok, count_mismatch = [], [], [], []
    c_err, c_cov = [], []
    within, missed, phantom, n_gt_open = 0, 0, 0, 0
    for name, rid in room_map.items():
        g, r = gt.get(name, {}), rooms[rid]
        if "wall" in g:
            pred = [w["length"] for w in r["walls"]]
            if len(pred) != len(g["wall"]):
                count_mismatch.append(name)
            for wname, j in _match_indices(g["wall"], [m["value"] for m in pred]):
                t, m = g["wall"][wname], pred[j]
                e = abs(m["value"] - t)
                w_err.append(e)
                w_cov.append(m["lo"] <= t <= m["hi"])
                w_ok.append(_wall_ok(e, t, tier))
        if "ceiling_height" in g:
            t = float(np.mean(list(g["ceiling_height"].values())))
            m = r["ceiling_height"]
            c_err.append(abs(m["value"] - t))
            c_cov.append(m["lo"] <= t <= m["hi"])
        truth_w = list(g.get("door_width", {}).values()) + list(g.get("window_width", {}).values())
        pred_w = [o["width"]["value"] for o in r["openings"]]
        n_gt_open += len(truth_w)
        if truth_w and pred_w:
            cost = np.abs(np.subtract.outer(truth_w, pred_w))
            ri, ci = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(ri, ci) if cost[i, j] <= MAX_OPENING_PAIR]
        else:
            pairs = []
        within += sum(cost[i, j] <= OPENING_TOL for i, j in pairs) if pairs else 0
        missed += len(truth_w) - len(pairs)
        phantom += len(pred_w) - len(pairs)
    denom = n_gt_open + phantom
    rate = within / denom if denom else None
    return {
        "tier": tier,
        "walls": {"n": len(w_err), "max_abs_err_m": max(w_err) if w_err else None,
                  "mean_abs_err_m": float(np.mean(w_err)) if w_err else None,
                  "within_gate": int(sum(w_ok)), "coverage": float(np.mean(w_cov)) if w_cov else None,
                  "count_mismatch_rooms": count_mismatch},
        "ceiling": {"n": len(c_err), "abs_err_m": max(c_err) if c_err else None,
                    "coverage": float(np.mean(c_cov)) if c_cov else None},
        "openings": {"n_truth": n_gt_open, "within_2cm": int(within), "missed": missed, "phantom": phantom,
                     "rate": rate},
        "gates": {"wall_lengths": bool(w_ok) and bool(all(w_ok)),
                  "opening_widths": bool(rate is not None and rate >= OPENING_RATE),
                  "ceiling_height": bool(c_err) and bool(max(c_err) <= CEILING_TOL)},
    }
