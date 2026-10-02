import numpy as np
from roomscan.compare_plans import align_plans, match_walls


def _plan(poly, rot_deg=0.0, shift=(0, 0), scale=1.0):
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    P = (np.array(poly, float) * scale) @ R.T + shift
    walls = [{"id": f"w{i}", "start": list(P[i - 1]), "end": list(P[i]),
              "length": {"value": float(np.linalg.norm(P[i] - P[i - 1]))}} for i in range(len(P))]
    return {"rooms": [{"id": "r0", "polygon": P.tolist(), "walls": walls}]}


def test_same_room_rotated_and_shifted_matches_exactly():
    poly = [(0, 0), (4.2, 0), (4.2, 3.1), (2.0, 3.1), (2.0, 4.0), (0, 4.0)]
    a, b = _plan(poly), _plan(poly, rot_deg=90, shift=(10, -3))
    R2, t = align_plans(a, b)
    m = match_walls(a, b, R2, t)
    assert len(m) == 6 and all(x["pass"] for x in m)


def test_length_difference_detected():
    poly = [(0, 0), (4.0, 0), (4.0, 3.0), (0, 3.0)]
    a, b = _plan(poly), _plan(poly, scale=1.01)          # 1% longer walls: fails 0.5% / 1 cm gate
    R2, t = align_plans(a, b)
    assert not all(x["pass"] for x in match_walls(a, b, R2, t))
