"""Simulated photo tier benchmark (disclosed in docs/TRADEOFFS.md): choose "photos" from a LiDAR sample
capture's video the way the capture protocol asks a person to take them — from inside each room, facing
every direction, looking across the room, sharp — plus one photo facing each door."""
import numpy as np

AIM_DEG = 25.0


def pick_room_views(idx, yaw, sharp, depth, sectors=8):
    """One frame per viewing-direction sector: among the room's sharper half, the one that looks farthest."""
    idx = np.asarray(idx)
    sec = np.floor((np.asarray(yaw)[idx] + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    thr = np.median(np.asarray(sharp)[idx])
    out = []
    for s in range(sectors):
        cand = idx[(sec == s) & (np.asarray(sharp)[idx] >= thr)]
        if len(cand):
            out.append(int(cand[np.argmax(np.asarray(depth)[cand])]))
    return out


def pick_door_view(idx, cams, fwd, sharp, door_mid, dist=(0.5, 5.0)):
    """Sharpest frame whose view direction points at the door (within AIM_DEG), from a sensible distance."""
    idx = np.asarray(idx)
    to = np.asarray(door_mid) - np.asarray(cams)[idx]
    d = np.linalg.norm(to, axis=1)
    f = np.asarray(fwd)[idx]
    cos = np.einsum("ij,ij->i", f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12), to / (d[:, None] + 1e-12))
    ok = (cos > np.cos(np.radians(AIM_DEG))) & (d >= dist[0]) & (d <= dist[1])
    if not ok.any():
        return None
    cand = idx[ok]
    return int(cand[np.argmax(np.asarray(sharp)[cand])])
