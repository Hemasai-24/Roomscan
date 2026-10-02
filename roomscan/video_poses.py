"""Video tier, step V2: camera poses + relative depth from VGGT, in overlapping chunks of consecutive frames.

VGGT fits ~20 frames in 8 GB. Each chunk comes back in its own coordinates and scale; chunk k is glued
onto the frames already placed using the frames they share (rotation from their camera rotations, scale
from their depth ratio, shift from their camera centres). Frames whose rotation jumps implausibly (mirror
or glass confusion, seen on the sample bathroom) are dropped before gluing."""
import sys
from pathlib import Path

import numpy as np

MIN_SHARED = 2
SHARED_MAX_RESIDUAL_DEG = 20.0   # a shared frame disagreeing more than this after gluing is dropped


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def apply_similarity(sim, T):
    s, R, t = sim
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = s * R @ T[:3, 3] + t
    return out


def similarity_from_overlap(Ta, Tb, depth_ratio):
    """Similarity (s, R, t) mapping chunk-b coordinates onto chunk-a, from the same frames seen in both."""
    M = sum(a[:3, :3] @ b[:3, :3].T for a, b in zip(Ta, Tb))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    s = float(depth_ratio)
    t = np.mean([a[:3, 3] - s * R @ b[:3, 3] for a, b in zip(Ta, Tb)], axis=0)
    return s, R, t


def reject_flips(T, max_deg=75.0, cap_deg=120.0):
    """Valid mask. Split the sequence where consecutive frames turn > max_deg; keep the longest run, then
    accept neighbouring runs only if they continue plausibly from the last accepted frame."""
    n = len(T)
    cuts = [0] + [i for i in range(1, n) if rot_angle_deg(T[i - 1][:3, :3].T @ T[i][:3, :3]) > max_deg] + [n]
    runs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]
    main = max(range(len(runs)), key=lambda k: runs[k][1] - runs[k][0])
    valid = np.zeros(n, bool)
    a, b = runs[main]
    valid[a:b] = True
    last = b - 1
    for s, e in runs[main + 1:]:
        lim = min(max_deg * (s - last), cap_deg)
        if rot_angle_deg(T[last][:3, :3].T @ T[s][:3, :3]) <= lim:
            valid[s:e] = True
            last = e - 1
    first = a
    for s, e in reversed(runs[:main]):
        lim = min(max_deg * (first - (e - 1)), cap_deg)
        if rot_angle_deg(T[e - 1][:3, :3].T @ T[first][:3, :3]) <= lim:
            valid[s:e] = True
            first = s
    return valid


def _depth_ratio(da, ca, db, cb):
    m = (ca > 0) & (cb > 0) & (db > 1e-6) & (da > 1e-6)
    return float(np.median(da[m] / db[m])) if m.any() else np.nan


def _glue(placed, ch, idx, shared):
    """Similarity from shared frames; drop shared frames that disagree after gluing and refit once."""
    for _ in range(2):
        if len(shared) < MIN_SHARED:
            return None
        ratios = [_depth_ratio(placed[idx[k]]["depth"], placed[idx[k]]["conf"], ch["depth"][k], ch["conf"][k])
                  for k in shared]
        sim = similarity_from_overlap([placed[idx[k]]["T"] for k in shared], [ch["T"][k] for k in shared],
                                      float(np.nanmedian(ratios)))
        res = [rot_angle_deg(apply_similarity(sim, ch["T"][k])[:3, :3].T @ placed[idx[k]]["T"][:3, :3])
               for k in shared]
        keep = [k for k, r in zip(shared, res) if r <= SHARED_MAX_RESIDUAL_DEG]
        if len(keep) == len(shared):
            return sim
        shared = keep
    return sim if len(shared) >= MIN_SHARED else None


def merge_chunks(chunks):
    placed = {}          # frame index -> dict(T, depth, conf, K, seg)
    seg = 0
    for ch in chunks:
        idx = list(ch["idx"])
        shared = [k for k, i in enumerate(idx) if ch["valid"][k] and i in placed]
        if placed and len(shared) < MIN_SHARED:
            seg += 1
            sim = (1.0, np.eye(3), np.zeros(3))
        elif not placed:
            sim = (1.0, np.eye(3), np.zeros(3))
        else:
            sim = _glue(placed, ch, idx, shared)
            if sim is None:
                seg += 1
                sim = (1.0, np.eye(3), np.zeros(3))
        for k, i in enumerate(idx):
            if i in placed or not ch["valid"][k]:
                continue
            placed[i] = {"T": apply_similarity(sim, ch["T"][k]), "depth": ch["depth"][k] * sim[0],
                         "conf": ch["conf"][k], "K": ch["K"][k], "seg": seg}
    order = sorted(placed)
    return {"idx": np.array(order), "T": [placed[i]["T"] for i in order],
            "depth": np.stack([placed[i]["depth"] for i in order]),
            "conf": np.stack([placed[i]["conf"] for i in order]),
            "K": np.stack([placed[i]["K"] for i in order]),
            "seg": np.array([placed[i]["seg"] for i in order]), "segments": seg + 1}


class VGGTRunner:
    """Pretrained VGGT-1B (Meta, 2025). Backbone in bfloat16, camera/depth heads in float32; only the four
    token layers the heads read are kept, so 20 frames at 392x518 fit in 8 GB (spike: 5.5 GB, 5.5 s)."""
    KEEP_LAYERS = {4, 11, 17, 23}

    def __init__(self, root=Path(__file__).resolve().parents[1]):
        import torch
        sys.path.insert(0, str(root / "third_party" / "vggt"))
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.torch, self._decode = torch, pose_encoding_to_extri_intri
        m = VGGT(enable_point=False, enable_track=False)
        m.load_state_dict(torch.load(root / "weights" / "vggt-1b" / "model.pt", map_location="cpu"), strict=False)
        self.model = m.eval().cuda()
        self.model.aggregator.to(torch.bfloat16)

    def run(self, images):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(255).cuda()[None]
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks, ps = self.model.aggregator(x.to(torch.bfloat16))
            tl = [t.float() if i in self.KEEP_LAYERS else None for i, t in enumerate(toks)]
            del toks
            pose = self.model.camera_head(tl)[-1]
            depth, conf = self.model.depth_head(tl, images=x, patch_start_idx=ps)
        ext, intr = self._decode(pose, x.shape[-2:])
        T = []
        for E in ext[0].float().cpu().numpy():
            M = np.eye(4)
            M[:3, :4] = E
            T.append(np.linalg.inv(M))
        out = {"T": T, "depth": depth[0, ..., 0].float().cpu().numpy(), "conf": conf[0].float().cpu().numpy(),
               "K": intr[0].float().cpu().numpy()}
        torch.cuda.empty_cache()
        return out


def run_chunks(runner, images, chunk=20, overlap=5):
    n = len(images)
    starts = list(range(0, max(1, n - overlap), chunk - overlap))
    chunks = []
    for s in starts:
        idx = np.arange(s, min(s + chunk, n))
        r = runner.run(images[idx])
        r["idx"] = idx
        r["valid"] = reject_flips(r["T"])
        chunks.append(r)
    return merge_chunks(chunks)
