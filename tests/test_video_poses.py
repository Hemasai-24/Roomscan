import numpy as np
from scipy.spatial.transform import Rotation

from roomscan.video_poses import apply_similarity, merge_chunks, reject_flips, similarity_from_overlap


def _traj(n, step_deg=5.0):
    out = []
    for i in range(n):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("yx", [step_deg * i, 3 * np.sin(i / 3)], degrees=True).as_matrix()
        T[:3, 3] = [0.1 * i, 0.01 * np.sin(i), 0.05 * i]
        out.append(T)
    return out


def _sim(s, yaw_deg, t):
    return s, Rotation.from_euler("yz", [yaw_deg, 10], degrees=True).as_matrix(), np.asarray(t, float)


def _inv_sim(sim):
    s, R, t = sim
    return 1 / s, R.T, -R.T @ t / s


def test_similarity_recovers_known_transform():
    truth = _traj(5)
    sim = _sim(2.5, 30, [1, -2, 0.5])
    b = [apply_similarity(_inv_sim(sim), T) for T in truth]
    S = similarity_from_overlap(truth, b, depth_ratio=2.5)
    for Tb, Ta in zip(b, truth):
        np.testing.assert_allclose(apply_similarity(S, Tb), Ta, atol=1e-6)


def test_flip_rejected():
    T = _traj(20)
    bad = list(range(12, 19))
    flip = Rotation.from_euler("x", 150, degrees=True).as_matrix()
    for i in bad:
        T[i] = T[i].copy()
        T[i][:3, :3] = T[i][:3, :3] @ flip
    valid = reject_flips(T, max_deg=60)
    assert [i for i in range(20) if not valid[i]] == bad


def _chunk(idx, truth, sim, depth=2.0):
    inv = _inv_sim(sim)
    n = len(idx)
    return {"idx": np.array(idx), "T": [apply_similarity(inv, truth[i]) for i in idx],
            "depth": np.full((n, 6, 8), depth / sim[0], np.float32), "conf": np.full((n, 6, 8), 5.0, np.float32),
            "K": np.tile(np.eye(3), (n, 1, 1)), "valid": np.ones(n, bool)}


def test_merge_three_chunks_consistent():
    truth = _traj(50)
    sims = [_sim(1.0, 0, [0, 0, 0]), _sim(0.4, 70, [3, 1, 0]), _sim(3.0, -40, [-1, 0, 2])]
    chunks = [_chunk(range(0, 20), truth, sims[0]), _chunk(range(15, 35), truth, sims[1]),
              _chunk(range(30, 50), truth, sims[2])]
    m = merge_chunks(chunks)
    assert m["segments"] == 1
    assert list(m["idx"]) == list(range(50))
    for i, T in zip(m["idx"], m["T"]):                    # merged lives in chunk 0's frame
        np.testing.assert_allclose(apply_similarity(sims[0], T), truth[i], atol=1e-6)
    np.testing.assert_allclose(m["depth"] * sims[0][0], 2.0, rtol=1e-5)


def test_unalignable_chunk_starts_new_segment():
    truth = _traj(35)
    c0 = _chunk(range(0, 20), truth, _sim(1.0, 0, [0, 0, 0]))
    c1 = _chunk(range(15, 35), truth, _sim(2.0, 50, [1, 1, 1]))
    c1["valid"][:5] = False                      # every overlap frame unusable
    m = merge_chunks([c0, c1])
    assert m["segments"] == 2
    assert len(m["idx"]) == 35


def test_one_bad_shared_frame_does_not_bend_the_merge():
    truth = _traj(35)
    sims = [_sim(1.0, 0, [0, 0, 0]), _sim(2.0, 50, [1, 1, 1])]
    c0, c1 = _chunk(range(0, 20), truth, sims[0]), _chunk(range(15, 35), truth, sims[1])
    flip = np.eye(4)
    flip[:3, :3] = Rotation.from_euler("x", 120, degrees=True).as_matrix()
    c1["T"][2] = c1["T"][2] @ flip               # shared frame 17 is wrong in chunk 1 only
    m = merge_chunks([c0, c1])
    for i, T in zip(m["idx"], m["T"]):
        if i >= 20:
            np.testing.assert_allclose(apply_similarity(sims[0], T), truth[i], atol=1e-6)


def test_pick_segment_prefers_flattest_floor():
    from roomscan.video_pipeline import pick_segment
    assert pick_segment({0: (35, 0.034), 1: (27, 0.15), 2: (50, 0.18)}, min_frames=12) == 0
    assert pick_segment({0: (5, 0.01), 1: (27, 0.15)}, min_frames=12) == 1
