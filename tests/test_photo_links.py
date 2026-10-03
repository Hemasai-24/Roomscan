import cv2
import numpy as np

from roomscan.photo_links import room_links


def _texture(seed, size=180):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.integers(0, 255, (size, size)).astype(np.uint8), (0, 0), 1.5)
    t = cv2.equalizeHist(t)
    return np.dstack([t, np.roll(t, 7, 0), np.roll(t, 13, 1)])


def test_failed_fundamental_fit_mask_is_not_counted(monkeypatch):
    """cv2.findFundamentalMat returns F=None with an uninitialised mask when it fails; that garbage must not
    count as matches (it inflated links to >1000 'matches' from 13 real ones)."""
    from roomscan import photo_links as pl
    garbage = np.full((13, 1), 253, np.uint8)
    monkeypatch.setattr(pl.cv2, "findFundamentalMat", lambda *a, **k: (None, garbage))
    monkeypatch.setattr(pl.cv2, "findHomography", lambda *a, **k: (None, np.full((13, 1), 77, np.uint8)))
    rng = np.random.default_rng(0)
    d = rng.random((13, 128)).astype(np.float32)
    pts = rng.random((13, 2)).astype(np.float32) * 300
    fa, fb = (pts, d), (pts + 1, d + 1e-3 * rng.random(d.shape).astype(np.float32))
    assert pl._inliers(fa, fb, cv2.BFMatcher(cv2.NORM_L2), robust_masks=True) == 0
    assert pl._inliers(fa, fb, cv2.BFMatcher(cv2.NORM_L2)) > 13          # shipped default (bug kept for the record)


def _photo(base_seed, patch=None, at=(100, 150), size=(392, 518)):
    w, h = size
    img = np.full((h, w, 3), 200, np.uint8)
    img[40:40 + 180, 40:40 + 180] = _texture(base_seed)          # this room's own wall texture
    if patch is not None:
        y, x = at
        img[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return img


def test_rooms_sharing_a_view_are_linked_and_others_not():
    shared = cv2.resize(_texture(99), (150, 150))
    room_a = np.stack([_photo(1), _photo(2, shared, at=(300, 200))])
    room_b = np.stack([_photo(3, cv2.resize(shared, (170, 170)), at=(250, 180)), _photo(4)])
    room_c = np.stack([_photo(5), _photo(6)])
    links = room_links([room_a, room_b, room_c], min_matches=25)
    assert (0, 1) in links
    assert links[(0, 1)]["photo_i"] == 1 and links[(0, 1)]["photo_j"] == 0
    assert (0, 2) not in links and (1, 2) not in links


def test_links_deterministic():
    shared = cv2.resize(_texture(99), (150, 150))
    rooms = [np.stack([_photo(1, shared)]), np.stack([_photo(2, shared, at=(260, 120))])]
    assert room_links(rooms) == room_links(rooms)
