"""Photo tier, step P4: which rooms see each other. A photo taken in room A that looks through a door shows
the inside of room B, so it shares many reliable feature matches with B's photos. Matches are SIFT
features, kept only if clearly better than the second-best candidate (ratio test) and consistent with one
camera geometry (RANSAC); RANSAC is seeded so the same photos always give the same links."""
import cv2
import numpy as np

RATIO = 0.75


def _features(images):
    sift = cv2.SIFT_create(nfeatures=1500)
    out = []
    for img in images:
        kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
        out.append((np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des))
    return out


def _inliers(fa, fb, matcher):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    good = [m for m, n in (p for p in matcher.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return 0
    a = pa[[m.queryIdx for m in good]]
    b = pb[[m.trainIdx for m in good]]
    best = 0
    cv2.setRNGSeed(0)
    F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.99)
    if F is not None and mask is not None:          # a failed fit returns an uninitialised mask: never count it
        best = int((mask != 0).sum())
    cv2.setRNGSeed(0)
    Hm, mask = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
    if Hm is not None and mask is not None:
        best = max(best, int((mask != 0).sum()))
    return best


def room_links(images_by_room, min_matches=25):
    """{(i, j): {"matches", "photo_i", "photo_j"}} for room pairs i < j whose best photo pair has at least
    `min_matches` geometrically consistent feature matches."""
    feats = [_features(imgs) for imgs in images_by_room]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    links = {}
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            best = (0, None, None)
            for p, fa in enumerate(feats[i]):
                for q, fb in enumerate(feats[j]):
                    m = _inliers(fa, fb, matcher)
                    if m > best[0]:
                        best = (m, p, q)
            same = [(p, q) for p, a in enumerate(images_by_room[i]) for q, b in enumerate(images_by_room[j])
                    if a.shape == b.shape and np.array_equal(a, b)]
            if same:                                   # the protocol's doorway photo, copied into both folders
                best = (best[0], *same[0])
            if best[0] >= min_matches:
                links[(i, j)] = {"matches": best[0], "photo_i": best[1], "photo_j": best[2], "same_photo": bool(same)}
    return links
