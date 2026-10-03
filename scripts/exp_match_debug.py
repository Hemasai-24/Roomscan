"""Compare photo_links' match counting with photo_register's matcher on two room folders."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.photo_folders import load_room_images            # noqa: E402
from roomscan.photo_links import _features, _inliers           # noqa: E402
from roomscan.photo_register import _matches                   # noqa: E402

out, ra, rb = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
A, _, na, _ = load_room_images(out / "rooms" / ra)
B, _, nb, _ = load_room_images(out / "rooms" / rb)
print("images", A.shape, B.shape, na, nb)
fa, fb = _features(A), _features(B)
m = cv2.BFMatcher(cv2.NORM_L2)
for p in range(len(A)):
    for q in range(len(B)):
        li = _inliers(fa[p], fb[q], m)
        xa, _ = _matches(A[p], B[q])
        if li > 20 or len(xa) > 20:
            print(f"{na[p]} - {nb[q]}: links-inliers {li}, register-matches {len(xa)}")
