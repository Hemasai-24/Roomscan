"""Debug one room-pair registration of an oracle run (counts at each step; true transform for 'all' runs).
usage: python scripts/exp_register_debug.py <photo_out_dir> <oracle_mode> <room_a> <room_b>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.photo_folders import load_room_images                     # noqa: E402
from roomscan.photo_register import _frame_points, _matches, rigid2d_ransac   # noqa: E402
from roomscan.photo_room import measure_photo_room                      # noqa: E402
from roomscan.load_capture import load_stray                            # noqa: E402
from roomscan.room_outline import to_plan                               # noqa: E402

out, mode, ra, rb = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
ia = measure_photo_room(out / f"oracle_{mode}" / "photo_rooms" / ra, ra)
ib = measure_photo_room(out / f"oracle_{mode}" / "photo_rooms" / rb, rb)
print("cap_dir present:", "cap_dir" in ia, "cap_dir" in ib, "angles deg:", np.degrees(ia["angle"]), np.degrees(ib["angle"]))
imgs_a = load_room_images(out / "rooms" / ra)[0]
imgs_b = load_room_images(out / "rooms" / rb)[0]
ca, cb = load_stray(ia.get("cap_dir", out / f"oracle_{mode}" / "photo_rooms" / ra)), \
    load_stray(ib.get("cap_dir", out / f"oracle_{mode}" / "photo_rooms" / rb))
A, B = [], []
for p, x in enumerate(imgs_a):
    for q, y in enumerate(imgs_b):
        xa, xb = _matches(x, y)
        if len(xa) < 8:
            continue
        pa, pb = _frame_points(ca, p, xa, x.shape), _frame_points(cb, q, xb, y.shape)
        ok = ~(np.isnan(pa).any(1) | np.isnan(pb).any(1))
        d3 = np.linalg.norm(pa[ok] - pb[ok], axis=1)
        print(f"photo {p}-{q}: {len(xa)} matches, {ok.sum()} with depth, 3D gap median {np.median(d3) if ok.any() else np.nan:.3f} m")
        A.append(to_plan(pa[ok], ia["angle"]))
        B.append(to_plan(pb[ok], ib["angle"]))
A, B = np.concatenate(A), np.concatenate(B)
r = rigid2d_ransac(A, B)
print("pairs:", len(A), "registration:", None if r is None else (np.round(r[0], 3).tolist(), np.round(r[1], 3).tolist(), int(r[2].sum())))
