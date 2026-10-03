# Fix loop

Sections 1-3 were written and committed before the fix (tag `fix-before`). The original wording is unchanged
at that tag; this version is shorter. Section 4 was added after the fix (tag `fix-after`).

## 1. Worst gate and its number

Photo tier, whole-home footprint and wall lengths, gate ±8 %. Our home, Galaxy M53, 24 photos, at `fix-before`:

| Room | Tape | Photo tier | Error |
|---|---|---|---|
| bedroom | 8.37 m² | 6.62 m² | −21 % |
| hall | 15.56 m² | 18.54 m² | +19 % |
| kitchen | 2.86 m² | 7.01 m² | +145 % |
| bathroom | 2.87 m² | 6.42 m² | +123 % |
| whole home | 29.65 m² | 38.58 m² | +30 % |

Mean wall-length error was 70 cm; 2 of 16 walls were within the gate.

## 2. Cause and evidence

A photo room was measured as all the floor its cameras could see. That includes floor seen through a doorway
into the next room, so small rooms absorbed part of the hall.

- The kitchen came out 4.4 m long. Its real length is 1.9 m.
- Scale explains only a small part: the highest wall points seen were 2.65-3.18 m in rooms with a 2.74 m
  ceiling, so the scale is off by 0 to 16 %, not by a factor of 2.
- With perfect LiDAR depth and poses for the same views (sample data), the footprint was still +32 %. The
  error is in how the room shape is built, not in the models.

## 3. Fix and prediction

Fix: bound each room by its own walls. From the cameras, look along the room's two main directions and stop at
the nearest tall wall (at least 1.4 m high, so furniture is ignored). Floor beyond that wall is not part of the
room. Only the photo tier changes (`roomscan/photo_room.py`).

Prediction: whole-home footprint from +30 % to within ±15 %; kitchen and bathroom within ±25 %; the ±8 % wall
gate still fails because of the 0-16 % scale error.

## 4. Result

| Room | Before | After | Prediction |
|---|---|---|---|
| bedroom | −21 % | −21 % | - |
| hall | +19 % | +13 % | - |
| kitchen | +145 % | +42 % | within ±25 %: missed |
| bathroom | +123 % | +27 % | within ±25 %: missed narrowly |
| whole home | +30 % | +7.5 % | within ±15 %: met |
| mean wall error | 70 cm | 74 cm | - |

The footprint moved from fail to pass. Part of that is cancellation: the bedroom is too small and the other
rooms too big, and the summed per-room error is still about 19 %. Wall lengths still fail.

The first version of the fix (commit `4455a6e`) cut the large rooms far too much (bedroom −65 %, hall −82 %).
With only 5-9 photos, the plane finder returns several copies of the same wall up to 0.6 m apart, and one copy
ran between two of the bedroom's own camera positions. "Nearest wall" picked these inner copies. The shipped
version (commit `0cb3be8`) only accepts a wall that lies beyond the outermost camera, ignoring one outlier
camera such as a doorway photo taken from the next room. Tests cover the doorway leak, the wall copy between
cameras, and the outlier camera.

**Caveat.** The second version was adjusted while looking at this benchmark, so it may be tuned to it. A
check on the sample flat cannot settle this, because the LiDAR reference changed between the two runs.

## How to regenerate

```bash
git checkout fix-before && python run.py data/raw/m53 --out outputs/fix_loop/before_photo
git checkout fix-after  && python run.py data/raw/m53 --out outputs/fix_loop/after_photo_v2
git checkout main       && python scripts/score_m53.py
```

Code diff: `docs/fix_loop.diff`. Saved scores: `results/fix_loop_before_after.json`, `results/m53_benchmark.json`.
