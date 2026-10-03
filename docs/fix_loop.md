# Fix loop: declaration (written before the fix; approved 2026-10-03 ~15:10)

## 1. Worst-performing gate, with the failing number
**Photo tier: wall lengths and whole-property footprint (gate: within ±8 %).**
Our own benchmark: Samsung Galaxy M53, 4 rooms (bedroom, hall, kitchen, bathroom), 24 photos, tape-measured
(`data/ground_truth/own_rooms.csv`). Run at tag `fix-before`:

| Room | Tape area | Photo tier | Error |
|---|---|---|---|
| bedroom | 8.37 m² | 6.62 m² | −21 % |
| hall | 15.56 m² | 18.54 m² | +19 % |
| kitchen | 2.86 m² | 7.01 m² | **+145 %** |
| bathroom | 2.87 m² | 6.42 m² | **+123 %** |
| **total footprint** | **29.66 m²** | **38.59 m²** | **+30 %** |

Mean wall-length error 70 cm (16 walls, 2 within the gate).

## 2. Root-cause hypothesis and evidence
**Hypothesis:** a photo room is measured as *all floor its cameras see*, which includes floor seen **through
doorways** into the next room, the room "leaks" into its neighbours.

**Evidence:**
- Each room's bounding box vs tape: bedroom 1.20× / 0.93×, hall 1.65× / 1.24×, kitchen **2.35× / 1.85×**,
  bathroom **2.47× / 1.21×**. The kitchen comes out 4.4 m long; it is 1.9 m.
- Scale explains only a small part: the highest wall points we saw (3.18 / 3.04 / 3.10 / 2.65 m) against the
  real 2.74 m ceiling put our metric scale at about **+0 % to +16 %**, not 1.2-2.5×.
- Earlier oracle experiment (`docs/experiments/tier_accuracy.md` on branch `exp-tier-accuracy`): with **perfect LiDAR depth and poses** for
  the same photos, the footprint was still **+32 %**, the error is in how the room shape is built, not in the
  models.

## 3. The fix we intend to ship, and the predicted number
**Fix:** bound each room by its own walls. From the cameras' position, look along the room's two main
directions (both ways) and stop at the **nearest tall wall plane** (a wall seen from near the floor to ≥ 1.4 m,
so furniture is ignored). Floor beyond that wall is excluded from this room. Applies to the photo tier's
room measurement (`roomscan/photo_room.py`); LiDAR/video unchanged.

**Prediction (same 24 photos, tag `fix-after`):**
- total footprint from **+30 %** to **within ±15 %**;
- kitchen and bathroom from +145 % / +123 % to **within ±25 %** each;
- the **±8 % gate is expected to still fail**: the remaining error is dominated by the ~10-16 % metric-scale
  error on this phone (Depth Anything bias was calibrated on the sample data's iPhone).

## 4. Result (after the fix): tag `fix-after`

**Regenerate:** `git checkout fix-before && python run.py data/raw/m53 --out outputs/fix_loop/before_photo`, then
`git checkout fix-after && python run.py data/raw/m53 --out outputs/fix_loop/after_photo_v2`; then `git checkout dev && python scripts/score_m53.py` (the scorer was added after `fix-after`; it writes
`outputs/m53_benchmark.json`; committed copies: `results/m53_benchmark.json`, `results/fix_loop_before_after.json`). Readable diff: `docs/fix_loop.diff`
(`git diff fix-before..fix-after -- roomscan/photo_room.py tests/test_photo_room_clip.py`).

| Room (tape) | Before | After | Predicted |
|---|---|---|---|
| bedroom (8.37 m²) | 6.62 (−21 %) | 6.62 (−21 %) | n/a |
| hall (15.56 m²) | 18.54 (+19 %) | 17.53 (+13 %) | n/a |
| kitchen (2.86 m²) | 7.01 (**+145 %**) | 4.07 (**+42 %**) | within ±25 %, **missed** |
| bathroom (2.87 m²) | 6.42 (**+123 %**) | 3.65 (**+27 %**) | within ±25 %, **missed, narrowly** |
| **total footprint (29.65 m²)** | **38.58 (+30 %)** | **31.87 (+7.5 %)** | within ±15 %, **met** |
| mean wall-length error | 70 cm | 74 cm | not predicted |
| true value inside the 95 % range | 4/4 rooms | 4/4 rooms | n/a |

**Did the gate move from fail to pass?** Partly. The total is the sum of room areas, so per-room errors partly cancel (bedroom −21 % against hall +13 %, kitchen +42 %, bathroom +27 %; summed absolute error 5.7 m² ≈ 19 %), and it is not a stitched footprint (2 of 4 rooms are drawn beside the plan). The **total footprint is now inside ±8 %** (+7.5 %), but the
per-room wall lengths are still far outside ±8 % (mean error 74 cm; 2 of 16 walls within the gate), so the
row "wall lengths within ±8 %" still **fails**. The photo-stitch row also still fails on adjacency (1 of 3
connections; unchanged, not what this fix addressed).

**What happened, honestly (two iterations of the declared fix):**
1. **v1** (`4455a6e`, nearest tall wall from the cameras' centre) made the two small rooms much better
   (kitchen +42 %, bathroom +27 %) but **over-cut the large rooms** (bedroom −65 %, hall −82 %; total −55 %).
   Diagnosis on the saved reconstructions: with only 5-9 photos, each photo's depth places the same physical
   wall slightly differently, so the plane finder returns **several copies of one wall up to 0.6 m apart**,
   and in the bedroom one "tall wall" ran **between two of the room's own camera positions**, impossible for
   a real wall of that room. "Nearest" picked these inner copies.
2. **v2** (`0cb3be8`) keeps the declared idea and fixes its premise: a bounding wall must lie **beyond the
   outermost cameras** (one outlier camera ignored, e.g. a doorway photo taken from the next room). Tests:
   the leak through a door is cut, a wall copy between cameras is ignored, a doorway photo from the
   neighbouring room is ignored.

**Root cause verdict:** confirmed for the small rooms, floor seen through doorways was the dominant error
(kitchen +145 % → +42 %, bathroom +123 % → +27 %). The remaining error is (a) inconsistent wall copies from
few photos (bedroom unchanged: no wall beyond its cameras was found on two sides) and (b) the metric scale
(+0-16 % on this phone, see section 2).

**Prediction vs result:** total footprint predicted within ±15 % → +7.5 % (met, better than predicted); small
rooms predicted within ±25 % → +42 % / +27 % (missed). We also predicted the ±8 % gate would still fail:
true for wall lengths, false for the total footprint (it passed).

**Caveats we want the reader to have:**
- v2's rule was refined while looking at this benchmark, so it may be tuned to it. **Held-out check** on the
  sample flat (photos simulated from LiDAR frames, LiDAR as reference): footprint error +21.5 % → −12.0 %
  (smaller), median top-wall error 28.5 % → 38.7 % (worse), adjacency precision 0.67 / recall 0.44. The
  LiDAR reference itself changed between the two runs (13 vs 10 rooms after the small-room fix), so this
  check is confounded, **no clean evidence that v2 generalises**.
- Wall-length scoring is unreliable on jagged outlines (predicted rooms have 4-28 walls vs 4 measured);
  floor area per room is the more trustworthy metric here.
