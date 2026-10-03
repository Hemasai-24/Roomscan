# Fix loop — declaration (written before the fix; approved 2026-10-03 ~15:10)

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
doorways** into the next room — the room "leaks" into its neighbours.

**Evidence:**
- Each room's bounding box vs tape: bedroom 1.20× / 0.93×, hall 1.65× / 1.24×, kitchen **2.35× / 1.85×**,
  bathroom **2.47× / 1.21×**. The kitchen comes out 4.4 m long; it is 1.9 m.
- Scale explains only a small part: the highest wall points we saw (3.18 / 3.04 / 3.10 / 2.65 m) against the
  real 2.74 m ceiling put our metric scale at about **+0 % to +16 %** — not 1.2-2.5×.
- Earlier oracle experiment (`docs/experiments/tier_accuracy.md`): with **perfect LiDAR depth and poses** for
  the same photos, the footprint was still **+32 %** — the error is in how the room shape is built, not in the
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

## 4. Result (after the fix)
*To be filled after the fix ships: before/after table, readable diff, and why it did or did not reach the gate.*
