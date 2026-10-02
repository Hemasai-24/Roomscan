# Interview notes — how Roomscan works and why

Format per section: **what it does · why this way · what it gets wrong · with more time**.
Numbers are from real runs on the sample data unless marked synthetic.

---

## 0. The one-sentence architecture
Every input tier (LiDAR, video, photos) is turned into the **same thing** — depth images +
camera poses + lens intrinsics, in metres — and from there **one shared back end** finds
surfaces, rooms, walls, doors and ranges. Tiers differ only in their front end and in how wide
their ranges are. That is how "the same output contract from each tier" is met.

---

## 1. The data (Stray Scanner)
- **What:** free iOS app that exports raw sensor data: `rgb.mp4`, LiDAR `depth/*.png` (mm,
  256x192), `confidence/*.png` (0/1/2), `odometry.csv` (camera position + rotation per frame,
  from ARKit), lens intrinsics.
- **Pose** = where the camera was and which way it faced. ARKit's world has **Y up** (gravity).
- **Why Stray:** free, 1-minute install for a non-engineer, raw data out, and the sample data
  is in this format. Route 1 (own app) needed a Mac + iPhone we did not have.
- **Gotcha we verified on real data:** Stray's poses map **OpenCV-style** camera points
  (x right, y down, z forward) straight to the world — no ARKit axis flip. With the flip the
  walls smeared; without it the floor came out perfectly level (normal 0.00, 1.00, 0.00).

## 2. Depth → 3D points (back-projection)
- **What:** pixel (u, v) with depth z becomes camera point
  `x = (u − cx)/fx · z`, `y = (v − cy)/fy · z`, `z`; the pose moves it into the room.
- Intrinsics are given for the 1920x1440 video; depth is 256x192, so **scale fx, fy, cx, cy by
  256/1920**. Keep only confidence-2 pixels. Merge all frames, thin to one point per 2 cm.
- **Limits:** LiDAR range ~5 m; glass and mirrors give wrong depth (low confidence helps).

## 3. Finding surfaces (RANSAC planes)
- **What:** repeatedly pick 3 random points, make a plane, count points within 2 cm, keep the
  best; remove its points; repeat. Label: horizontal & lowest big one = **floor**, horizontal
  ≥ 1.9 m above floor = **ceiling**, vertical = **wall**, else **other** (tables, counters).
- **Why our own RANSAC:** Open3D's is multi-threaded and gave **different planes on repeated
  runs** (caught by a test: 1 in 4 runs differed). The brief's repeatability gate means same
  input → same output, so ours is seeded and single-threaded.
- **Bug found on real data:** a horizontal plane at 0.8 m slices through *every* wall and
  collects a thin band of wall points → fake "floors" at every height, and only 3 walls found.
  **Fix:** a point supports a plane only if its own surface normal faces the same way
  (|n·n_plane| > 0.85). Walls found: **3 → 32**; fake slices gone.
- **Floor in several pieces** (drift / slight tilt, bathroom 5 cm lower): pieces within 5 cm
  are merged; their height spread widens the ceiling-height range.

## 4. Rooms and outlines
- **Splitting rooms:** top-down picture of the floor, walls painted black; distance from each
  floor pixel to the nearest wall; pixels > 0.45 m from walls are "room cores"; grow cores
  until they meet (watershed) → the meeting lines are doorways. Merge two pieces sharing an
  open boundary > 1.2 m (one open-plan room). Only tall walls (≥ 1.4 m) block, so a kitchen
  counter doesn't cut a room. Sample floor-only scan: 10 rooms, 7 door connections.
- **Outline:** trace the room's floor, square it to the two main wall directions ("Manhattan"
  assumption), then **snap each side to the fitted wall plane** behind it — the plane is the
  average of thousands of points, which is where cm accuracy comes from.
- **What it gets wrong (honest):** outlines are still jagged (92 walls for 10 rooms) because
  furniture against walls makes the floor edge ragged; neighbouring rooms don't share a wall
  line exactly; angled walls are squared off.
- **With more time:** fit the outline from wall planes directly (polygon from wall lines),
  enforce shared walls between neighbouring rooms, minimum wall length.

## 5. Doors and windows (free-space carving)
- **What:** a depth ray that ends *behind* a wall line went **through** it → that spot of the
  wall is open. Group open spots into blobs. Bottom touches floor → **door**, else **window**.
  Reject blobs < 0.5 m wide or as wide as the whole wall; only on fitted walls.
- **Why:** a wall area the camera never looked at has no rays through it, so "unseen" is never
  mistaken for "open".
- **Gets wrong:** closed doors are invisible (flat like a wall); mirrors make phantom windows;
  glass shower doors can look like doors. **More time:** use the colour image (detector + SAM2)
  for closed doors and mirrors.

## 6. Ranges (uncertainty) — "honest intervals"
- Every number is `{value, lo, hi}`, a 95% range. Wall length range combines the uncertainty of
  the two walls that end it; area and perimeter propagate from wall offsets (each offset moves
  two wall lengths → perimeter σ doubled, a reviewer catch).
- **Tier floors:** LiDAR ≥ 0.8 cm per wall; video/photo get a relative scale term.
- **Honest unseen ceiling:** if no ceiling plane covers the room we report
  "at least <highest wall point>" up to max(+0.3 m, 2.9 m) and `ceiling_observed: false`.
- **Calibration** (to do with ground truth): widen ranges until ~95% of true values fall inside;
  the brief caps the score for "confident garbage".

## 7. Drift correction (brief: "poses used as-is is an automatic fail")
- **What drift is:** ARKit's position estimate slowly wanders over a long walk → the same wall
  seen at the start and end lands in two places ("doubled walls" — visible in the sample).
- **Our method:** cut the walk into ~5 s chunks; first chunk is the reference; each next chunk
  gets a small turn + shift (solved together, ICP-style) so its wall/floor points land on the
  already-placed ones; limited to 2° / 10 cm per chunk; blended smoothly between chunks.
- **Evidence:** synthetic room with injected 3° / 8 cm drift: wall thickness **34 → 13 mm**.
  Real sample scans: our sharpness metric barely moves (59 → 57 mm, 53 → 55 mm) because
  furniture near walls dominates it — so correction is **off by default** and the on/off
  comparison is reported as-is. **More time:** loop closure (the walks end 17 cm / 39 cm from
  where they started) with a pose graph; a better drift metric (wall-pair thickness only).

## 8. Video tier
- **Front end:** pick sharp frames (3 per second), estimate camera poses + depth with **VGGT**
  (Meta, 2025) in 20-frame chunks overlapping by 5; real-world scale from **Depth Anything V2
  Metric-Indoor** (it reads depth 1.41x too far on the sample phone → calibrated); package as a
  Stray-like capture → same back end.
- **Fitting an 8 GB GPU:** VGGT ran out of memory as shipped; works in bf16 with only the
  needed layers (20 frames: 5.5 s, 5.5 GB).
- **Honest result:** poor on the sample — chunks glue together with drift, and the bathroom
  mirror flips poses by 110-165°. Video-vs-LiDAR footprint overlap 0.08-0.22. Ranges widened.
  **More time:** global alignment instead of chaining (MASt3R-SfM style), mirror masking.

## 9. Photo tier
- (in progress) VGGT on each room's 2-8 photos together (its best case), scale from Depth
  Anything, one room per folder; rooms joined by matching doors (door widths + feature matches
  between one room's door photo and the next room's photos), no overlaps allowed.

## 10. Process choices worth mentioning
- Tests first for every piece (synthetic rooms with known answers), then real data.
- Every deviation from a plan was written down as a "ruling" with its cost if wrong.
- A fresh reviewer checked Plan 1; 6 of its 8 important findings were fixed with tests.
- Limitations live in `docs/TRADEOFFS.md`, as the hiring team asked.
