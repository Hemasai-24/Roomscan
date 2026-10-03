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
  Metric-Indoor** (it reads depth 1.41x too far on the sample phone → calibrated, leave-one-out
  scale error -2% to +4%); package as a Stray-like capture → **same back end** as LiDAR.
- **Fitting an 8 GB GPU:** VGGT ran out of memory as shipped; works with the backbone in bf16
  and only the 4 layers the heads use (20 frames: 5.5 s, 5.5 GB).
- **Honest result:** poor on the sample — chunks glue together with drift, and the bathroom
  mirror flips poses by 110-165°; only one room's worth of the walk survives. Footprint overlap
  with LiDAR 0.08-0.22; gate (±3%) **not met**; ranges widened. Inside a good piece the scale is
  right (camera height 1.43 m vs 1.45 m LiDAR).
- **More time:** global alignment of all frames instead of chaining chunks; mask mirrors; try
  MASt3R-SfM (not tried, to save setup time).

## 9. Photo tier
- **Per room:** VGGT on all 2-8 photos of the room at once (its best case, no chunking), lens
  focal length from EXIF when present, metres from Depth Anything; floor must be ≥ 0.5 m below
  the cameras, else assume 1.4 m camera height (with a warning). One room per folder.
- **Joining rooms (the hard gate):** two rooms are linked if a photo of one shows the inside of
  the other (feature matches through the doorway), then placed so their doors coincide, walls
  facing, one wall-thickness apart; any overlap is pushed out by the smallest move.
- **Honest result on the simulated benchmark (photos picked from the sample video):** one
  stitched plan, no overlaps (max 0.1 m²), but walls are off by a median **28%** vs LiDAR and
  total area **+21%**; ranges widened (σ 60%) so the LiDAR value falls inside 86-93% of the
  time when calibrated on the other capture. Gate (±8%) **not met**.
- **Why so far off:** 2-8 photos of white walls give VGGT little to work with; the room is
  measured from where cameras stand + visible floor, which over-/under-shoots.
  **More time:** per-room layout model (walls from the image directly), using door width
  (~0.8-0.9 m) and ceiling height priors as scale checks.

## 10. Our own benchmark and the fix loop (the 25 % part)
- **Benchmark:** our home (3 rooms + hall) with a Galaxy M53, tape-measured. Photo tier before: footprint +30 %
  (kitchen +145 %, bathroom +123 %). Clue: rooms 1.2-2.5× too big, but scale only ≤ 1.16× too big (walls
  "seen" up to 3.18 m in a 2.74 m room) → it's not scale.
- **Root cause:** a photo room = all floor its cameras see → floor seen **through doorways** counted (the
  kitchen "saw" the hall). Morning oracle test agreed: perfect data still +32 %.
- **Fix:** cut each room at its own walls. **v1** (nearest wall) over-cut big rooms (−65 / −82 %) because few
  photos give several copies of one wall up to 0.6 m apart. **v2:** a room's own wall can't lie between two
  of its own camera positions → nearest wall beyond the outermost cameras, ignoring one outlier (a doorway
  photo taken from the next room). **After: +7.5 %** (inside ±8 %); kitchen +42 %, bathroom +27 %.
- **Say honestly:** v2 was refined on the same data; the held-out check on the sample is confounded; wall
  lengths and adjacency still fail; prediction for small rooms (±25 %) was missed.

## 11. Process choices worth mentioning
- Tests first for every piece (synthetic rooms with known answers), then real data.
- Every deviation from a plan was written down as a "ruling" with its cost if wrong.
- A fresh reviewer checked Plan 1; 6 of its 8 important findings were fixed with tests.
- Limitations live in `docs/TRADEOFFS.md`, as the hiring team asked.
