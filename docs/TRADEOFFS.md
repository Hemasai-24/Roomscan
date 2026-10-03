# Trade-offs and limitations

Read this first: it lists every place this submission departs from the ideal case in the
brief, why, and what that does to the reported numbers. Kept up to date as work proceeds.

## Assumptions (what the code takes for granted)
Each line: the assumption · where it matters · what breaks when it is false.

**Geometry**
- **Rooms are rectilinear** (walls meet at right angles, two main directions per capture) · outlines,
  wall snapping, photo room bounds · angled or curved walls are squared off.
- **One floor level per capture, gravity known** (ARKit / phone "up" = +Y) · floor/ceiling classification ·
  a split-level home or a stair void can be mis-labelled (seen once: a 6.14 m "ceiling" over the stairs).
- **A wall is vertical, flat and at least 1.4 m tall** · what blocks room splitting and bounds photo rooms ·
  tall furniture (wardrobes, fridges) counts as wall; furniture along walls makes outlines jagged.
- **An opening is where depth passes through a wall** · doors/windows · closed doors are invisible; mirrors and
  glass can create phantom openings.
- **A room's own wall never lies between two of its own photo positions** (fix loop v2) · photo room bounds ·
  photos taken from far outside a room (more than one) can stop a real wall being used.
- **The ceiling is the highest large horizontal plane above 1.9 m** · ceiling height · when no ceiling is seen
  we report a wide flagged range, never a confident number.

**Scale and cameras**
- **Depth Anything's bias (1.408, calibrated on the sample iPhone) holds for other phones** · video/photo
  metric scale · on the Galaxy M53 the scale came out ~0-16 % large.
- **The lens focal length in EXIF is right** (photo tier) · scale · edited/screenshotted photos lose EXIF; the
  model's own focal guess is used then (8-22 % low on the sample).
- **People hold the phone roughly upright** · first guess of "up" for video/photo · upside-down or rolled
  captures need `--rotate`.

**Our benchmark**
- **Tape measurements are ±0.5 cm** and rooms are rectangles (two lengths per room were measured).
- **Window values were written "width, height"**; we assumed that order (not stated by the measurer).
  Sill heights and a second ceiling spot were not measured.
- **LiDAR results on the sample data are a reference, not ground truth** (none exists for the sample).

**Uncertainty**
- **Ranges are 95 % and errors are independent per wall**; video/photo add a relative scale term fitted on
  the sample (video σ 0.25, photo σ 0.6) — honest on our home (truth inside 4/4 areas, 16/16 walls) but wide.

## Constraints we worked under
| Constraint | Consequence |
|---|---|
| No iPhone available (author's phone is a Samsung Galaxy M53, no LiDAR) | Could not record our own LiDAR captures, a Polycam LiDAR export, or iPhone photo/video |
| No ground truth exists for the provided sample data (confirmed by the hiring team, 2026-10-02) | Sample-data numbers are consistency checks, not accuracy |
| ~42 hours from brief to submission | Scope triaged by score weight; see "Not done" |

## What we did instead
| Brief requirement | What we did | Effect on the numbers |
|---|---|---|
| Capture route | Route 2: Stray Scanner (free; same format as the sample data) + one-page protocol | None; the protocol was not tested by us on an iPhone |
| LiDAR-tier accuracy vs laser ground truth | **Not done** (ARKitScenes was considered; no time) | LiDAR tier has no accuracy number against ground truth |
| Video and photo tiers on the sample apartment | Simulated from the sample `rgb.mp4` (video: RGB only, no depth/poses; photos: 2-8 sharp frames per room) | Video frames are sharper and more evenly spaced than a typical handheld clip; photos are video frames, not stills |
| Our own photo/video benchmark with tape ground truth | **Done:** author's home (3 rooms + hall), Samsung Galaxy M53 main camera (1x), tape measure | Android camera instead of iPhone 15; tape ±0.5 cm, not laser |
| Repeatability (same room twice, same tier) | The two whole-floor sample scans of the same apartment | Different walk paths, so this is a harder test than two identical captures |

## Own benchmark (2026-10-03)
Captured our own home (bedroom, hall, kitchen, bathroom) with a **Samsung Galaxy M53** — photos and videos —
and tape-measured every wall, door, window and ceiling (`data/ground_truth/`). Results and gates:
`docs/benchmark_report.md`. Caveats: tape not laser (±0.5 cm); an Android phone, not an iPhone 15; window
widths/heights assigned assuming "width first"; sill heights and a second ceiling spot not measured; staged
damage is small (stain 9 × 6 cm, crack 8 cm). The fix-loop rule (v2) was refined while looking at this same
capture (see `docs/fix_loop.md`).

**Tried and rejected — room shape, measured on our home (photo tier) and the LiDAR sample:**
| Change (branch) | Bedroom | Hall | Kitchen | Bathroom | Home total | LiDAR sample area |
|---|---|---|---|---|---|---|
| none (shipped) | −21 % | +10 % | +42 % | +27 % | +6 % | 41.0 m² |
| low furniture counts as floor (`low-furniture`) | **−6 %** | +25 % | +42 % | +27 % | +18 % | 47.4 m² |
| fill shallow furniture notches (`fill-furniture-notches`) | **−9 %** | +10 % | +74 % | +29 % | +13 % | 48.3 m² |
| UniDepth V2 metric depth with EXIF focal (`exp-metric-depth`) | +16 % | +61 % | +32 % | **−5 %** | +39 % | — |

Each fixes the room it targets and breaks others. UniDepth's **scale** is far better (0.998× LiDAR depth,
per-frame std 0.11, vs Depth Anything 1.425×, std 0.30 on the sample — `docs/experiments/metric_depth.md` on its
branch), but our room-shape rules use thresholds in metres that were tuned at the old scale. Conclusion: the
bottleneck is building a room's shape from a few photos; it needs a scale-robust redesign, not more rules.

**Tried and rejected: simpler outlines** (merge wall steps < 20-30 cm, `OUTLINE_EPS` / `MIN_JOG` in
`roomscan/room_outline.py`). Fewer walls (home 52 → 24-36; LiDAR 208 → 116-140) but not more accurate: mean
room-area error on our home 25 % → 27-29 %, main-wall error 25 % → 19-23 %, LiDAR repeatability no better.
Kept the original setting; the plan drawing hides wall labels under 0.5 m instead.

**Tried and rejected (measured on our home):** sharing doorway views between rooms automatically (branch
`auto-doorway-share`). Only hall↔kitchen had enough feature matches (28) to be detected; bedroom and bathroom
stayed unconnected, and the kitchen grew to +229 % because it absorbed hall floor from the shared photo. The
morning experiment's 3/3 connections needed the *identical* doorway photo placed in both folders by hand —
so the capture protocol asks for that, and automatic detection remains future work.

## Not done (and why)
| Requirement | Status | Why |
|---|---|---|
| Staged damage room captured at all tiers | **Partial:** captured at photo and video tiers (bedroom; stain 9 × 6 cm, crack 8 cm) | No LiDAR tier (no LiDAR phone); the staged damage is small |
| Head-to-head vs Polycam at the LiDAR tier | **Not done** | Needs a LiDAR iPhone |

## Engineering decisions with known costs
- **Own RANSAC instead of Open3D's.** Open3D's threaded `segment_plane` gave different planes on
  repeated runs of the same input despite a fixed seed. We use a seeded single-threaded NumPy
  RANSAC so the same capture always yields the same plan (repeatability gate). Cost: slower.
- **Unseen ceilings use a prior.** When the ceiling was never in view, we report it as at least
  the highest wall point seen, with the upper end at max(that + 0.3 m, 2.9 m), and flag
  `ceiling_observed: false`. The interval is wide on purpose.
- **Glass and mirrors (known gap).** Depth seen through glass or in a mirror looks like an
  opening in the wall. We only report openings on fitted walls and reject implausible sizes;
  explicit glass/mirror detection is not done yet, so a glass shower door can show as a door.
- **Manhattan (right-angle) room footprints.** Wall outlines are snapped to two perpendicular
  directions. Rooms with angled walls will be squared off.

## Plan 2 results (LiDAR tier, sample data, 2026-10-03 early morning — current numbers: `docs/benchmark_report.md` §2)
| Check | Result | Gate |
|---|---|---|
| Rooms found (single_room / floor-only / with-ceiling) | 2 / 10 / 8 | - |
| Runtime per capture (drift off) | 17 s / 34 s / 53 s | - |
| Drift on vs off, wall thickness (lower = sharper) | with-ceiling 58.9 -> 56.6 mm; floor-only 52.8 -> 54.8 mm; single_room 43.7 -> 37.6 mm | must show an ablation |
| Drift correction size | up to 2.9-4.2 deg / 12-23 cm on the whole-floor scans, 9-14 chunks rejected by the trust limit | - |
| Repeatability (floor-only vs with-ceiling, drift off) | 14 walls paired, 0 within 1 cm / 0.5% | fail |

What this means, honestly:
- **Drift correction is implemented and switchable** (`--drift-correction on|off`, default off) and the
  ablation script exists (`scripts/drift_ablation.py`). On synthetic data with injected drift it makes
  doubled walls ~2.6x sharper. On the real scans the effect is within noise: our sharpness measure
  (~40-60 mm on real data vs ~11 mm synthetic) is dominated by furniture near walls, so it cannot yet
  show whether real drift was removed. Default is off because on the short single_room scan it moved
  cameras 6.6 cm and split a room differently.
- **Repeatability fails.** The two whole-floor scans are split into rooms differently and room outlines
  still have many short steps, so wall pieces do not correspond one-to-one.
- A ceiling of 6.14 m appears in one room with drift on (likely the stair void / an upper-floor plane).

## Video tier (2026-10-03)
**Models used (pretrained, no training; weights fetched by `scripts/fetch_weights.py`):**
| Model | Version / source | License | Used for |
|---|---|---|---|
| VGGT-1B | `facebook/VGGT-1B` on Hugging Face, code `github.com/facebookresearch/vggt` (main, 2025) | weights: non-commercial research license (a separate `VGGT-1B-Commercial` checkpoint exists, gated) | camera poses + relative depth from video frames |
| Depth Anything V2 Metric-Indoor-Large | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` via transformers 4.56 | see model card (CC-BY-NC-4.0 for Large) | real-world scale only |

**Calibration on our data:** Depth Anything reads ~41% too far on the sample phone (bias 1.408, fitted on the
three LiDAR sample captures; leave-one-capture-out scale error -1.7% / -2.0% / +3.7%; same phone and flat, so
optimistic). Video ranges (`TIER_REL["video"] = 0.25`) were widened from the single_room comparison below and
are provisional until re-fitted on tape-measured videos.

**Measured (video tier vs LiDAR tier on the same sample capture; LiDAR is the reference, not ground truth):**
| Capture | Video rooms / area | LiDAR rooms / area | Footprint IoU | Walls >= 1 m within 3% | LiDAR length inside video 95% range |
|---|---|---|---|---|---|
| single_room (ranges fitted here) | 1 / 4.9 m2 | 2 / 12.3 m2 | 0.22 | 0 of 2 | 5 of 6 |
| floor_only (held out) | 1 / 5.2 m2 | 10 / 36.9 m2 | 0.08 | 0 of 3 | 4 of 14 |

What this means, honestly:
- **The video tier runs end to end but does not yet meet its gate (walls within 3%).** It produces a plan
  for only one piece of the walk.
- **Cause:** VGGT fits 20 frames at a time on our 8 GB GPU; chunks are glued by shared frames, and the glue
  drifts. Mirrors and glass (sample bathroom) flip whole chunks. The walk breaks into 3-5 pieces; the large
  pieces have smeared floors (10-26 cm), so the pipeline keeps the most consistent piece (33-35 frames).
- Inside a good piece poses are fine (rotation 1.4 deg, camera position 13 cm over a 4.2 m walk; camera height
  1.43 m vs 1.45 m from LiDAR, so the metric scale is close).
- Video sees little floor (~8% of points), so the walked camera path is used as known floor.
- Next steps with more time: a global pose graph over all chunks instead of chained gluing; a larger GPU (one
  VGGT pass over the whole walk); classic SfM (COLMAP) as a second opinion.

## Photo tier
**How it works:** each room folder (2-8 photos) goes through VGGT jointly (poses + relative depth), Depth
Anything V2 for metres (same calibrated bias as the video tier), EXIF focal length when present, then the
LiDAR back end measures ONE room (the floor region the cameras stand in). Rooms are linked when photos of one
show the inside of another (SIFT + ratio test + seeded RANSAC) and stitched at shared doors (detected door, or
the wall the linking photo looks at); a placement that would overlap is moved the smallest distance that
clears it. Same models and licences as the video tier.

**Benchmark is simulated (disclosed):** no photo set with ground truth exists for the sample flat, so
`scripts/make_photo_folders.py` picks protocol-like "photos" from each LiDAR room's video frames (inside the
room, one per viewing direction, sharp, looking across; plus one per detected door), with the true focal in
EXIF. The reference is the LiDAR plan of the same capture, not tape. The LiDAR rooms used as reference are
themselves fragments of the real rooms (Plan 2 limitations), which inflates some errors.

| Capture | Rooms photo / LiDAR | Footprint photo / LiDAR | Footprint IoU after alignment | Adjacency precision / recall vs LiDAR | Max overlap | Median wall error | LiDAR value inside photo range (areas / walls) |
|---|---|---|---|---|---|---|---|
| floor_only | 9 / 10 | 42.7 / 35.3 m2 (+21 %) | 0.21 | 0.50 / 0.43 | 0.096 m2 | 28 % | 9/9, 35/36 |
| with_ceiling | 7 / 8 | 44.4 / 36.5 m2 (+22 %) | 0.32 | 0.33 / 1.00 | 0.096 m2 | 29 % | 7/7, 28/28 |

Per-room area error vs LiDAR ranges from -69 % to +320 %; 6 of 16 rooms are within +-25 %.

What this means, honestly:
- **The photo tier stitches a whole-property plan from per-room folders, with no overlaps, but its
  dimensions do not meet the +-8 % gate.** Median wall error 28 % (before the fix loop; 38.7 % after, see `docs/fix_loop.md`).
- **Ranges are honest but very wide:** relative sigma 0.6 (from the 90th-percentile wall error, 117 %).
  Calibrated in-sample; leave-one-capture-out wall coverage 86 % and 93 %. Lower bounds clipped at 0.
- **Causes:** 3-8 photos leave wall sections unseen, so a room "leaks" into floor seen through doors (areas
  too big) or loses unseen corners (too small); VGGT's focal guess is 8-22 % low (EXIF fixes this on phones);
  few real doors are detected from photos, so most links use the wall a photo looks at, and many placements
  have to be moved 1-4 m to avoid overlaps, so the layout only loosely matches LiDAR (IoU 0.2-0.3).
- Rooms with < 2 photos are skipped and named in the warnings; rooms nobody can see from another room are drawn
  beside the plan with a warning (2 on floor_only).
- In real use the Galaxy M53/iPhone protocol (corner shots across the room) should cover walls better than these
  walkthrough frames; to be re-measured on our own tape-measured rooms.

## Damage

**Models (pretrained, no training, fetched by `scripts/fetch_weights.py`):**
| Model | Source | Revision | Licence | Use |
|---|---|---|---|---|
| Grounding DINO base | `IDEA-Research/grounding-dino-base` (Hugging Face) | `12bdfa3` | Apache-2.0 | boxes from the text "water stain. crack. mold. peeling paint. hole." |
| SAM 2.1 hiera-small | `facebook/sam2.1-hiera-small` (Hugging Face, transformers `Sam2Model`) | `ee5bba1` | Apache-2.0 | exact outline inside each box |

2.4 GB VRAM together, ~0.75 s per image (RTX 2000 Ada). Each mask is turned into 3D with the frame's depth
and pose and measured on the wall/floor/ceiling plane it lies on (area, width, height, height above floor).

**No damage in the sample flat, so we measured false alarms there and recall on painted damage:**
- Detector alone, 20 frames: false alarms 33 / 17 / 7 / 2 / 0 at score 0.25 / 0.30 / 0.35 / 0.40 / 0.45 (tile
  grout as "crack", plants and posters as "peeling paint", ceiling lights and vents as "hole"). A stain and a
  jagged line painted onto a frame score 0.44 / 0.46 — so a high threshold would also miss real damage. We keep
  0.35 and filter instead: a region must be seen in **2+ views** (LiDAR/video), **no "hole" on ceilings**,
  **floors keep only mould** (shiny tiles: reflections and grout — the brief's "wet-look surfaces"), and wall
  damage lying entirely in the **bottom 12 cm** (skirting-board joint) is dropped.
- Full pipeline on the three undamaged captures (2 views/s, max 150, 1280 px): **0 / 2 / 3 false regions**
  (single_room / floor_only / with_ceiling), from 1 / 10 / 4 before the floor and skirting rules. The 5 left are
  persistent look-alikes (a door edge, a poster, a vent, a shelf edge) — multiple views cannot reject those.
- **Painted damage with known truth** (`scripts/synthetic_damage_e2e.py`: 0.40 x 0.30 m stain and 0.60 m crack
  drawn in 3D on a real wall of with_ceiling, true poses, depth occlusion): the stain was visible in only 3 of
  150 views; it **was found** (2 views, right height) but split into 3 records because it sat on an inside
  corner next to a partition; their summed area 0.17 m2 vs 0.094 m2 true (pieces overlap). **The painted crack
  was not detected** (the detector finds jagged lines in a single close frame but not this one across views).
- Settings trade-off: at 1 view/s (60 views, 640 px) false alarms were 0/0/0 but the painted stain was seen in
  only one view and was dropped. We chose recall (2 views/s); the reviewer sees `n_views` and `score` on every
  region.

**Known limits:** thin cracks are weak; a stain on an inside corner is reported per wall; photo-tier damage is
kept from a single view (`single_view: true`) because a room has only 2-8 photos; photo rooms have no
`floor_level` (rule R5 cannot fire there); damage on a floor is only reported as mould. Real staged damage on
our own Galaxy M53 benchmark room (`data/ground_truth/own_rooms.csv`, `scripts/eval_damage.py`) is the true test.
