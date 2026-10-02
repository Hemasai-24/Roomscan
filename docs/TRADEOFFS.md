# Trade-offs and limitations

Read this first: it lists every place this submission departs from the ideal case in the
brief, why, and what that does to the reported numbers. Kept up to date as work proceeds.

## Constraints we worked under
| Constraint | Consequence |
|---|---|
| No iPhone available (author's phone is a Samsung Galaxy S23, no LiDAR) | Could not record our own LiDAR captures, a Polycam LiDAR export, or iPhone photo/video |
| No ground truth exists for the provided sample data (confirmed by the hiring team, 2026-10-02) | Sample-data numbers are consistency checks, not accuracy |
| ~42 hours from brief to submission | Scope triaged by score weight; see "Not done" |

## What we did instead
| Brief requirement | What we did | Effect on the numbers |
|---|---|---|
| Capture route | Route 2: Stray Scanner (free; same format as the sample data) + one-page protocol | None; the protocol was not tested by us on an iPhone |
| LiDAR-tier accuracy vs laser ground truth | Planned: ARKitScenes scenes (iPad Pro LiDAR, laser-scanned reference) | iPad Pro, not iPhone; disclosed dataset |
| Video and photo tiers on the sample apartment | Simulated from the sample `rgb.mp4` (video: RGB only, no depth/poses; photos: 2-8 sharp frames per room) | Video frames are sharper and more evenly spaced than a typical handheld clip; photos are video frames, not stills |
| Our own photo/video benchmark with tape ground truth | Planned: author's rooms, Samsung S23 main camera (1x), tape measure | Android camera instead of iPhone 15; tape ±2-3 mm, not laser |
| Repeatability (same room twice, same tier) | The two whole-floor sample scans of the same apartment | Different walk paths, so this is a harder test than two identical captures |

## Not done (and why)
| Requirement | Status | Why |
|---|---|---|
| Staged damage room captured at all tiers | Pending | Needs a physical room + phone; S23 attempt planned |
| Head-to-head vs Polycam at the LiDAR tier | Pending | Needs a LiDAR iPhone |

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

## Plan 2 results (LiDAR tier, sample data, 2026-10-03)
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
