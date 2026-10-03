# Why are the video and photo tiers inaccurate? Root-cause experiments

Branch `exp-tier-accuracy` (not merged). Run on 2026-10-03 on the three sample captures, with the **LiDAR
tier as the reference** (the sample has no tape ground truth). Every row is regenerable:

```
python scripts/make_photo_folders.py <stray capture> outputs/exp/photo_<name> [--doorway-shared]
python scripts/exp_photo_oracle.py   outputs/exp/photo_<name> <stray capture> baseline scale pose pose_scale all walls ...
python scripts/exp_video_oracle.py   <stray capture> outputs/exp/video_<name> baseline scale pose pose_scale pose_framescale all
python scripts/exp_photo_stitch_oracle.py outputs/exp/photo_<name>      # rooms at their TRUE positions
python scripts/exp_video_global_poses.py  <stray capture> <out.json> 20 chained:20:10 chained:12:6
python scripts/exp_scale_priors.py outputs/exp/photo_<name>/oracle_scale
python scripts/exp_depth_pro_focal.py <out.json> <stray captures...>
python scripts/exp_tables.py                                           # prints the tables below
```

## The method: "oracles"
To find out which part of a tier is wrong, we swap one part at a time for the LiDAR truth (an *oracle*)
and see how much the error drops.

| mode | camera poses | depth shape | metric scale | what is left |
|---|---|---|---|---|
| `baseline` | VGGT | VGGT | Depth Anything | the shipped tier |
| `scale` | VGGT | VGGT | **true** | everything but scale |
| `pose` | **LiDAR** | VGGT | Depth Anything | everything but poses |
| `pose_scale` | **LiDAR** | VGGT | **true** (one per run) | depth shape + room method |
| `pose_framescale` (video) | **LiDAR** | VGGT, **each frame scaled to LiDAR** | true per frame | VGGT depth shape + room method |
| `all` | **LiDAR** | **LiDAR depth of the same frames** | true | only the room-shape method (+ stitching for photos) |

Caveat on `pose`/`pose_scale` (video): VGGT chunks that break into separate pose *segments* each start at
their own scale (`merge_chunks` re-starts at scale 1.0), and these two modes apply one scale to all segments,
so they mix scales. `pose_framescale` is the clean "true poses" row.

**Metrics are noisy on 8-13 rooms.** Differences under ~10 percentage points in median wall error are not
meaningful. For multi-room plans, *footprint area* and *footprint IoU* are more reliable than matched-wall error
(jagged outlines pair walls differently even when the geometry is the same: the LiDAR-depth `all` row still
shows 25 % "wall error" on floor_only with IoU 0.75).

---

## Photo tier

### Decomposition (simulated photos picked from the sample video, 2-8 per room)
floor_only (10 photo rooms / 13 LiDAR rooms):

| mode | median wall err % | footprint err % | adj. precision / recall | max overlap m2 |
|---|---|---|---|---|
| baseline | 28.5 | +21.5 | 0.57 / 0.44 | 0.09 |
| scale (true) | 28.2 | +17.5 | 0.75 / 0.33 | 0.01 |
| pose (LiDAR) | 14.0 | -11.5 | 0.75 / 0.33 | 0.00 |
| pose_scale | 25.1 | -12.8 | 0.75 / 0.33 | 0.09 |
| all (LiDAR depth + poses) | 18.2 | +32.3 | 0.75 / 0.33 | 0.10 |

with_ceiling (8 / 9):

| mode | median wall err % | footprint err % | adj. precision / recall |
|---|---|---|---|
| baseline | 29.4 | +21.1 | 0.29 / 0.67 |
| scale (true) | 24.0 | +5.1 | 0.25 / 0.33 |
| pose (LiDAR) | 22.9 | -14.3 | 0.50 / 0.67 |
| pose_scale | 18.9 | -12.5 | 0.33 / 0.67 |
| all | 32.9 | +34.4 | 0.33 / 0.33 |

**In plain words:**
- **Scale is not the problem.** Giving the photo tier the true scale barely changes wall error (28.5 → 28.2,
  29.4 → 24.0). Depth Anything's per-room scale error is a median **6 %** (floor_only rooms).
- **Poses matter some** (28.5 → 14.0 and 29.4 → 22.9 with LiDAR poses), but even **perfect depth and poses
  leave 18-33 % wall error and +32-34 % room area**. With only 2-8 photos a room's floor is seen partially and
  its outline also swallows floor seen *through doors*.
- **Stitching is the dominant whole-property error** — the decisive experiment:

| perfect per-room data (`all`) | footprint IoU vs LiDAR | union area vs LiDAR |
|---|---|---|
| rooms at their **true positions** (no stitching) — floor_only | **0.675** | 39.4 vs 41.0 m2 (**-4 %**) |
| rooms at their **true positions** — with_ceiling | **0.686** | 38.5 vs 40.3 m2 (**-4.5 %**) |
| rooms **stitched** by our door matching — floor_only / with_ceiling | **0.097 / 0.228** | — |

  Placed correctly, the rooms' union is already **inside the ±8 % footprint gate**; our door-to-door placement
  destroys it.

### Bug found: photo links count garbage as matches
`roomscan/photo_links.py::_inliers` sums the RANSAC mask from `cv2.findFundamentalMat`. When the fit fails,
OpenCV returns `F = None` **and an uninitialised mask** (values like 24, 127, 253). Measured: a photo pair with
**13** ratio-test matches and **5** real homography inliers was reported as **882**; the two strongest "links"
on floor_only (1529, 1149 "matches") were of this kind. Stitching trusts these numbers. Opt-in fix:
`room_links(..., robust_masks=True)` (count masks of successful fits only, non-zero entries only); test
`test_failed_fundamental_fit_mask_is_not_counted`.
Effect (perfect per-room data, floor_only): adjacency **precision 0.75 → 1.0**, recall 0.33 → 0.22 — every false
connection gone; what remains shows the true problem: **the protocol's photos rarely see from one room into
another** (4 of 10 rooms have no genuine visual link).

### Candidate fixes measured
| candidate | result | verdict |
|---|---|---|
| room extent from the best wall plane on each side (`walls`) | area +80 % (perfect data), walls 47.6 % | rejected: picks the next room's wall seen through a door |
| room extent = single floor piece with most cameras (`core`) | area +150 % | rejected: 2-8 photos give too little floor to split |
| camera-height prior (1.4 m) as metric scale | 12.6 % vs **6.1 %** Depth Anything; better in 1/9 rooms | rejected |
| robust link counting | precision 0.75 → **1.0** | **keep** (bug fix) |
| 3D-point registration of linked rooms (`register`) | no gain on the shipped photo set: ~10 genuine matches between rooms, mostly wrong (3D gap ~7 m) | needs cross-room views |
| **protocol: one doorway photo copied into both rooms' folders** (`--doorway-shared`) + robust links + registration + **clip overlaps instead of pushing** | perfect data, floor_only: IoU 0.097 → **0.419**, footprint error +32 % → **+25.5 %**, max overlap **0** | **recommended direction**; not enough alone (rooms LiDAR found no door between stay unconnected) |

Same protocol change on the **real VGGT reconstructions** (floor_only, `--doorway-shared` folders):

| mode | footprint err % | adj. precision / recall | max overlap m2 | footprint IoU |
|---|---|---|---|---|
| shipped (`baseline`, original folders) | +21.5 | 0.57 / 0.44 | 0.09 | 0.25 |
| doorway photos + robust links (`baseline+robust`) | **+15.3** | **0.80** / 0.44 | 0.07 | 0.30 |
| + registration + clipping (`register+robust`) | -10.7 | 0.60 / 0.33 | **0.00** | 0.09 |

Registration helps with perfect data but **not with VGGT data**: each room is reconstructed at its own metric
scale (Depth Anything, ~6 % per room), and a rigid 2D fit cannot absorb a scale difference between rooms, so
placements go wrong (IoU 0.09). A *similarity* fit (with scale) is the next step, untested.

## Video tier

### Decomposition
floor_only (340-frame walk, 13 LiDAR rooms):

| mode | footprint m2 (LiDAR 41.0) | footprint IoU | frames used | Depth Anything scale err |
|---|---|---|---|---|
| baseline | 5.1 (-88 %) | 0.08 | 33 of 347 | +5.1 % |
| scale (true) | 4.7 | 0.06 | 33 | — |
| pose (LiDAR, mixed segment scales) | 46.4 | 0.58 | 347 | -3.8 % |
| **pose_framescale** (LiDAR poses, VGGT depth per-frame true scale) | **42.5 (+3.6 %)** | **0.60** | 347 | — |
| all (LiDAR depth + poses of the picked frames) | 46.9 (+14 %) | **0.75** | 347 | — |

single_room: baseline 5.3 m2 vs 12.25 (IoU 0.23, 35 of 112 frames); `pose_framescale` **12.26 m2** (IoU 0.30);
`all` 17.5 m2 with the walkable-path widening, **12.5 m2 / IoU 0.85 / walls 3.3 %** without it (back end on perfect
data, `scripts/exp_video_backend.py`). On VGGT data the widening is needed (without it the area collapses to
1 m2), so it is not a fix by itself.

**In plain words:**
- **Scale is not the problem** (Depth Anything 4-13 % off; true scale changes nothing).
- **The camera poses are.** VGGT runs on 20-frame chunks glued by their overlap; the glue breaks (mirror,
  glass, fast turns) into **5 segments**, and the tier keeps only the most consistent one — **35 % of the walk**
  on floor_only (median rotation error 5.3°, trajectory error 0.17 m within it). With true poses the same
  VGGT depth gives the whole flat within **+3.6 %** of LiDAR area.
- VGGT's per-frame depth scale varies by **12-18 %** between frames (min/max 1.23-3.35), so even with true poses
  one scale per walk is not enough — per-frame or per-chunk scale alignment is needed.

### Candidate fixes measured
| candidate | result | verdict |
|---|---|---|
| one VGGT pass on 20 keyframes spread over the whole walk | ATE **3.07 m**, rotation error **146°** | rejected: frames too far apart to relate |
| chunk 20 / overlap 10 (shipped: 20 / 5) | keeps **82 %** of the walk in one piece, but ATE **2.0 m**, rotation **41°** | rejected: the breaks were catching genuinely wrong poses |
| chunk 12 / overlap 6 | keeps 57 %, ATE 0.54 m (6.3 % of the flat), rotation 13° | trade-off only, no fix |
| shipped chunking, for reference | keeps 35 % (floor_only) / 32 % (with_ceiling), ATE 0.17 / 0.35 m, rotation 5.3° / 11.2° | |
| Depth Pro with the known focal (vs Depth Anything), 18 frames over 3 captures | Depth Pro ratio to LiDAR 0.25-0.31, spread ±40 % (our wrapper of the HF `predicted_depth` is likely not plain depth before post-processing — **inconclusive**, not a fair test); calibrated Depth Anything **0.977, spread 7.5 %** | keep Depth Anything; Metric3D v2 / UniDepth not tried (would change the shared venv's torch pins) |

## Recommended fixes and predicted numbers
### Photo tier — strongest fix-loop candidate
**Root cause (evidence above):** the whole-property plan fails because rooms are **placed** wrong, not because
rooms are measured wrong (true positions: union area -4 %, IoU 0.68; stitched: IoU 0.10-0.23). Two causes:
1. a **bug**: link strength counts uninitialised OpenCV RANSAC masks, so stitching trusts fake links
   (882-1529 "matches" from 13 real; adjacency precision 0.57);
2. the protocol's photos rarely see from one room into the next, so true links are few (recall 0.33-0.44).

**Fix to ship:** (a) robust link counting (code, opt-in today: `robust_masks=True`); (b) capture protocol:
*"stand in each doorway and take one photo; copy it into BOTH rooms' folders"* (simulated by
`--doorway-shared`); (c) clip a room's over-extension instead of pushing rooms apart (code exists for
registered placements).

**Predicted after the fix** (gate: photo-tier whole-property stitch — one plan, correct adjacency, no overlaps,
footprint within ±8 %): on the simulated benchmark the measured effect of (a)+(b) is footprint **+21.5 % → +15.3 %**,
adjacency precision **0.57 → 0.80**, overlaps ≤ 0.07 m2. **Prediction for the real S23 photo set: footprint error
10-20 %, adjacency precision ≥ 0.8, overlaps ≤ 0.1 m2 — the ±8 % footprint row is still predicted to FAIL**;
the adjacency/no-overlap rows to pass. Reaching ±8 % needs every room linked (recall) and a scale-aware
(similarity) placement; with perfect placement the method is at -4 to -4.5 %.

### Video tier — root cause clear, fix not in reach before the deadline
**Root cause:** pose chaining. VGGT's 20-frame chunks break into ~5 pose segments (mirror/glass, fast turns);
the tier keeps the best one, i.e. **32-35 % of the walk** → the plan covers one or two rooms (5 m2 of 41 m2).
With true poses the same depth gives the flat within **+3.6 %** area (IoU 0.60). Scale is fine (Depth Anything
4-13 %). Not fixable by chunk tuning (more overlap → 82 % coverage but 41° rotation errors) or by one global
pass on sparse keyframes (146° errors).
**Fix that would work:** a real global alignment — loop closure / pose graph over all chunks, or COLMAP/GLOMAP
SfM on the picked frames with VGGT depth for density — plus per-chunk scale alignment (VGGT depth scale varies
12-18 % between frames). **Predicted:** video footprint from ~-88 % to within ~±15 % (bounded by the
`pose_framescale` oracle, +3.6 % / IoU 0.60), walls still above the ±3 % gate (the back end on perfect data
reaches 3.3 % on one room only). Estimated 4-8 h of work: **not a good fix-loop pick for this deadline**;
document as a known failure with this evidence.

### Costs (RTX 2000 Ada 8 GB)
VGGT-1B bf16: 20 frames ≈ 5.4 s, ~5.5 GB; whole video tier on floor_only (347 frames) ≈ 2-4 min. Photo tier
11 rooms ≈ 2 min (+1 min with registration). Depth Anything V2 Metric-Indoor-Large ≈ 1.8 GB. Running two GPU
models at once overflows 8 GB (one CUDA OOM observed) — experiments were queued one at a time.
Registration and robust links: CPU only, < 1 min per property.
