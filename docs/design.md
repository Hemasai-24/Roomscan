> Original design, written 2026-10-02 before any code. What was built and measured differs in places — see
> `docs/technical_report.md` (current design) and `docs/TRADEOFFS.md` (assumptions and limitations).

# Floor-Plan Pipeline — Design Spec

Date: 2026-10-02 · Deadline: 2026-10-04 10:00 (~42 h) · Source brief: `Applied_AI.pdf`

## Goal
One command per capture (`python run.py <capture_dir>`) that turns a handheld iPhone
capture (photos, video, or LiDAR) into a JSON plan matching our published schema plus a
rendered whole-property floor plan, with a confidence interval on every measurement.
Must run cold on an unseen space at the defense.

## Decisions (and why)
- **Capture route: Route 2 (stock app + one-page protocol).** No Mac/TestFlight risk;
  pipeline runs on Linux. LiDAR tool: **Stray Scanner** (free; the provided sample data is
  in its export format: `rgb.mp4` 1920x1440@60 HEVC, `depth/*.png` 256x192 uint16 mm,
  `confidence/*.png` 0-2, `odometry.csv` per-frame ARKit pose + intrinsics, `imu.csv`).
  Photos and video: native Camera app.
- **Language/stack:** Python 3.10, Open3D, NumPy/SciPy, OpenCV, PyTorch (RTX 2000 Ada, 8 GB).
  Pretrained models (disclosed): Depth Anything V2 Metric (scale), MASt3R or COLMAP
  (photo/video poses), OWLv2 (damage detection). Weights fetched by `scripts/fetch_weights.sh`.
- **Primary data: the provided `sample_data/`** (one two-storey apartment):
  `single_room` (37 s, kitchen/utility), `single_scan_floor_only` (115 s, whole floor,
  little ceiling, loop gap 0.17 m), `single_scan_with_ceiling` (215 s, same floor,
  loop gap 0.39 m). Uses: multi-room stitch + drift ablation (both long scans), repeatability
  (the two long scans of the same rooms), honest-interval test (ceiling barely seen in
  `floor_only`).
- **Tier derivation from LiDAR captures (disclosed):** video tier = `rgb.mp4` only (depth,
  poses, intrinsics stripped); photo tier = 2-8 sharp stills per room sampled from the
  video into per-room folders. Our own iPhone captures, if obtained, replace these.
- **Ground truth:** none provided. Requested from the hiring team; until then the LiDAR
  tier is the reference for video/photo tiers, labelled as such. Own laser-measured
  captures (if a phone is obtained) supply true GT, the damage room and the Polycam
  head-to-head; otherwise those rows are marked not-met in the compliance matrix.
- Video is stored rotated 90° and HEVC random seek is unreliable: decode sequentially.
- **Honest intervals over point accuracy on thin tiers.** Calibration is scored at every
  tier; confident garbage caps the total score.

## Architecture
```
capture_dir ──► ingest/<tier>.py ──► Frames(rgb, depth?, pose?, K, scale_source)
                                        │
                     ┌──────────────────┴─────────────────┐
              geometry/recon.py                     damage/detect.py
       (TSDF fusion or MASt3R points)            (OWLv2 on keyframes)
                     │                                    │
              geometry/planes.py  ──── surfaces ──────►  damage/project.py
       (RANSAC walls/floor/ceiling,                   (box → wall-plane metric
        openings = wall-plane gaps)                    extent, surface id)
                     │                                    │
              geometry/room.py                       damage/rules.py
       (polygon, area, ceiling ht, intervals)     (concealed flags + rule id,
                     │                              scope line items per surface)
              stitch/stitch.py  (doorway matching across rooms, drift correction
                     │           toggle: --drift-correction on|off)
              output/schema.py + output/render.py  ──► plan.json, plan.svg/png
```
Each unit has one job and a dataclass interface; tiers differ only in `ingest/` and in
the uncertainty model, so all tiers share one output contract.

## Tier design
| Tier | Poses | Scale | Expected accuracy (to verify) |
|---|---|---|---|
| LiDAR | app ARKit poses, refined by ICP + plane anchoring | sensor depth | ~1–2 cm walls |
| Video | MASt3R/COLMAP on sharp keyframes | metric-depth model, median-fused | ±3% target |
| Photos (2–8/room) | MASt3R per room folder | metric-depth model | ±8% target |

## Stitching and drift
- LiDAR/video multi-room: one continuous trajectory; **drift correction = plane-anchored
  Manhattan alignment + pose-graph loop closure when the walk revisits a room**. Ablation:
  same capture with correction on/off, footprint error reported both ways.
- Photos: rooms reconstructed separately; adjacency from doorway openings matched across
  folders (shared features through the doorway + opening width compatibility); rooms
  placed by aligning matched doorways; overlap check rejects placements that overlap.

## Uncertainty
Every measurement emits `{value, lo, hi}` (95%). Interval = per-tier error model
(plane-fit residuals + scale uncertainty for monocular tiers), then **calibrated on our
benchmark**: report empirical coverage per tier; inflate until coverage ≥ 90%.

## Damage
Classes: water stain, crack, mould, hole (≥2 staged). OWLv2 text queries on keyframes,
projected onto the owning surface for metric extent (m²). Concealed-damage rules, e.g.
`R1 ceiling_stain_below_wet_room`, `R2 stain_at_wall_base → possible subfloor moisture`;
each flag records the rule id. Scope line items keyed to `surface_id`.

## Benchmark and reports
`bench/run_bench.py` regenerates every number from `data/raw/` + `data/ground_truth/*.csv`.
Gates: opening width, ceiling height, repeatability, drift ablation, photo stitch, wall
length ±8%/±3%, interval coverage. Head-to-head: LiDAR tier vs Polycam (free) on 2 rooms.
Fix loop: `fix/declaration.md`, before/after runs tagged in git (`fix-before`, `fix-after`).

## Deliverables map
Compliance matrix `docs/compliance.md`, protocol `docs/capture_protocol.md`, device matrix
`docs/device_matrix.md`, README (fresh machine < 15 min), technical report ≤ 6 pages,
raw data in `data/` (large files via `scripts/fetch_data.sh`).

## Schedule
- **Oct 2 eve:** repo, schema, Stray Scanner ingest, LiDAR tier on `single_room`;
  protocol page; email for ground truth; try to borrow iPhone Pro + laser.
- **Oct 3 AM:** stitching + drift ablation; video and photo tiers.
- **Oct 3 PM:** damage + rules; benchmark harness; first full run; head-to-head.
- **Oct 3 night:** fix declaration → ship fix → after run.
- **Oct 4 AM:** reports, compliance matrix, clean-machine README test. Submit by 10:00.

## Known risks (accepted)
Photo-tier stitching is the weakest path; mirrors/glass create phantom walls (mitigation:
depth-confidence filtering + reflective-surface flag); 8 GB VRAM limits MASt3R frame count.
No phone yet: damage room, Polycam head-to-head and true GT depend on obtaining one by
Oct 3 noon; otherwise reported as gaps, not faked.
