# Floor-Plan Pipeline — Design Spec

Date: 2026-10-02 · Deadline: 2026-10-04 10:00 (~42 h) · Source brief: `Applied_AI.pdf`

## Goal
One command per capture (`python run.py <capture_dir>`) that turns a handheld iPhone
capture (photos, video, or LiDAR) into a JSON plan matching our published schema plus a
rendered whole-property floor plan, with a confidence interval on every measurement.
Must run cold on an unseen space at the defense.

## Decisions (and why)
- **Capture route: Route 2 (stock app + one-page protocol).** No Mac/TestFlight risk;
  pipeline runs on Linux. Tools: *3D Scanner App* (Laan Labs, free, raw depth+poses+
  intrinsics export) for LiDAR; native Camera for photos and video.
- **Language/stack:** Python 3.10, Open3D, NumPy/SciPy, OpenCV, PyTorch (RTX 2000 Ada, 8 GB).
  Pretrained models (disclosed): Depth Anything V2 Metric (scale), MASt3R or COLMAP
  (photo/video poses), OWLv2 (damage detection). Weights fetched by `scripts/fetch_weights.sh`.
- **Dev data before our phone arrives:** ARKitScenes sample scenes (disclosed). Benchmark
  numbers come only from our own captures.
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
- **Oct 2 eve:** repo, schema, LiDAR tier on ARKitScenes; protocol page; **borrow iPhone
  Pro + laser; capture benchmark tonight**.
- **Oct 3 AM:** stitching + drift ablation; video and photo tiers.
- **Oct 3 PM:** damage + rules; benchmark harness; first full run; head-to-head.
- **Oct 3 night:** fix declaration → ship fix → after run.
- **Oct 4 AM:** reports, compliance matrix, clean-machine README test. Submit by 10:00.

## Known risks (accepted)
Photo-tier stitching is the weakest path; mirrors/glass create phantom walls (mitigation:
depth-confidence filtering + reflective-surface flag); 8 GB VRAM limits MASt3R frame count.
No phone yet — if no capture by Oct 3 noon, the benchmark is not possible and we report that.
