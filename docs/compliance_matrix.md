# Compliance matrix

Each row is one requirement from the brief. Status is **Met**, **Gap** (met, with the gap stated) or
**Not met** (with the reason). Numbers come from `docs/benchmark_report.md`.

**Summary:** 30 met, 6 with a gap, 5 not met. Four of the five are accuracy gates (openings, ceiling height,
repeatability, wall lengths). The fifth is the comparison with a consumer app, which needed a LiDAR iPhone.

## Part 1: Capture and input tiers

| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 1.1 | One-page capture protocol | `docs/capture_protocol.md` | Route 2: Stray Scanner and the camera app | Met |
| 1.2 | Photo tier: 2-8 photos per room, one plan | `roomscan/photo_*.py`, `stitch_rooms.py` | our home: 24 photos, 4 rooms, one plan | Met |
| 1.3 | Video tier | `roomscan/video_*.py` | 4 clips of our home run end to end | Met |
| 1.4 | LiDAR tier | `roomscan/load_capture.py`, `pipeline.py` | all 3 sample captures | Met |
| 1.5 | Same output from every tier; wider ranges for weaker input | `schema/plan.schema.json`, `roomscan/measurements.py` | one schema; video and photo add a scale term | Met |
| 1.6 | Device matrix with measured accuracy | `docs/device_matrix.md` | photo and video measured against tape | Gap: no tape result for LiDAR |

## Part 2: Output

| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 2.1 | Per room: walls, ceiling, area, openings | `roomscan/room_outline.py`, `room_surfaces.py`, `visual_openings.py` | every room record has all four | Met |
| 2.2 | Stitched plan with correct adjacency | `roomscan/split_rooms.py`, `stitch_rooms.py` | our home: 3 of 3 door connections, rooms in the right order, no overlap | Met |
| 2.3 | Damage regions with class and size | `roomscan/damage_*.py` | class, surface and size in metres | Gap: the staged stain was in one photo only and is dropped |
| 2.4 | Concealed-damage flags naming the rule | `roomscan/damage_rules.py` | rules R1-R5 | Met |
| 2.5 | Repair items keyed to surfaces | `roomscan/repair_scope.py` | surface, action, quantity with range | Met |
| 2.6 | A 95 % range on every number | `roomscan/measurements.py` | truth inside 4 of 4 areas, 15 of 16 walls | Met |
| 2.7 | One command per capture | `run.py` | `python run.py <input>` | Met |
| 2.8 | JSON to a published schema | `schema/plan.schema.json` | checked on every run | Met |
| 2.9 | Rendered plan | `roomscan/draw_plan.py` | `plan.png` and `plan.svg` | Met |

## Part 2: Benchmark set

| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 2.10 | 3+ rooms and a connector | raw data on Drive (`data/DATA_URL`) | bedroom, kitchen, bathroom, hall | Met |
| 2.11 | Furnished room with two kinds of staged damage | same | bedroom: water stain and crack | Met |
| 2.12 | Same rooms at all three tiers | same | photo and video | Gap: no LiDAR phone |
| 2.13 | One room captured twice | same | bedroom video, two takes | Met |
| 2.14 | Measured ground truth; raw data shared | `data/ground_truth/` | every wall, door, window and ceiling taped | Met |

## Part 2: Gates

| # | Gate | Evidence | Status |
|---|---|---|---|
| 2.15 | Openings within 2 cm on 85 % | 0 of 9; doors from straight-on photos are 3-10 cm too wide, windows wrong | Not met |
| 2.16 | Ceiling height within 1.5 cm; say whether biased or unrepeatable | both: video 0.6-0.9 m low, photos 0.5-1.6 m high, bedroom takes differ by 25 cm; no capture saw the ceiling | Not met |
| 2.17 | Same room twice within 1 cm per wall | video 0 of 3 walls; LiDAR sample 0 of 14 | Not met |
| 2.18 | Drift method and on/off comparison | technical report section 5 | Met |
| 2.19 | Photo stitch: adjacency, no overlap, footprint ±8 % | 3 of 3 door connections, right order, no overlap, +6 % | Met |
| 2.20 | Wall lengths: photo ±8 %, video ±3 % | photo 2 of 16 walls; video rooms −31 % to +92 % | Not met |
| 2.21 | Ranges contain the truth | 4 of 4 areas, 15 of 16 walls; ranges are wide | Met |

## Part 3: Comparison with a consumer app

| # | Requirement | Evidence | Status |
|---|---|---|---|
| 3.1 | LiDAR tier vs Polycam on 2 rooms | no LiDAR iPhone | Not met |

## Part 4: Fix loop

| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 4.1 | Gate, cause and prediction written before the fix | `docs/fix_loop.md`, tag `fix-before` | committed before the code change | Met |
| 4.2 | Fix shipped; before and after reproducible | tags `fix-before`, `fix-after`; `docs/fix_loop.diff` | footprint +30 % → +7.5 % | Met |
| 4.3 | Gate passes, or the reason it does not | `docs/fix_loop.md` section 4 | footprint passes | Gap: wall lengths still fail |

## Part 5: Process and deliverables

| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 5.1 | Commit history | git log | about 120 commits over two days | Met |
| 5.2 | Fresh capture to plan in under 15 minutes | `README.md` | fresh clone: install 43 s, LiDAR run 21 s | Gap: video and photo need a one-time 7.8 GB model download |
| 5.3 | Script that regenerates every number | `scripts/reproduce_all.sh`, `results/` | CPU part checked end to end | Met |
| 5.4 | Benchmark report | `docs/benchmark_report.md` | all tiers, repeatability, timing | Met |
| 5.5 | Technical report, 6 pages or less | `docs/technical_report.md` | about 4 pages | Met |
| 5.6 | Raw data and ground truth | `data/DATA_URL`, `data/ground_truth/` | photos, videos, tape notes | Met |
| 5.7 | Mirrors, glass, shiny floors, low light | `docs/TRADEOFFS.md` | each described with its measured effect | Gap: not solved |
| 5.8 | Weights downloaded by script; no private servers | `scripts/fetch_weights.py` | all models run locally | Met |
| 5.9 | Models and datasets disclosed | technical report section 9 | names and licences | Met |
| 5.10 | Trade-offs and limitations listed | `docs/TRADEOFFS.md` | assumptions, gaps, failed experiments | Met |
