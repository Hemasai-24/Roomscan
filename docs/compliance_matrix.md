# Compliance matrix

Each row lists one requirement of the brief, where it is in the repository, the evidence for it, and its
status. The status is **Met**, **Met with a gap** (the row states the gap) or **Not met** (the row states the
reason). The numbers come from `docs/benchmark_report.md`.

**Summary.** 30 requirements are met, 6 are met with a stated gap, and 5 are not met. The five not met are
four accuracy gates (opening widths, ceiling height, repeatability, and photo and video wall lengths) and the
head-to-head against a consumer app, which needed a LiDAR iPhone that was not available.

## Part 1: Capture route and input tiers
| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 1.1 | Capture route with a one-page protocol a non-engineer follows | `docs/capture_protocol.md` | Route 2: Stray Scanner (LiDAR) + Camera app (video, photos); install, walk, avoid, hand-off on one page | Met |
| 1.2 | Photo tier: 2-8 photos per room, one folder per room → one stitched plan | `run.py`, `roomscan/photo_*.py`, `stitch_rooms.py` | Own home, 24 photos in 4 folders → one plan, 4 rooms placed | Met |
| 1.3 | Video tier: handheld walkthrough | `run.py`, `roomscan/video_*.py` | 68 s walkthrough → plan; 4 clips run | Met |
| 1.4 | LiDAR tier: depth, poses, intrinsics | `run.py`, `roomscan/load_capture.py` … `pipeline.py` | Sample captures → 2 / 13 / 9 rooms | Met |
| 1.5 | Same output contract from every tier; intervals widen as data thins | `schema/plan.schema.json`, `roomscan/measurements.py` | One schema for all tiers; σ floor LiDAR 0.8 cm, video +25 % relative, photo +60 % relative | Met |
| 1.6 | Device matrix: tier → hardware → honest accuracy | `docs/device_matrix.md` | Photo/video accuracy measured against tape; LiDAR accuracy has no tape value (no LiDAR phone) | Met with a gap |

## Part 2: Output contract
| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 2.1 | Per-room plan: walls, ceiling height, floor area, openings | `roomscan/room_outline.py`, `room_surfaces.py`, `find_doors_windows.py`, `visual_openings.py` | Every room record has walls, area, ceiling, openings | Met |
| 2.2 | Stitched multi-room plan with correct adjacency | `roomscan/split_rooms.py`, `connect_rooms.py`, `stitch_rooms.py` | Photo tier, own home: all 3 true connections, 0 overlaps (doorway photo in both folders, as the protocol says); LiDAR sample: 9 connections | Met |
| 2.3 | Per-surface damage regions with class and metric extent | `roomscan/damage_detect.py`, `damage_measure.py`, `damage_pipeline.py` | Class, surface id, area/width/height in metres; detector finds damage in 5/10 public photos; staged stain dropped by the 2-view rule (seen in one photo) | Met with a gap |
| 2.4 | Concealed-damage flags with the rule that fired | `roomscan/damage_rules.py` | Rules R1-R5; every flag names its rule, damage ids and surfaces | Met |
| 2.5 | Scope line items keyed to surfaces | `roomscan/repair_scope.py` | Each item has a surface id, action, quantity with range | Met |
| 2.6 | A confidence interval on every measurement | `roomscan/measurements.py` | Every number is `{value, lo, hi, unit}`; tape truth inside 4/4 room areas and 16/16 walls | Met |
| 2.7 | One command per capture | `run.py` | `python run.py <input>`, a Stray folder, a video file, or a folder of room folders | Met |
| 2.8 | JSON to the published schema | `schema/plan.schema.json`, `roomscan/save_plan.py` | Validated on every run | Met |
| 2.9 | Rendered plan | `roomscan/draw_plan.py` | `plan.png` / `plan.svg` with dimensions, doors, windows, damage | Met |

## Part 2: Benchmark set
| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 2.10 | One multi-room capture: 3+ rooms plus a connector | `data/raw/m53` (Drive) | Bedroom, kitchen, bathroom + hall | Met |
| 2.11 | One furnished room with staged damage of two classes | same | Bedroom: water stain + crack (small: 9 × 6 cm, 8 cm) | Met |
| 2.12 | The same rooms captured at all three tiers | same | Photo and video; no LiDAR tier (no LiDAR phone) | Met with a gap |
| 2.13 | At least one room captured twice at the same tier | `data/raw/m53/video` | Bedroom video, two takes | Met |
| 2.14 | Tape / laser ground truth on everything; raw data submitted | `data/ground_truth/`, `data/DATA_URL` | Every wall, door, window, ceiling taped; photos and videos on Drive | Met |

## Part 2: Gates
| # | Gate | Where | Evidence | Status |
|---|---|---|---|---|
| 2.15 | Opening widths ≤ 2 cm on ≥ 85 % (miss and phantom count) | `roomscan/score_benchmark.py` | Photo tier: 0 of 9 within 2 cm (our photos lack full-frame door shots) | Not met |
| 2.16 | Ceiling height ≤ 1.5 cm; repeat spread ≤ 1 cm | `roomscan/measurements.py` | Ceilings not visible in our photos/videos → reported as wide flagged ranges (truth inside) | Not met |
| 2.17 | Repeatability: same room twice within 1 cm / 0.5 % per wall | `scripts/score_m53.py`, `scripts/repeatability.py` | Video bedroom ×2: 0 of 3 walls; LiDAR sample: 0 of 14 | Not met |
| 2.18 | Drift accountability: method stated + footprint ablation on/off | `roomscan/drift_correction.py`, `scripts/drift_ablation.py` | Chunk-wise re-alignment; ablation table in the technical report (fixes injected drift; no measurable gain on the real sample) | Met |
| 2.19 | Photo-tier whole-property stitch: correct adjacency, no overlaps, footprint ±8 % | `scripts/score_m53.py` | Own home: 3/3 connections, 0 overlap, footprint +6 % | Met |
| 2.20 | Photo wall lengths ±8 % / video ±3 %, calibrated | `scripts/score_m53.py` | Photo: room areas −21 % … +42 %, 2/16 walls; video −31 % … +92 % | Not met |
| 2.21 | Calibration at every tier (no confident garbage) | `docs/benchmark_report.md` | Tape truth inside the range for every room area and wall; ranges are wide | Met |

## Part 3: Head-to-head
| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 3.1 | LiDAR tier vs a consumer app on 2 rooms, beat or tie ≥ 70 % | n/a | No LiDAR iPhone available to run either side | Not met |

## Part 4: Fix loop
| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 4.1 | Declaration: worst gate + failing number, root cause + evidence, fix + predicted number | `docs/fix_loop.md` §1-3 | Committed before the fix (tag `fix-before`) | Met |
| 4.2 | Fix shipped; before and after regenerable; readable diff | tags `fix-before`, `fix-after`; `docs/fix_loop.diff`; `scripts/score_m53.py` | Photo footprint +30 % → +7.5 % | Met |
| 4.3 | Gate moves from fail to pass, or the report says why it fell short | `docs/fix_loop.md` §4 | Footprint now inside ±8 %; per-room wall lengths still fail, explained, with the first attempt's failure | Met with a gap |

## Part 5: Process and deliverables
| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| 5.1 | Commit history that shows the work | git log | ~120 commits over two days, with build plans in `docs/plans/` | Met |
| 5.2 | README: running on a fresh capture in < 15 min, one command per capture | `README.md` | Fresh clone: install 43 s, LiDAR run 21 s; video/photo also need a one-time 7.8 GB model download | Met with a gap |
| 5.3 | Reproduction bundle: regenerate every number | `scripts/reproduce_all.sh`, `fetch_weights.py`, `fetch_data.py`, `results/` | CPU part verified end to end; GPU part runs the same scripts | Met |
| 5.4 | Benchmark report: gates at all tiers, repeatability, head-to-head, timing | `docs/benchmark_report.md` | Gates for photo/video against tape, LiDAR on the sample; head-to-head marked not done | Met |
| 5.5 | Technical report ≤ 6 pages | `docs/technical_report.md` | ~4 pages | Met |
| 5.6 | Raw benchmark data: sensor logs, ground truth, app exports | `data/DATA_URL`, `data/ground_truth/` | Photos, videos, tape notes | Met |
| 5.7 | Mirrors, glass, wet-look surfaces and low light covered | `docs/TRADEOFFS.md`, bathroom low-light clip | Each named with its measured effect; not solved | Met with a gap |
| 5.8 | Weights fetched by script; nothing calls our own infrastructure | `scripts/fetch_weights.py` | All models run locally | Met |
| 5.9 | Pretrained models and datasets disclosed | `docs/technical_report.md` §8, `docs/TRADEOFFS.md` | Names, versions, licences | Met |
| 5.10 | Trade-offs and limitations listed (requested by the hiring team) | `docs/TRADEOFFS.md` | Assumptions, constraints, every experiment that did not work, with numbers | Met |
