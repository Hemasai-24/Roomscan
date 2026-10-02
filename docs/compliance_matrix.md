# Compliance matrix

Status: **done** · **partial** (works, gate not met or not fully covered) · **planned** · **gap** (not possible under our constraints — see `docs/TRADEOFFS.md`).
Draft as of 2026-10-03 early morning; updated as work lands.

## Part 1 — capture and tiers
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock app + one-page protocol) | `docs/capture_protocol.md` | protocol page | done (not tested on an iPhone by us) |
| Photo tier: 2-8 stills/room, per-room folders → stitched plan | `roomscan/` photo modules (branch plan4-photo-tier), `run.py` | plan.json + plan.png | planned / in progress |
| Video tier: handheld clip | `roomscan/video_*.py`, `run.py` | plan.json + plan.png | partial (runs; accuracy gate not met) |
| LiDAR tier: depth, poses, intrinsics | `roomscan/load_capture.py` … `pipeline.py`, `run.py` | plan.json + plan.png | done |
| Same output contract from each tier, intervals widen as data thins | `schema/plan.schema.json`, `roomscan/measurements.py` (TIER tables) | schema-valid JSON | done (LiDAR, video); photo planned |
| Device matrix | `docs/device_matrix.md` | table | partial (accuracy cells await benchmark) |

## Part 2 — output contract and gates
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| Per-room plan: walls, ceiling height, floor area, openings | `roomscan/room_outline.py`, `room_surfaces.py`, `find_doors_windows.py`, `measurements.py` | `rooms[]` in plan.json | done (outline quality partial) |
| Stitched multi-room plan, correct adjacency | `roomscan/split_rooms.py`, `connect_rooms.py` | `adjacency[]`, plan.png | partial (LiDAR/video); photo planned |
| Per-surface damage regions, class + metric extent | — | `damage[]` | planned |
| Concealed-damage flags with the rule that fired | — | `concealed_damage_flags[]` | planned |
| Scope line items keyed to surfaces | — | `scope_items[]` (surface_id ready on walls) | planned |
| Confidence interval on every measurement | `roomscan/measurements.py` | `{value, lo, hi, unit}` everywhere | done (calibration planned) |
| One command per capture | `run.py` | CLI | done |
| JSON to the published schema | `schema/plan.schema.json`, `roomscan/save_plan.py` | validated on every run | done |
| Rendered plan | `roomscan/draw_plan.py` | plan.svg / plan.png | done |
| Benchmark set (multi-room+connector, damage room, all 3 tiers, repeat capture, tape GT) | `docs/benchmark_capture_s23.md`, `data/ground_truth/own_rooms.csv` | raw data + GT | planned (S23, Oct 3 morning); no iPhone → gap for LiDAR GT |
| Gate: opening widths ≤ 2 cm on ≥ 85% | `scripts/` benchmark (planned) | benchmark report | planned |
| Gate: ceiling height ≤ 1.5 cm; repeat spread ≤ 1 cm | — | benchmark report | planned |
| Gate: repeatability ≤ 1 cm / 0.5% per wall | `scripts/repeatability.py`, `roomscan/compare_plans.py` | repeatability table | partial (runs; 0/14 walls pass on sample) |
| Gate: drift accountability + on/off ablation | `roomscan/drift_correction.py`, `scripts/drift_ablation.py` | ablation table + plans | partial (method + ablation exist; no measurable gain on real sample) |
| Gate: photo-tier whole-property stitch, ±8% footprint | photo tier (planned) | eval table | planned |
| Photo ±8% / video ±3% wall lengths, calibrated | `scripts/eval_video_vs_lidar.py` | eval table | partial (video fails gate on sample) |

## Part 3 — head-to-head
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| LiDAR tier vs a consumer app on 2 rooms | — | comparison table | gap (no LiDAR iPhone; see TRADEOFFS) |

## Part 4 — fix loop
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| Fix declaration (worst gate, root cause, fix, prediction) | `docs/fix_loop.md` | declaration | planned (after benchmark) |
| Before/after runs regenerable + readable diff | git tags `fix-before`, `fix-after` | runs + diff | planned |

## Part 5 / deliverables
| Requirement | File / path | Status |
|---|---|---|
| Commit history as you work | git log | done (ongoing) |
| README: fresh capture running < 15 min, one command | `README.md` | partial (verified for LiDAR tier on a fresh clone) |
| Reproduction bundle (regenerate every number) | `scripts/`, `scripts/fetch_weights.py` | partial |
| Benchmark report | `docs/benchmark_report.md` | planned |
| Technical report ≤ 6 pages | `docs/technical_report.md` | planned |
| Raw benchmark data | `data/` + fetch script | planned |
| Mirrors, glass, wet-look, low light covered | `docs/TRADEOFFS.md`, S23 low-light clip | partial |
| Weights fetched by script, no own infrastructure | `scripts/fetch_weights.py` | done |
| Trade-offs listed (hiring team request) | `docs/TRADEOFFS.md` | done (ongoing) |
