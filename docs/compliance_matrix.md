# Compliance matrix

Status: **done** · **partial** (works, gate not met or not fully covered) · **planned** · **gap** (not possible under our constraints — see `docs/TRADEOFFS.md`).
As of 2026-10-03 16:00; updated as work lands.

## Part 1 — capture and tiers
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock app + one-page protocol) | `docs/capture_protocol.md` | protocol page | done (not tested on an iPhone by us) |
| Photo tier: 2-8 stills/room, per-room folders → stitched plan | `roomscan/photo_*.py`, `stitch_rooms.py`, `run.py` | plan.json + plan.png | partial (own home: footprint +7.5 %, walls 2/16 within ±8 %; sample-sim after fix: median wall error 38.7 %) |
| Video tier: handheld clip | `roomscan/video_*.py`, `run.py` | plan.json + plan.png | partial (runs; accuracy gate not met) |
| LiDAR tier: depth, poses, intrinsics | `roomscan/load_capture.py` … `pipeline.py`, `run.py` | plan.json + plan.png | done |
| Same output contract from each tier, intervals widen as data thins | `schema/plan.schema.json`, `roomscan/measurements.py` (TIER tables) | schema-valid JSON | done (all three tiers) |
| Device matrix | `docs/device_matrix.md` | table | done (photo/video accuracy measured on our tape-measured home; LiDAR: no tape GT) |

## Part 2 — output contract and gates
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| Per-room plan: walls, ceiling height, floor area, openings | `roomscan/room_outline.py`, `room_surfaces.py`, `find_doors_windows.py`, `measurements.py` | `rooms[]` in plan.json | done (outline quality partial) |
| Stitched multi-room plan, correct adjacency | `roomscan/split_rooms.py`, `connect_rooms.py`, `stitch_rooms.py` | `adjacency[]`, plan.png | partial (all tiers; photo adjacency on own home 1/3; sample-sim precision 0.67 / recall 0.44) |
| Per-surface damage regions, class + metric extent | `roomscan/damage_detect.py`, `damage_measure.py`, `damage_pipeline.py` | `damage[]` | partial (2-view rule: 0 false alarms on our photos, but the single-photo staged stain is dropped; detector finds 5/10 public damage photos) |
| Concealed-damage flags with the rule that fired | `roomscan/damage_rules.py` (R1-R5) | `concealed_damage_flags[]` | done (wired in all tiers; fires on false alarms too) |
| Scope line items keyed to surfaces | `roomscan/repair_scope.py` | `scope_items[]` | done |
| Confidence interval on every measurement | `roomscan/measurements.py` | `{value, lo, hi, unit}` everywhere | done (tape truth inside 4/4 areas, 16/16 walls; photo/video ranges wide) |
| One command per capture | `run.py` | CLI | done |
| JSON to the published schema | `schema/plan.schema.json`, `roomscan/save_plan.py` | validated on every run | done |
| Rendered plan | `roomscan/draw_plan.py` | plan.svg / plan.png | done |
| Benchmark set (multi-room+connector, damage room, all 3 tiers, repeat capture, tape GT) | `docs/benchmark_capture.md`, `data/ground_truth/`, `data/DATA_URL` | raw data + GT | partial (own home: 3 rooms + hall, staged damage, photo + video, bedroom twice, tape; no LiDAR tier — no LiDAR phone) |
| Gate: opening widths ≤ 2 cm on ≥ 85% | `scripts/score_m53.py`, `roomscan/score_benchmark.py` | `docs/benchmark_report.md` | fail (photo: 1 of 9; 6 phantoms) |
| Gate: ceiling height ≤ 1.5 cm; repeat spread ≤ 1 cm | `roomscan/measurements.py` | `docs/benchmark_report.md` | fail (ceilings not seen in photos/video; honest flagged ranges) |
| Gate: repeatability ≤ 1 cm / 0.5% per wall | `scripts/repeatability.py`, `roomscan/compare_plans.py`, `scripts/score_m53.py` | `docs/benchmark_report.md` | fail (LiDAR sample 0/14; video bedroom ×2: 0/3) |
| Gate: drift accountability + on/off ablation | `roomscan/drift_correction.py`, `scripts/drift_ablation.py` | ablation table + plans | partial (method + ablation exist; no measurable gain on real sample) |
| Gate: photo-tier whole-property stitch, ±8% footprint | `roomscan/photo_*.py`, `stitch_rooms.py`, `scripts/score_m53.py` | `docs/benchmark_report.md` | partial (own home: footprint +7.5 % passes; adjacency 1/3, 2 rooms unplaced) |
| Photo ±8% / video ±3% wall lengths, calibrated | `scripts/score_m53.py` | `docs/benchmark_report.md` | fail (photo 2/16 walls; video −31 % … +92 %); truth inside ranges, ranges wide |

## Part 3 — head-to-head
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| LiDAR tier vs a consumer app on 2 rooms | — | comparison table | gap (no LiDAR iPhone; see TRADEOFFS) |

## Part 4 — fix loop
| Requirement | File / path | Artifact | Status |
|---|---|---|---|
| Fix declaration (worst gate, root cause, fix, prediction) | `docs/fix_loop.md` | declaration | done |
| Before/after runs regenerable + readable diff | tags `fix-before` / `fix-after`, `docs/fix_loop.diff`, `scripts/score_m53.py` | before/after table | done (footprint +30 % → +7.5 %) |

## Part 5 / deliverables
| Requirement | File / path | Status |
|---|---|---|
| Commit history as you work | git log | done (ongoing) |
| README: fresh capture running < 15 min, one command | `README.md` | partial (verified for LiDAR tier on a fresh clone) |
| Reproduction bundle (regenerate every number) | `scripts/reproduce_all.sh`, `fetch_weights.py`, `fetch_data.py` | done (CPU part verified end to end) |
| Benchmark report | `docs/benchmark_report.md` | done |
| Technical report ≤ 6 pages | `docs/technical_report.md` | done (~4 pages) |
| Raw benchmark data | `data/DATA_URL` (Google Drive), `scripts/fetch_data.py`, `data/ground_truth/` | done |
| Mirrors, glass, wet-look, low light covered | `docs/TRADEOFFS.md`, Galaxy M53 low-light clip | partial |
| Weights fetched by script, no own infrastructure | `scripts/fetch_weights.py` | done |
| Trade-offs listed (hiring team request) | `docs/TRADEOFFS.md` | done (ongoing) |
