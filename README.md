# Roomscan

Turn a phone capture into a dimensioned floor plan (JSON + SVG/PNG) with a 95%
interval on every measurement.

Read [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) first: it lists the constraints this was
built under and what they do to the reported numbers.

## Quickstart (clean Linux machine, Python 3.10+)
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt                 # LiDAR tier (CPU only)
    .venv/bin/pip install -r requirements-models.txt          # video + photo tiers (NVIDIA GPU, >= 8 GB)
    .venv/bin/python scripts/fetch_weights.py                 # model weights (~7.8 GB, once)

One command per capture; the tier is detected from what you pass:

    .venv/bin/python run.py <stray_scanner_folder>            # LiDAR tier
    .venv/bin/python run.py <walkthrough.mp4>                 # video tier
    .venv/bin/python run.py <folder_of_room_folders>          # photo tier (one sub-folder of photos per room)

Output: `outputs/<name>/plan.json` (schema: `schema/plan.schema.json`), `plan.svg`, `plan.png`.
How to capture: [`docs/capture_protocol.md`](docs/capture_protocol.md) (one page). Options: `--tier`,
`--drift-correction on|off`, `--rotate` (video recorded sideways without rotation metadata).

## Code map (one file per pipeline step)
| Step | File | What it does |
|---|---|---|
| 1 | `roomscan/load_capture.py` | Read a Stray Scanner export: depth images + phone position per frame |
| 2 | `roomscan/point_cloud.py` | Depth pixels -> 3D points in room coordinates |
| 3 | `roomscan/find_surfaces.py` | Flat surfaces (planes) labelled floor / ceiling / wall |
| 4 | `roomscan/room_outline.py` | Room outline from above, snapped to the walls; wall lengths |
| 5 | `roomscan/find_doors_windows.py` | Holes that depth rays pass through: door (reaches floor) or window |
| 6 | `roomscan/measurements.py` | Every number as value + 95% range |
| 7 | `roomscan/save_plan.py`, `roomscan/draw_plan.py` | JSON output (schema in `schema/`) and plan drawing |
| 4a | `roomscan/split_rooms.py` | Cut the floor into rooms at doorways |
| 4b | `roomscan/room_surfaces.py`, `roomscan/connect_rooms.py` | Each room's own floor/ceiling/walls; which rooms connect |
| - | `roomscan/drift_correction.py` | Re-align a long walk chunk by chunk (on/off: `scripts/drift_ablation.py`) |
| video | `roomscan/video_frames.py`, `video_poses.py`, `video_scale.py`, `video_capture.py`, `video_pipeline.py` | Sharp frames -> VGGT poses+depth -> metric scale -> Stray-like capture -> same back end |
| photo | `roomscan/photo_folders.py`, `photo_room.py`, `photo_links.py`, `stitch_rooms.py`, `photo_pipeline.py` | One room per folder -> join rooms at shared doors -> one plan |
| damage | `roomscan/damage_measure.py`, `damage_rules.py`, `repair_scope.py` | Damage size on its surface; hidden-damage rules R1-R5; repair line items |
| bench | `roomscan/score_benchmark.py`, `compare_plans.py`, `scripts/` | Score vs tape ground truth; repeatability; tier-vs-LiDAR evals |
| - | `roomscan/pipeline.py`, `run.py` | Runs the steps; the one command |

Docs: `docs/design.md` (design), `docs/plans/` (build plans), `docs/TRADEOFFS.md` (limitations),
`docs/INTERVIEW_NOTES.md` (how and why, step by step), `docs/device_matrix.md`.

## Tests
    .venv/bin/python -m pytest    # synthetic tests; sample-data tests run when sample_data/ exists

## Status
See [`docs/compliance_matrix.md`](docs/compliance_matrix.md) for every requirement and its status, and
[`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) for measured limitations.
