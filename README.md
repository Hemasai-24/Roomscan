# Roomscan

Turn a phone capture into a dimensioned floor plan (JSON + SVG/PNG) with a 95%
interval on every measurement.

Read [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) first: it lists the constraints this was
built under and what they do to the reported numbers.

## Quickstart (clean Linux machine, Python 3.10+)
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python run.py <capture_dir>          # -> outputs/<name>/plan.json, plan.svg, plan.png

Capture with the free **Stray Scanner** iOS app (see `docs/capture_protocol.md`).

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
| - | `roomscan/pipeline.py`, `run.py` | Runs steps 1-7; the one command |

Docs: `docs/design.md` (design), `docs/plans/` (build plans), `docs/TRADEOFFS.md` (limitations).

## Tests
    .venv/bin/python -m pytest    # synthetic tests; sample-data tests run when sample_data/ exists

## Status
LiDAR tier, single room: implemented. Multi-room stitch, video, photo, damage: in progress.
