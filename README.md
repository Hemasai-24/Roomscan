# Roomscan

Turn a phone capture into a dimensioned floor plan (JSON + SVG/PNG) with a 95%
interval on every measurement.

Read [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) first: it lists the constraints this was
built under and what they do to the reported numbers.

## Quickstart (clean Linux machine, Python 3.10+)
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python run.py <capture_dir>          # -> outputs/<name>/plan.json, plan.svg, plan.png

Capture with the free **Stray Scanner** iOS app (see `docs/capture_protocol.md`).

## Tests
    .venv/bin/python -m pytest    # synthetic tests; sample-data tests run when sample_data/ exists

## Status
LiDAR tier, single room: implemented. Multi-room stitch, video, photo, damage: in progress.
