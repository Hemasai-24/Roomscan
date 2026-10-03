#!/usr/bin/env bash
# Build data/raw/m53_with_doors: the protocol photo set (data/raw/m53_doorway_shared) plus one straight-on
# photo of each room's door, in <room>/doors/ (data/raw/m53_doors, taken later the same day), as docs/capture_protocol.md asks.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=data/raw/m53_doorway_shared; DST=data/raw/m53_with_doors
DOORS=data/raw/m53_doors                       # local copy; scripts/fetch_data.py puts it in data/raw/m53/m53_doors
[ -d "$DOORS" ] || DOORS=data/raw/m53/m53_doors
[ -d "$SRC" ] || bash scripts/make_doorway_shared.sh
rm -rf "$DST"; cp -r "$SRC" "$DST"
for d in "$DOORS"/*/; do mkdir -p "$DST/$(basename "$d")/doors"; cp "$d"*.jpg "$DST/$(basename "$d")/doors/"; done
echo "built $DST"
