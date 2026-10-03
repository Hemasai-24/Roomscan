#!/usr/bin/env bash
# Build data/raw/m53_doorway_shared: the same 24 M53 photos, with the 3 doorway shots (each taken from a room
# looking into the hall) also copied into the hall folder - what docs/capture_protocol.md now asks for.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=data/raw/m53; DST=data/raw/m53_doorway_shared
rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC"/Room1_bedroom "$SRC"/Room2_hall "$SRC"/Room3_kitchen "$SRC"/Room4_bathroom "$DST"/
for f in Room1_bedroom/20261003_123632.jpg Room3_kitchen/20261003_125545.jpg Room4_bathroom/20261003_125714.jpg; do
  cp "$SRC/$f" "$DST/Room2_hall/shared_$(basename "$f")"
done
echo "built $DST"
