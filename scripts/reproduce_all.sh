#!/usr/bin/env bash
# Regenerate every reported number from raw inputs (brief deliverable 4: reproduction bundle).
#
#   bash scripts/reproduce_all.sh            # everything (needs GPU for video/photo/damage)
#   bash scripts/reproduce_all.sh lidar      # only the CPU parts (LiDAR tier, drift, repeatability)
#
# Inputs:  sample_data/  (the provided Stray Scanner captures)
#          data/raw/s23/ (our own captures; fetch with: python scripts/fetch_data.py)
#          data/ground_truth/own_rooms.csv, data/ground_truth/room_map.json (tape measurements, room id mapping)
# Outputs: outputs/repro/...  and outputs/repro/timing.tsv (wall-clock seconds per step)
# Everything runs live (no cached model outputs); runs are deterministic on the same hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=outputs/repro
MODE=${1:-all}
mkdir -p "$OUT"
: > "$OUT/timing.tsv"

SINGLE=sample_data/single_room/c00a170fe1
FLOOR=sample_data/single_scan_floor_only/1a8384c3f6
CEIL=sample_data/single_scan_with_ceiling/c7d28f72c6

step() {   # step <name> <command...>
  local name=$1; shift
  echo "== $name"
  local t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1 || { echo "   FAILED (see $OUT/$name.log)"; return 1; }
  local t1=$(date +%s)
  printf "%s\t%s\n" "$name" "$((t1 - t0))" >> "$OUT/timing.tsv"
  tail -n 3 "$OUT/$name.log" | sed 's/^/   /'
}

# --- LiDAR tier on the sample captures -------------------------------------------------------
for c in "$SINGLE" "$FLOOR" "$CEIL"; do
  n=$(basename "$(dirname "$c")")
  step "lidar_$n" $PY run.py "$c" --out "$OUT/lidar/$n" --no-damage
done
# Drift accountability: correction off vs on
for c in "$SINGLE" "$FLOOR" "$CEIL"; do
  n=$(basename "$(dirname "$c")")
  step "drift_ablation_$n" $PY scripts/drift_ablation.py "$c" "$OUT/drift_ablation/$n"
done
# Repeatability: the two whole-floor scans of the same flat
step repeatability $PY scripts/repeatability.py "$OUT/lidar/single_scan_floor_only/plan.json" \
  "$OUT/lidar/single_scan_with_ceiling/plan.json" "$OUT/repeatability.json"

[ "$MODE" = "lidar" ] && { echo "done (lidar only): $OUT"; exit 0; }

# --- Video and photo tiers vs LiDAR on the sample (LiDAR is the reference) --------------------
for c in "$SINGLE" "$FLOOR"; do
  n=$(basename "$(dirname "$c")")
  step "video_vs_lidar_$n" $PY scripts/eval_video_vs_lidar.py "$c" --rotate 90
done
for c in "$FLOOR" "$CEIL"; do
  n=$(basename "$(dirname "$c")")
  step "make_photos_$n" $PY scripts/make_photo_folders.py "$c" "$OUT/photo_sim/$n"
  step "photo_vs_lidar_$n" $PY scripts/eval_photo_vs_lidar.py "$OUT/photo_sim/$n"
done
# --- Damage: false alarms on undamaged captures + painted damage with known truth ------------
for c in "$SINGLE" "$FLOOR" "$CEIL"; do
  n=$(basename "$(dirname "$c")")
  step "damage_$n" $PY run.py "$c" --out "$OUT/damage/$n"
done
step damage_synthetic_e2e $PY scripts/synthetic_damage_e2e.py "$CEIL" "$OUT/damage_synthetic"

# --- Our own tape-measured benchmark (Samsung S23) ---------------------------------------------
S23=data/raw/s23
if [ -d "$S23" ]; then
  [ -d "$S23/photos" ] && step s23_photo $PY run.py "$S23/photos" --out "$OUT/s23/photo"
  for v in "$S23"/video/*.mp4 "$S23"/video/*.MP4 "$S23"/video/*.mov "$S23"/video/*.MOV; do
    [ -f "$v" ] || continue
    b=$(basename "${v%.*}")
    step "s23_video_$b" $PY run.py "$v" --out "$OUT/s23/video_$b"
  done
  if [ -f data/ground_truth/room_map.json ]; then
    step s23_score $PY scripts/score_s23.py "$OUT/s23" data/ground_truth/own_rooms.csv data/ground_truth/room_map.json
  fi
else
  echo "== s23: $S23 not found (python scripts/fetch_data.py) - skipped"
fi
echo "done: $OUT  (timing: $OUT/timing.tsv)"
