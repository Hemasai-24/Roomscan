#!/usr/bin/env bash
# Regenerate every reported number from raw inputs (brief deliverable 4: reproduction bundle).
#
#   bash scripts/reproduce_all.sh            # everything (needs GPU for video/photo/damage)
#   bash scripts/reproduce_all.sh lidar      # only the CPU parts (LiDAR tier, drift, repeatability)
#
# Inputs:  sample_data/  (the provided Stray Scanner captures)
#          data/raw/m53/ (our own captures; fetch with: python scripts/fetch_data.py)
#          data/ground_truth/own_rooms.csv (tape measurements)
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

# --- Our own tape-measured benchmark (Samsung Galaxy M53) --------------------------------------
M53=data/raw/m53            # room folders of photos + video/  (python scripts/fetch_data.py)
if [ -d "$M53" ]; then
  step m53_photo $PY run.py "$M53" --tier photo --out outputs/fix_loop/after_photo_v2
  for v in "$M53"/video/*.mp4; do
    [ -f "$v" ] || continue
    b=$(basename "${v%.*}")
    step "m53_video_$b" $PY run.py "$v" --out "outputs/m53/video_$b"
  done
  step m53_doorway_shared bash scripts/make_doorway_shared.sh
  step m53_photo_protocol $PY run.py data/raw/m53_doorway_shared --tier photo --out outputs/m53/photo_protocol
  step m53_with_doors bash scripts/make_with_doors.sh
  step m53_photo_doors $PY run.py data/raw/m53_with_doors --tier photo --out outputs/m53/photo_doors
  step m53_score $PY scripts/score_m53.py outputs/fix_loop/after_photo_v2/plan.json
else
  echo "== m53: $M53 not found (python scripts/fetch_data.py) - skipped"
fi
echo "done: $OUT  (timing: $OUT/timing.tsv)"
