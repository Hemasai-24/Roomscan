# SDD ledger — plan: experiment directive (tier accuracy root cause), branch exp-tier-accuracy
Setup: Ruling: venv, sample_data, weights, third_party symlinked from main checkout — sandbox isolation — cost if wrong: none
A: Ruling: oracle "all" mode added (LiDAR depth + poses of the same frames) beyond the directive's scale/pose/both — isolates the room-shape method floor — cost if wrong: none
A: Ruling: video "pose_framescale" mode added — pose_scale gave 59.5% wall error vs 5.8% for "all", pointing at per-chunk VGGT depth scale drift; this tests it — cost if wrong: none
B1: Ruling: Depth Pro with true focal instead of installing Metric3D v2 / UniDepth v2 — Depth Pro is focal-aware, already downloaded; new packages would modify the shared venv (torch pins) used by the main checkout — cost if wrong: Metric3D/UniDepth not compared
B1: Ruling: Depth Pro runs on CPU — GPU is shared with the VGGT queue (an earlier GPU co-run caused a CUDA OOM) — cost if wrong: slower experiment only
Eval: Ruling: eval_video_vs_lidar footprint IoU now works for multi-room plans (was None) — needed to compare modes — cost if wrong: none
B1: Ruling: Depth Pro moved from CPU (3 frames/10 min) to GPU after the VGGT queue, 6 frames/capture — time — cost if wrong: smaller frame sample
A: Ruling: oracle "pose"/"pose_scale" modes use ALL pose segments with one scale, but merge_chunks starts each segment at scale 1.0, so those rows mix segment scales; "pose_framescale" (per-frame true scale) is the clean pose-only row — cost if wrong: none, documented
Found: photo_links counts uninitialised RANSAC masks (F=None) as matches (882-1529 "matches" from 13 real); opt-in robust_masks fix — Ruling: not fixed in default path (directive: fix ships after the declaration) — cost if wrong: none
B4: Ruling: wall-plane extent ("walls") and single-piece extent ("core") measured and rejected (worse) — cost if wrong: none
B2: Ruling: camera-height prior rejected as scale source (12.6 % vs 6.1 % for Depth Anything, better in 1/9 rooms) — cost if wrong: none
Photo stitch: Ruling: added shared-doorway-photo protocol simulation + registration + clip-on-overlap (all opt-in) — oracle evidence says placement, not room shape, dominates the whole-property error — cost if wrong: none (default unchanged)
