# Plan 3: Video Tier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python run.py walkthrough.mp4` (no depth, no poses) produces the same plan JSON and drawing
as the LiDAR tier, marked `tier: "video"`, with wider ranges.

**Approach in plain words:** pick ~2 sharp frames per second; a pretrained 3D model (VGGT) works out
where the camera was for each frame and how far away each pixel is — but only up to an unknown
size; a second model (Depth Anything V2, metric) gives distances in metres, which fixes the size;
"up" is taken from how people hold a phone (image-down ≈ gravity) and refined by the floor. The
result is written out **in exactly the Stray Scanner layout** (depth PNGs in mm + odometry.csv),
so the whole LiDAR back end — rooms, walls, doors, ranges — runs on it unchanged.

**Spec:** `docs/design.md` · **Tech:** as before + VGGT (third_party, script-cloned), transformers.

## Feasibility evidence (spike on sample `single_room`, 2026-10-03, RTX 2000 Ada 8 GB)

| Check | Result |
|---|---|
| Stray `rgb.mp4` | HEVC 1920x1440, no rotation metadata, frames stored sideways (gravity along image +x) → rotate 90° clockwise to make upright. Variable frame rate ~46 fps: ffmpeg must use `-vsync passthrough`, otherwise it pads to 60 fps and frame indices drift (2223 vs 1714 frames). |
| VGGT-1B, all heads, fp32 weights | Out of memory at 16 frames. |
| VGGT-1B, backbone bf16, camera+depth heads fp32, only the 4 token layers the heads use kept | **20 frames (392x518): 5.5 s, 5.5 GB peak.** |
| VGGT poses, 20 consecutive frames at 2 fps (10 s) | median rotation error **2.1°**, camera centre error **8.4 cm** (rotation-based similarity alignment). |
| VGGT poses, 16 frames spread over 37 s | fails (relative rotations wrong by up to 170°): frames too far apart. → process **consecutive chunks**. |
| VGGT on bathroom frames with **mirror + glass shower** | 7 consecutive frames flipped by 110–165°. → reject frames whose pose jumps > 60° from neighbours 0.5 s apart (a hand can't turn that fast). |
| VGGT focal length | 373 px vs true 432 px (−14%). Accepted; disclosed. |
| Depth Anything V2 Metric-Indoor-Large vs LiDAR | ratio **1.39 ± 0.19** per frame (consistent over-estimate), 0.27 s/frame. |
| Depth Pro vs LiDAR | ratio 1.33 ± 0.34; its focal estimate is +30% (560 vs 432 px) and its depth error follows it. Worse scatter, 4x slower → not chosen. |
| MASt3R | not tried: VGGT met the memory/runtime need; MASt3R install + global alignment would cost ~1 h. |

Decision: **VGGT for poses + relative depth, Depth Anything V2 metric for scale**, with the DA2
bias **calibrated on the LiDAR sample captures** (leave-one-capture-out error reported).

## Global Constraints
- Default test suite must not need the GPU or weights: model tests carry `@pytest.mark.model` and skip
  if `weights/` is missing.
- Output capture layout must load with `roomscan.load_capture.load_stray` unchanged.
- World +Y up in the written capture; poses camera→world, OpenCV camera axes.
- Same input → same output (no random sampling without a fixed seed; models in eval mode).
- Commits plain, no co-author trailers.

## Review Focus
1. Mirror/glass frames: flipped poses must be dropped, not fused (Task 3 test).
2. A chunk that cannot be aligned to the previous one (too few valid overlap frames) must start a new
   segment with a warning, not be glued on wrongly (Task 3 test).
3. Phone held in landscape: "up" estimate must still be right (Task 2 test with rolled cameras).
4. Very short video (< 2 s): clear error, not a crash (Task 5 test).
5. Scale calibration must not be evaluated on the capture it was fitted on (Task 4 leave-one-out).

---

### Task 1: Read the video and pick sharp frames — `roomscan/video_frames.py`

**In plain words:** decode the video frame by frame (exact frames, no padding), turn it upright if
asked, and in every half-second window keep the sharpest frame (most edge detail = least blur).

**Interfaces:** `video_info(path) -> (n_frames:int, fps:float)`;
`read_frames(path, size:(w,h), rotate:int=0) -> Iterator[(index, np.uint8 HxWx3 RGB)]`;
`pick_sharpest(path, per_sec=2.0, size=(392,518), rotate=0) -> (indices:list[int], images:np.ndarray[N,H,W,3])`;
`sharpness(img) -> float`.

- [ ] Tests `tests/test_video_frames.py`: write a 3-second 30 fps synthetic mp4 with cv2.VideoWriter
  (checkerboard; every frame except one per 15-frame window blurred) → `video_info` = (90, ~30);
  `pick_sharpest(per_sec=2)` returns exactly the 6 unblurred indices; `rotate=90` returns images whose
  height/width are swapped relative to `rotate=0` and equal `np.rot90(..., k=-1)` of the unrotated frame.
- [ ] Run → fail. Implement with ffmpeg (`-vsync passthrough`, `transpose=1` for 90° cw, `2` for 270°,
  `hflip,vflip` for 180°), Laplacian variance for sharpness. Run → pass. Commit.

### Task 2: Gravity and writing a Stray-layout capture — `roomscan/video_capture.py`

**In plain words:** people hold a phone roughly upright, so averaging every frame's "image-down"
direction gives gravity. Rotate the whole reconstruction so that is −Y. Then write depth PNGs (mm),
confidence PNGs, and odometry.csv exactly like Stray Scanner does, so the LiDAR loader reads it.

**Interfaces:** `estimate_up(T_wc:list[4x4]) -> np.ndarray[3]` (unit, world "up");
`level(T_wc, up) -> list[4x4]` (rotates world so `up` → +Y);
`refine_up(points, normals, up, max_deg=25) -> np.ndarray[3]` (mean of normals within 25° of up);
`write_capture(root, depths_m:list[HxW], confs:list[HxW uint8 0..2], T_wc:list[4x4], K:3x3) -> Path`
(odometry intrinsics written at a nominal 1920-px width so `load_stray`'s scaling is exact).

- [ ] Tests `tests/test_video_capture.py`: (a) cameras whose image-down is world −Y rolled by ±10° →
  `estimate_up` within 3° of +Y; (b) the same cameras with the whole world tilted 30° → `level` brings
  `estimate_up` back to +Y within 1°; (c) cameras rolled 90° (landscape) for all frames → `estimate_up`
  still returns +Y because image-down is computed per frame from the frame's own rotation (test name
  `test_landscape_held_phone`); (d) `write_capture` then `load_stray` round-trips poses (1e-6), K
  (1e-4 relative) and depth (≤ 1 mm).
- [ ] Run → fail; implement; pass; commit.

### Task 3: Camera poses from VGGT in overlapping chunks — `roomscan/video_poses.py`

**In plain words:** VGGT fits 20 frames at a time in 8 GB. Run it on chunks of 20 consecutive frames
that overlap by 5. Each chunk comes back in its own coordinates and size; glue chunk k onto chunk k−1
using the 5 shared frames: the turn from their camera rotations, the size from the ratio of their
depths, the shift from their camera positions. Before gluing, drop frames whose rotation jumps more
than 60° from both neighbours (mirror/glass confusion).

**Interfaces:** `rot_angle_deg(R) -> float`; `reject_flips(T_list, max_deg=60) -> np.ndarray[bool]` (valid);
`similarity_from_overlap(Ta:list, Tb:list, depth_ratio:float) -> 4x4` (maps chunk-b coords → chunk-a);
`merge_chunks(chunks:list[dict(idx, T, depth, conf, K)], overlap:int) -> dict(idx, T, depth, conf, K, segments)`;
`VGGTRunner(weights_dir).run(images:np.ndarray[N,H,W,3]) -> dict(T, depth, conf, K)` (model test only);
`run_chunks(runner, images, chunk=20, overlap=5) -> merged dict`.

- [ ] Tests `tests/test_video_poses.py` (pure, synthetic):
  `test_similarity_recovers_known_transform` (chunk b = known scale 2.5, 30° yaw, shift applied to chunk
  a's poses → recovered within 1e-6); `test_flip_rejected` (7 of 20 poses rotated 150° → those 7
  invalid, others valid); `test_unalignable_chunk_starts_new_segment` (overlap frames all invalid →
  `segments == 2` and no exception); `test_merge_three_chunks_consistent` (3 chunks of a known
  trajectory each in their own random similarity frame → merged poses match truth up to one global
  similarity within 1e-6).
- [ ] Model test `tests/test_video_poses_model.py` (`@pytest.mark.model`): 20 frames of sample
  single_room → median rotation error vs Stray odometry < 5°.
- [ ] Run → fail; implement; pass; commit.

### Task 4: Real-world scale — `roomscan/video_scale.py` + `scripts/calibrate_metric_depth.py`

**In plain words:** Depth Anything V2 says how far things are in metres, but on this phone it reads
~39% too far. We measure that bias on the LiDAR sample captures (we know the true depth there) and
divide it out. The capture's scale is the median, over all frames and confident pixels, of
(calibrated metric depth ÷ VGGT depth). We report the calibration error leave-one-capture-out: fit the
bias on two captures, test on the third.

**Interfaces:** `scale_from_depths(rel:list[HxW], metric:list[HxW], conf:list[HxW], bias:float) -> (scale, spread)`;
`load_bias() -> float` (from `roomscan/calibration.json`, default 1.0 with a warning);
`MetricDepth(weights_dir).predict(images) -> list[HxW metres]` (model test only).

- [ ] Tests `tests/test_video_scale.py`: rel depth = metric/3.0 with noise and bias 1.4 → scale 3.0/1.4
  within 1%; low-confidence pixels ignored (put garbage there); `spread` = robust std of per-frame ratios.
- [ ] Calibration script writes `roomscan/calibration.json`
  `{"depth_anything_v2_metric_indoor_large": {"bias": .., "per_capture": {...}, "leave_one_out_error": {...}}}`.
- [ ] Run → fail; implement; pass; run the script on the 3 sample captures; commit (json included).

### Task 5: Pipeline and command line

**In plain words:** `run.py` looks at what it was given: a Stray folder → LiDAR tier; a video file
(or a folder with only a video) → video tier. The video tier builds the Stray-layout capture under the
output folder and runs the same back end with `tier="video"` (every frame used, since frames are
already 2 per second).

**Interfaces:** `detect_tier(path) -> ("lidar"|"video", Path)`; `run_lidar(capture_dir, drift_correction=False,
tier="lidar", stride=5)`; `run_video(video, out_dir, rotate=0) -> plan`; CLI `run.py <path> [--tier auto|lidar|video] [--rotate 0|90|180|270]`.

- [ ] Tests `tests/test_tier_detection.py`: Stray folder → lidar; `.mp4` → video; folder with only
  `walk.mov` → video; folder with nothing → `SystemExit` with a clear message; video shorter than 2 s →
  `CaptureError("video too short ...")`.
- [ ] Model end-to-end test (`@pytest.mark.model`): `run.py sample single_room/rgb.mp4 --rotate 90` →
  schema-valid plan, tier "video", ≥ 1 room.
- [ ] Run → fail; implement; pass; commit.

### Task 6: Video vs LiDAR evaluation and honest ranges — `scripts/eval_video_vs_lidar.py`

**In plain words:** run both tiers on the same sample capture, align the two plans, pair up walls,
and print a table: wall lengths, room areas, ceiling heights, video error vs LiDAR. Then set the video
tier's relative uncertainty (`TIER_REL["video"]`) from the measured scale error so the ranges are
honest, and record the numbers in `docs/TRADEOFFS.md` ("Video tier").

- [ ] Script prints and writes `outputs/eval_video/<capture>.json`; run on single_room (and floor_only if
  time); update `TIER_REL["video"]` with a comment citing the measured numbers; commit.
