# Plan 4: Photo Tier (per-room photo folders → one stitched whole-property plan)


**Goal:** `python run.py <folder of room folders>` (each subfolder = one room, 2-8 photos, no depth, no
poses) produces ONE whole-property `plan.json` / `plan.svg` with tier `"photo"`: every room placed,
connected, dimensioned, no overlaps, with honest (wide) ranges.

**Approach in plain words:**
1. For each room folder, the 3D model (VGGT, same as the video tier) looks at all of that room's photos
   together and works out where each photo was taken from and how far away everything is.
2. Depth Anything gives real-world size (metres), as in the video tier. If the photos carry the lens
   focal length (EXIF, every phone writes it), we use it instead of the model's guess.
3. The result is packaged like a Stray Scanner capture and measured by the same back end as LiDAR,
   but as ONE room: the room the photos were taken from (camera positions), not the rooms seen
   through its doors.
4. Rooms are then put together like puzzle pieces: if a photo in room A shows the inside of room B
   (through the door), the two rooms share that door. Room B is turned and moved so its door lands
   on room A's door, facing it, one wall-thickness apart. A room is never allowed to overlap another;
   a room that cannot be connected is still drawn (beside the plan) with a warning.

**Spec:** `docs/design.md`. Builds on Plan 3 (`roomscan/video_*`).

## Feasibility evidence (spike on sample data, 2026-10-03, RTX 2000 Ada 8 GB)
Photo folders simulated from `single_scan_floor_only`: LiDAR pipeline rooms → for each room, frames
whose camera is inside it, one sharp frame per viewing-direction sector (8 sectors), preferring frames
that look across the room. 3-6 photos per room (the walkthrough faces few directions per room).
- VGGT on one room's photos jointly: **0.6-1.4 s** per room, fits in 8 GB (≤ 8 photos at 392×518).
- VGGT focal length is **8-22 % low** vs. the true focal (sample: 333-403 px vs 435 px at 392×518).
  Using the EXIF focal (simulated from the true intrinsics) instead changed room areas by
  up to ±40 % — so EXIF is used when present.
- Per-room area vs LiDAR (EXIF focal): -60 % … +357 % (median |error| ≈ 33 %). Causes: 3-6 views
  leave wall sections unseen, so the room "leaks" into floor seen through doors; the LiDAR "rooms"
  used as reference are themselves fragments of the real rooms; 2 of 10 rooms had too little floor
  in view ("no floor found").
- Conclusion: the photo tier will not meet the ±8 % gate on this data. It must (a) never crash,
  (b) stitch, (c) report ranges wide enough to contain the truth (calibrated on this benchmark).

## Global Constraints
- Same input → identical output (seeded RANSAC; deterministic SIFT/RANSAC via `cv2.setRNGSeed(0)`).
- Default test suite needs no GPU; VGGT/Depth Anything tests use `@pytest.mark.model`.
- Plain commit messages; no co-author trailers.
- Photo folders: any of `.jpg .jpeg .png .heic`-less (`.jpg/.jpeg/.png`); 2-8 photos per room (fewer → warning,
  room still reported if it reconstructs; 1 photo → room skipped with a warning).

## Review Focus
1. **Room folder whose photos show almost no floor** — must not crash the whole property: floor from the
   camera-height prior (phone ~1.4 m above floor), warning. Test in Task 3.
2. **Room with no detected door** — still stitched if visual evidence links it (virtual door on the wall
   the matching photo looks at). Test in Task 4.
3. **Room that matches nothing** — placed beside the plan, warning, never dropped. Test in Task 4.
4. **Two placements that would overlap** — pushed apart along the door normal until overlap ≤ 0.1 m². Test in Task 4.
5. **Ranges** — must contain the LiDAR value for most rooms in the simulated benchmark (coverage reported). Task 6.

---

### Task 1: Find photo folders and read photos (`roomscan/photo_folders.py`)
**In plain words:** decide that the input is "photos" (a folder whose subfolders contain only images),
list the rooms, load each room's photos upright at the model's size, and read the lens focal length
from EXIF (35 mm-equivalent), converting it to pixels.

**Interfaces:**
- `is_photo_set(path) -> bool`
- `room_folders(path) -> list[Path]` (sorted; subfolders with ≥ 1 image)
- `load_room_images(folder, size=(392, 518)) -> (np.uint8[N,H,W,3], fx_px | None, list[str])`
- `focal_px_from_35mm(f35, w, h) -> float` (`f35 / 43.27 * hypot(w, h)`)
- `pipeline.detect_tier` returns `("photo", path)` for photo sets.

**Tests (`tests/test_photo_folders.py`):** detection true/false; rooms sorted; EXIF focal read back from
a JPEG written with Pillow; portrait/landscape images both returned at `size`; `detect_tier` → photo.

### Task 2: One room's photos → capture (`roomscan/photo_room.py`, model)
**In plain words:** VGGT on all of the room's photos at once → poses + depth; scale to metres with Depth
Anything (calibrated bias); level so "up" is +Y (phones are held upright); write as a Stray-layout
capture. If EXIF focal is present it replaces VGGT's focal when packaging.

**Interfaces:** `photos_to_capture(images, fx_exif, out_dir, runner, metric) -> (cap_dir, info)`;
`info = {"n_photos", "focal_px", "focal_source", "metric_scale", "metric_scale_spread"}`.
**Tests:** `@pytest.mark.model` end-to-end on a simulated sample room folder (capture loads, frames == photos,
camera heights 0.8-2.0 m above the detected floor); no-GPU test of the packaging with a fake runner/metric.

### Task 3: Measure one room from its capture (`roomscan/photo_room.py`)
**In plain words:** run the LiDAR back end steps on the capture but keep only the room the cameras stand
in (union of split pieces that contain a camera position). If no floor plane is found, assume the floor
is 1.4 m below the median camera height (warning). If geometry still fails, report the rectangle around
the points near the cameras, with very wide ranges and a warning.

**Interfaces:** `measure_photo_room(cap_dir, room_id, tier="photo") -> dict` with keys
`room` (schema room record, local coordinates), `cameras_plan` (N×2), `forward_plan` (N×2),
`angle`, `warnings`.
**Tests:** synthetic box-room capture (render_box_depth): one room, area within 5 %; capture with the
floor cut out of the depth → floor-prior path, warning; cameras outside every piece → fallback rectangle.

### Task 4: Stitch rooms (`roomscan/stitch_rooms.py`, pure geometry)
**In plain words:** see "Approach" step 4. Greedy: start from the room with the most links, repeatedly
add the best-scoring (placed room, door, new room, door) pair; score = visual evidence + door-width
agreement − overlap; resolve residual overlap by pushing the new room away; unplaced rooms go in a
row beside the plan.

**Interfaces:**
- `door_frames(room) -> list[dict]` each `{"id", "mid": (2,), "normal_out": (2,), "width"}`
- `virtual_door(room, camera, forward, width=0.8) -> dict` (on the wall the photo looks at)
- `place(room_b, door_b, door_a, wall_thickness=0.15) -> (R2, t)` (B's door faces A's door)
- `transform_room(room, R2, t) -> dict`
- `stitch(rooms: list[dict], links: dict[(i, j)] -> {"matches", "photo_i", "photo_j"}) -> (rooms_placed, adjacency, warnings)`
**Tests (`tests/test_stitch_rooms.py`):** two rectangles with doors → B placed on the far side of A's door,
door mids coincide (± wall thickness), no overlap; three rooms in a row; room without door linked by
evidence → virtual door; unlinked room → placed beside, warning; forced overlap → pushed apart ≤ 0.1 m².

### Task 5: Visual links between rooms (`roomscan/photo_links.py`)
**In plain words:** for every pair of rooms, count reliable feature matches between their photos (SIFT,
ratio test, geometric check). Many matches between photo p of room A and photo q of room B means p looks
into B (or q into A).
**Interfaces:** `room_links(images_by_room: list[np.ndarray[N,H,W,3]], min_matches=25) -> dict[(i,j)] -> {"matches", "photo_i", "photo_j"}`
**Tests:** synthetic: two "rooms" whose photos share a textured patch → linked; unrelated textures → not;
deterministic (two runs equal).

### Task 6: Photo pipeline, ranges, CLI (`roomscan/photo_pipeline.py`, `run.py`)
**In plain words:** tie Tasks 1-5 together; add the metric-scale uncertainty to every length; photo
ranges per `measurements.py` tables, recalibrated from the simulated benchmark (Task 7).
**Interfaces:** `run_photos(path, out_dir) -> plan`; `run.py <photo set>` auto-detects; `--tier photo`.
**Tests:** no-GPU test with monkeypatched reconstruction (synthetic room records) → schema-valid plan with
tier "photo", adjacency, no overlaps.

### Task 7: Simulated benchmark (`scripts/make_photo_folders.py`, `scripts/eval_photo_vs_lidar.py`)
**In plain words:** from a LiDAR sample capture, make per-room photo folders (as in the spike, plus one
door-facing photo per detected door), run the photo tier, compare against the LiDAR plan: per-room
area and wall lengths, total footprint, adjacency found vs LiDAR adjacency, overlaps, range coverage.
**Tests:** pure helpers (sector selection, door-facing choice) unit-tested.
