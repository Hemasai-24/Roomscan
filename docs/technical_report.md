# Roomscan: technical report

*Max 6 pages. 2026-10-03. Numbers: `docs/benchmark_report.md` (own tape-measured home, Samsung Galaxy M53;
provided sample captures).* Limitations are listed in full in `docs/TRADEOFFS.md`.

## 1. Architecture

```
 LiDAR:  Stray Scanner export ─────────────────────────────────────────────┐
 Video:  .mp4/.mov ─► sharp frames ─► VGGT poses+depth ─► Depth Anything scale ┼─► "capture" = depth + poses + K (metres)
 Photo:  room folders ─► per room: VGGT + Depth Anything + EXIF focal ───────┘        │
                                                                                      ▼
   points ─► planes (seeded RANSAC, normal-consistent) ─► floor / ceiling / walls ─► rooms (watershed at doorways)
     ─► per room: outline snapped to wall planes · floor level · ceiling · openings (free-space carving)
     ─► adjacency (doors) ─► damage (Grounding DINO + SAM2 → measured on its surface) ─► rules R1-R5 ─► scope
     ─► every number as {value, lo, hi} (95 %) ─► plan.json (published schema) + plan.svg/png
```

**One idea carries the design:** every tier is converted into the same intermediate form, depth images,
camera poses and intrinsics in metres, written in Stray Scanner's file layout, and one back end produces the
plan. Tiers differ only in their front end and in how wide their ranges are. This gives the brief's "same
output contract from each tier, intervals that widen as sensor data thins" by construction, and every
improvement to the back end helps all three tiers.

**One command per capture:** `python run.py <input>`; the tier is detected from the input (Stray folder,
video file, folder of room folders). Failures are one readable line (`cannot make a plan: …`), and a room or
video segment that cannot be measured is skipped with a warning rather than losing the capture.

**Determinism:** RANSAC and feature matching are seeded and single-threaded (Open3D's threaded RANSAC returned
different planes on identical input, caught by a test); frame selection is deterministic; same input → same
JSON.

## 2. Tier design and device matrix

| Tier | Phones | Capture tool | Front end | Range model |
|---|---|---|---|---|
| LiDAR | iPhone 12 Pro and later Pro models | Stray Scanner (free) | ARKit poses + LiDAR depth (confidence 2 only) | plane-fit standard error, ≥ 0.8 cm per wall offset |
| Video | any iPhone 15+ | Camera app | VGGT-1B in 20-frame chunks (bf16, 5.5 GB), Depth Anything V2 Metric-Indoor for scale (bias-calibrated) | + relative scale σ |
| Photo | any iPhone 15+ | Camera app | VGGT on each room's 2-8 photos jointly, EXIF focal, Depth Anything scale; rooms stitched at shared doors | + relative scale σ (wide) |

**Capture route:** Route 2, `docs/capture_protocol.md` (one page). Chosen because it needs no Mac,
TestFlight or install step beyond a free app, and the provided sample data is in Stray Scanner's format.
**Device matrix with measured accuracy:** `docs/device_matrix.md`.

**Key engineering facts:**
- Stray Scanner's odometry maps OpenCV-convention camera points to an ARKit world with **+Y up**; verified on
  data (floor normal (0.00, 1.00, 0.00) without an axis flip; walls smeared with one).
- A horizontal RANSAC hypothesis crosses every wall in a thin band; on real scans this produced fake "floors"
  at every height and found only 3 walls. Requiring each point's normal to agree with the plane
  (|n·n_p| > 0.85) gives 32 walls and no fake slices.
- Rooms: distance transform of the free floor, cores > 0.45 m from walls, watershed; open boundaries > 1.2 m
  merge (open plan); tall-wall-only obstacles (counters don't split rooms); wall-enclosed small spaces (WC,
  corridor) kept.
- Openings: a depth ray ending > 15 cm behind a wall line went *through* it → that wall cell is open.
  Unobserved wall is never "open". Door if the opening reaches the floor, else window.

## 3. Drift handling

**Method (`roomscan/drift_correction.py`, `--drift-correction on|off`):** the walk is cut into ~5 s chunks;
chunk 0 is the reference; each further chunk gets a small yaw + 3D shift, solved jointly (ICP-style against
the already-placed wall/floor points of matching orientation), limited to 2° / 10 cm per chunk, then blended
linearly between chunk centres. Poses are never used without this option being available, and the
ablation is reported.

**Ablation (`scripts/drift_ablation.py`):**

| Capture | Wall thickness off → on | Rooms off → on |
|---|---|---|
| synthetic room, injected 3° / 8 cm drift | 34 → 13 mm | n/a |
| with_ceiling (sample) | 58.9 → 56.6 mm | 8 → 10 |
| floor_only (sample) | 52.8 → 54.8 mm | 10 → 8 |

**Honest reading:** the method removes injected drift, but on the real scans the effect is within the noise of
our sharpness metric (dominated by furniture near walls), and correction changes how rooms are split, so the
default is **off**. Next: loop closure, both whole-floor walks end 17 / 39 cm from their start, with a pose
graph, and a drift metric restricted to wall-pair thickness.

## 4. Error budget (LiDAR tier, per wall length)

| Source | Size | How handled |
|---|---|---|
| LiDAR depth noise per point | ~1 cm | averaged by plane fit over thousands of points |
| Plane offset (fit standard error) | ≤ 1 mm statistical | floored at 0.8 cm per offset (systematics) |
| Drift between the two walls' observation times | 0-several cm on long walks | ablation; widened by floor spread |
| Outline snapping (wall plane behind the edge) | 0 when snapped; ~3 cm raster when not | `plane_supported` flag per wall |
| Floor split into slabs (drift / tilt) | 2-8 cm between slabs | slab spread widens ceiling-height range |
| Unseen ceiling | unbounded above | reported as ≥ highest wall point, `ceiling_observed: false` |

Video/photo add a relative scale term (Depth Anything bias residual, leave-one-out −2 % … +4 % on the sample
phone) and pose error (VGGT chaining for video).

**Measured on our own tape-measured home (Galaxy M53):** photo-tier room areas −21 % … +42 %, whole-property
footprint +7.5 % (after the fix loop); decomposition from the oracle experiments: placement/room shape >
poses > scale for photos; poses ≫ scale for video. Scale error on this phone ≈ +0-16 % (walls observed up to
3.18 m in rooms with a 2.74 m ceiling).

## 5. Calibration analysis

Every reported number is a 95 % range. Calibration means: the true value falls inside ~95 % of the time.
- **LiDAR:** no tape ground truth (no LiDAR phone); ranges are plane-fit based and floored at 0.8 cm per wall.
- **Video:** relative σ widened to 0.25 from the sample comparison (LiDAR as reference): LiDAR length inside the
  video range 5/6 (in-sample) and 4/14 (held-out) → still over-confident on held-out data.
- **Photo:** relative σ 0.6; leave-one-capture-out wall coverage 86 % and 93 % vs LiDAR. **On our tape-measured
  home: 4/4 room areas and 16/16 walls inside their ranges, calibrated, but the ranges are so wide (area
  lower bound 0) that they carry little information.** Tightening them needs the accuracy fixed first.
- Rule: never confident garbage, when an input cannot support a number (unseen ceiling, a room from < 2
  photos, a failed reconstruction), the number is either wide and flagged or omitted with a warning.

## 6. The fix loop (`docs/fix_loop.md`, tags `fix-before` / `fix-after`)

**Gate:** photo tier, wall lengths / whole-property footprint (±8 %). **Before:** footprint +30 % on our home
(kitchen +145 %, bathroom +123 %). **Root cause:** a photo room was "all floor its cameras see", including floor
seen through doorways (room boxes 1.2-2.5× too big while scale explained ≤ 1.16×; perfect-data oracle still
+32 %). **Fix:** bound each room by its own walls. **v1** (nearest tall wall from the cameras' centre) fixed the
small rooms but over-cut the large ones (−65 % / −82 %): few photos produce several copies of one wall up to
0.6 m apart, some between the room's own cameras. **v2:** a bounding wall must lie beyond the outermost
cameras (one outlier ignored). **After:** footprint **+7.5 %** (inside ±8 %, but per-room errors partly cancel: summed absolute error ≈ 19 %), kitchen +42 %, bathroom +27 %;
wall-length gate and adjacency still fail. Predicted: footprint within ±15 % (met), small rooms within ±25 %
(missed). Held-out check on the sample is confounded (the LiDAR reference changed); no clean evidence that v2
generalises.

## 7. Doors, windows and damage

**Doors and windows** come from two sources. Depth: a ray that ends beyond a wall passed through an opening
(free-space carving), so an unseen wall is never called an opening. Image: a detector box ("door. window.") is
turned into metres by casting rays through its left, right, top and bottom edges onto the fitted wall; the
same opening in several views is merged. A box cut at the photo's left or right edge is rejected; if only its
top or bottom is cut, the width is kept and the height is reported as unknown. Implausible sizes are rejected.
Photo and video use the image openings; LiDAR keeps its carved openings and adds image openings only where none
was carved (closed doors). On our home's photos this measured 0 of 9 openings within 2 cm, because the photos
have no straight-on, full-frame door shots; the protocol now asks for them.

**Damage:** Grounding DINO boxes, SAM 2.1 masks, each mask measured on its wall/floor/ceiling plane in metres.
A region must be seen in two views on every tier, cover at least 0.003 m², and "peeling paint" needs a score of
0.45. Detectors compared on 10 public damage photos and the undamaged sample flat:

| Detector | Real damage found (10 photos) | False alarms (3 clean captures) |
|---|---|---|
| Grounding DINO, threshold 0.30 (default) | 5 | 4 |
| OWLv2, threshold 0.20 | 4 | 5 |
| YOLO11n-seg fine-tuned on crack-seg (cracks only) | 1 of 4 cracks | 16-105 of 153 frames |

Rules R1-R5 turn damage into concealed-damage flags, each naming its rule; repair items are keyed to surfaces.

## 8. Known failure modes

- **Room shape from a few photos** is the main photo-tier error. Three targeted fixes each repaired the room
  they aimed at and broke others (low furniture as floor, filling furniture notches, UniDepth metric depth);
  see `docs/TRADEOFFS.md`.
- **Mirrors and glass:** video poses flip 110-165° in front of the sample bathroom mirror; LiDAR confidence
  drops at the glass shower; mirrors can create phantom openings. Pose-flip rejection and confidence-2 depth
  reduce this; it is not solved.
- **Wet-look / shiny floors:** reflections and tile grout fool the damage detector; floors keep only mould.
- **Low light:** the 9 s bathroom clip with one small light gave 5.5 m² for a 2.87 m² room (+92 %).
- **Jagged outlines:** furniture along walls makes floor edges ragged, so outlines have many short walls, and
  repeatability across two scans of the flat fails (0/14 walls within 1 cm).
- **Cracks** are found in 1 of 4 public crack photos; damage seen in only one photo is not reported.
- **Video:** only the most consistent piece of a long walk is used (pose tracking breaks into pieces).

## 9. Models and data used (disclosure)
VGGT-1B (Meta; non-commercial research licence), Depth Anything V2 Metric-Indoor-Large (CC-BY-NC-4.0),
Grounding DINO base (Apache-2.0), SAM 2.1 hiera-small (Apache-2.0), OWLv2 base (Apache-2.0, optional
detector); all pretrained and run locally, weights fetched by `scripts/fetch_weights.py`. Experiments only:
UniDepth V2 (CC BY-NC 4.0) and YOLO11n-seg (Ultralytics, AGPL-3.0) fine-tuned on the Ultralytics crack-seg
dataset; public damage photos from Wikimedia Commons (licences in `data/public_damage/credits.txt`). Data: the provided Stray Scanner sample captures (no ground
truth); our own Samsung Galaxy M53 photos/videos with tape measurements (no iPhone was available).
