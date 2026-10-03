# Roomscan — technical report

*Max 6 pages. Draft 2026-10-03 07:00; sections marked **[pending S23 benchmark]** are filled from the
tape-measured benchmark captured on 2026-10-03.* Limitations are listed in full in `docs/TRADEOFFS.md`.

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

**One idea carries the design:** every tier is converted into the same intermediate form — depth images,
camera poses and intrinsics in metres, written in Stray Scanner's file layout — and one back end produces the
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

**Capture route:** Route 2 — `docs/capture_protocol.md` (one page). Chosen because it needs no Mac,
TestFlight or install step beyond a free app, and the provided sample data is in Stray Scanner's format.
**Device matrix with measured accuracy:** `docs/device_matrix.md` **[accuracy column pending S23 benchmark]**.

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
| synthetic room, injected 3° / 8 cm drift | 34 → 13 mm | — |
| with_ceiling (sample) | 58.9 → 56.6 mm | 8 → 10 |
| floor_only (sample) | 52.8 → 54.8 mm | 10 → 8 |

**Honest reading:** the method removes injected drift, but on the real scans the effect is within the noise of
our sharpness metric (dominated by furniture near walls), and correction changes how rooms are split, so the
default is **off**. Next: loop closure — both whole-floor walks end 17 / 39 cm from their start — with a pose
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
phone) and pose error (VGGT chaining for video). **[Per-tier measured budget pending S23 benchmark.]**

## 5. Calibration analysis

Every reported number is a 95 % range. Calibration means: the true value falls inside ~95 % of the time.
- **LiDAR:** **[pending S23 benchmark — coverage of tape values]**.
- **Video:** relative σ widened to 0.25 from the sample comparison (LiDAR as reference): LiDAR length inside the
  video range 5/6 (in-sample) and 4/14 (held-out) → still over-confident on held-out data.
- **Photo:** relative σ 0.6; leave-one-capture-out wall coverage 86 % and 93 % vs LiDAR.
- Rule: never confident garbage — when an input cannot support a number (unseen ceiling, a room from < 2
  photos, a failed reconstruction), the number is either wide and flagged or omitted with a warning.

## 6. The fix loop **[pending — declared after the S23 benchmark]**

Evidence collected beforehand: oracle ablations that replace our scale / poses with LiDAR truth to locate
where video/photo error comes from (`docs/experiments/tier_accuracy.md`). Declaration: `docs/fix_loop.md`;
before/after runs regenerable from tags `fix-before` / `fix-after`.

## 7. Known failure modes

- **Mirrors and glass:** video poses flip 110-165° in front of the sample bathroom mirror; LiDAR confidence
  drops at the glass shower; mirrors can create phantom openings. Mitigations: pose-flip rejection, keeping the
  most consistent pose segment, confidence-2 depth only. Not solved.
- **Wet-look / shiny floors:** reflections and tile grout fool the damage detector; floors keep only mould.
- **Low light:** **[pending S23 low-light clip]**.
- **Closed doors** are invisible to free-space carving (flat like a wall).
- **Jagged outlines:** furniture along walls makes floor edges ragged → many short wall segments; repeatability
  across two scans of the flat fails (0/14 walls within 1 cm).
- **Thin cracks** are often missed; a stain across an inside corner is reported per wall.
- **Video:** only the most consistent piece of a long walk is used; **photo:** wall error median 28 % on
  simulated photos.

## 8. Models and data used (disclosure)
VGGT-1B (Meta; non-commercial research licence), Depth Anything V2 Metric-Indoor-Large (CC-BY-NC-4.0),
Grounding DINO base (Apache-2.0), SAM 2.1 hiera-small (Apache-2.0); all pretrained, no training, weights
fetched by `scripts/fetch_weights.py`, run locally. Data: the provided Stray Scanner sample captures (no ground
truth); our own Samsung S23 photos/videos with tape measurements (no iPhone was available).
