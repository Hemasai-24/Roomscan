# Roomscan technical report

Roomscan turns a phone capture of a home into a dimensioned floor plan. It accepts three inputs: a LiDAR scan
from Stray Scanner, a walkthrough video, or 2-8 photos per room. Every measurement in the output has a 95 %
range. Results are in `docs/benchmark_report.md`. Limitations are in `docs/TRADEOFFS.md`.

## 1. Design

All three inputs go through one back end. Each tier has a small front end that converts its input into depth
images, camera poses and camera intrinsics in metres, saved in Stray Scanner's file layout. The back end then
builds the plan the same way for every tier. The tiers differ only in the front end and in how wide their
ranges are, so any fix to the back end helps all three.

```
LiDAR scan   ──────────────────────────────────────────┐
Video        ── sharp frames ── VGGT + Depth Anything ──┼── depth + poses + intrinsics (metres)
Room photos  ── per room: VGGT + Depth Anything + EXIF ─┘                │
                                                                         ▼
planes ── floor / ceiling / walls ── rooms ── outlines, openings, ceiling ── damage ── plan.json + plan.png
```

**Front ends**
- LiDAR: reads the Stray Scanner export. Only high-confidence depth is used.
- Video: picks sharp frames, estimates poses and depth with VGGT-1B in chunks of 20 frames (to fit an 8 GB
  GPU), and sets the metric scale with Depth Anything V2 Metric-Indoor.
- Photos: the same models, run on one room folder at a time, with the focal length taken from EXIF. Rooms are
  joined with the doorway photo that appears in both rooms' folders: that photo was taken from one spot, so
  the room is turned and moved until the photo's camera is at the same place in both rooms, and then slid
  up against the hall wall.

**Back end**
1. Fit planes with RANSAC and label them floor, ceiling or wall.
2. Split the floor into rooms at doorways (watershed on distance to the nearest wall).
3. Draw each room outline with right angles and snap its edges to the wall planes.
4. Find doors and windows (section 4).
5. Detect damage, apply the concealed-damage rules, and list repair items.
6. Attach a 95 % range to every number. Write `plan.json` (published schema), `plan.png` and `plan.svg`.

**One command:** `python run.py <input>`. The tier is detected from the input. If a plan cannot be made, the
tool prints one line saying why. A room or video segment that cannot be measured is skipped with a warning.

**Same input, same output:** RANSAC and feature matching are seeded and single-threaded. Open3D's RANSAC
returned different planes for the same input, so we wrote our own.

## 2. Tiers and devices

| Tier | Phone | App | How depth and poses are obtained | Range |
|---|---|---|---|---|
| LiDAR | iPhone 12 Pro or later Pro | Stray Scanner (free) | ARKit poses, LiDAR depth | plane-fit error, at least 0.8 cm per wall |
| Video | any recent phone | camera app | VGGT poses and depth, Depth Anything scale | plus a relative scale term (σ 0.25) |
| Photo | any recent phone | camera app | the same, per room, with EXIF focal length | plus a larger scale term (σ 0.6) |

The capture route is Route 2: Stray Scanner for LiDAR and the normal camera app for video and photos. It needs
no Mac or developer account, and the sample data was already in Stray Scanner format. The capture steps fit on
one page (`docs/capture_protocol.md`). Measured accuracy per tier is in `docs/device_matrix.md`.

## 3. Decisions that changed the results

- **Axes.** Stray Scanner poses already have +Y up, so no axis flip is needed. With the flip the walls smeared;
  without it the floor normal is (0, 1, 0).
- **Plane fitting.** Plain RANSAC found fake "floors" at every height, because a horizontal plane cuts through
  every wall in a thin band. Requiring each point's surface normal to agree with the plane removed them and
  raised the wall count from 3 to 32 on the sample.
- **Room splitting.** Only tall obstacles split rooms, so kitchen counters and beds do not. Small rooms closed
  by walls, such as a toilet, are kept.
- **Openings.** A wall is only called open where a depth ray passed through it. A wall that was never seen is
  not an opening.
- **Unseen ceilings.** If the ceiling never appears in the capture, the ceiling height is reported as a wide
  range and flagged `ceiling_observed: false`.

## 4. Doors, windows and damage

**Doors and windows** come from two sources. From depth: rays that pass through a wall mark an opening; it is a
door if it reaches the floor and a window otherwise. From images: a detector draws boxes for "door" and
"window", and the box edges are projected onto the fitted wall to get width and height in metres. A box cut off
at the left or right edge of the photo is rejected. If only the top or bottom is cut off, the width is kept and
the height is marked unknown. Photo and video use the image openings. LiDAR keeps its depth openings and adds
image openings only where depth found none, which covers closed doors.

**Damage** is found with Grounding DINO (boxes from text such as "water stain" and "crack") and SAM 2.1 (exact
outlines). Each outline is placed on its wall, floor or ceiling and measured in square metres. To cut false
alarms, a region must appear in at least two views, cover at least 0.003 m², and "peeling paint" needs a score
of at least 0.45. Five rules (R1-R5) turn damage into concealed-damage flags. For example, R1 flags a water stain on
a ceiling as a possible leak from above, and R4 flags a crack longer than 1 m as possible structural movement. Each flag names its rule, and each repair item names the surface it applies to.

## 5. Drift

Long walks drift: the phone's position estimate slowly goes wrong, so the same wall can appear twice.

**Method** (`roomscan/drift_correction.py`, switch `--drift-correction on|off`): the walk is cut into chunks of
about 5 seconds. Each chunk is aligned to the walls and floor already placed by earlier chunks, allowing at
most 2° of rotation and 10 cm of shift. Corrections are blended between chunks.

| Capture | Wall thickness, off → on | Rooms, off → on |
|---|---|---|
| synthetic room with 3° / 8 cm drift added | 34 → 13 mm | - |
| sample, with_ceiling | 58.9 → 56.6 mm | 8 → 10 |
| sample, floor_only | 52.8 → 54.8 mm | 10 → 8 |

The method removes drift that we added on purpose. On the real scans the change is within the noise of the
measure, which is dominated by furniture near walls, and it changes how rooms are split. It is therefore off
by default. The next step would be loop closure: both whole-floor walks end 17 cm and 39 cm from where they
started.

## 6. Error budget and calibration

| Error source (LiDAR, per wall) | Size | Handling |
|---|---|---|
| depth noise per point | about 1 cm | averaged over thousands of points by the plane fit |
| plane position | under 1 mm statistically | floored at 0.8 cm |
| drift between two walls | 0 to several cm on long walks | drift option; floor spread widens the range |
| outline not snapped to a wall plane | about 3 cm | flagged per wall (`plane_supported`) |
| unseen ceiling | unbounded | wide range, flagged |

Video and photo add a scale error (Depth Anything on this kind of scene: −2 % to +4 % on the sample iPhone,
0 to +16 % on our Galaxy M53) and pose errors.

**Calibration.** A 95 % range is calibrated if the true value falls inside it about 95 % of the time. On our
tape-measured home, the truth is inside the photo-tier range for 4 of 4 room areas and 15 of 16 walls. The
ranges are therefore honest, but they are too wide to be useful (the lower bound of every room area is 0).
Narrower ranges need better accuracy first. There is no tape check for LiDAR because we had no LiDAR phone.

## 7. Fix loop

The worst gate was the photo-tier footprint: +30 % on our home, with the kitchen at +145 % and the bathroom at
+123 %. The cause was that a room included all the floor its cameras could see, including floor seen through
doorways. The fix bounds each room by its own walls, using only walls that lie beyond the outermost camera.
After the fix the footprint is +7.5 % and the kitchen and bathroom are +42 % and +27 %. Per-room wall lengths
still fail. Details: `docs/fix_loop.md`.

## 8. Known failure modes

- **Room shape from a few photos** is the main photo-tier error (per room −21 % to +42 %).
- **Video poses** break up on long walks, so only the longest consistent piece is used.
- **Mirrors and glass** flip video poses and can look like openings.
- **Shiny floors** fool the damage detector, so floors only report mould.
- **Low light**: a 9 s bathroom clip with one small light gave +92 % area.
- **Furniture along walls** makes outlines jagged, which also breaks repeatability.
- **Cracks** are found in 1 of 4 public crack photos. Damage seen in only one photo is dropped.
- **Doors** are only measured when a photo shows the whole door straight on.

## 9. Models and data

All models are pretrained, run locally, and are downloaded by `scripts/fetch_weights.py`.
- VGGT-1B (Meta, non-commercial research licence): poses and depth for video and photos.
- Depth Anything V2 Metric-Indoor-Large (CC-BY-NC-4.0): metric scale.
- Grounding DINO base (Apache-2.0) and SAM 2.1 hiera-small (Apache-2.0): damage, doors and windows.
- OWLv2 base (Apache-2.0): optional alternative damage detector.
- Tried and not used: UniDepth V2 (CC-BY-NC-4.0); YOLO11n-seg (AGPL-3.0) fine-tuned on Ultralytics crack-seg.

Data: the three Stray Scanner sample captures provided with the brief (no ground truth), our own Galaxy M53
photos and videos with tape measurements, and 10 public damage photos from Wikimedia Commons (licences in
`data/public_damage/credits.txt`).
