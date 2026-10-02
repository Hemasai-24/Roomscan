# Plan 5: Damage, Concealed-Damage Rules and Repair Scope

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** DRAFT for the user's review (written 2026-10-03 ~01:45 while the photo tier was being built).

**Goal:** every plan.json gets `damage[]` (class + metric size, keyed to the wall/floor/ceiling
it is on), `concealed_damage_flags[]` (each naming the rule that fired) and `scope_items[]`
(repair line items keyed to surfaces, with quantities and ranges) — for all three tiers.

**Approach in plain words:**
1. Take a few sharp colour frames (LiDAR/video) or the photos (photo tier). Every tier already
   has, per frame, a depth image and a camera pose (the video/photo front ends produce them).
2. **Find damage:** Grounding DINO gets the text "water stain. crack. mold. peeling paint. hole"
   and returns boxes; **SAM2** traces the exact outline (mask) inside each box.
3. **Measure it:** each mask pixel + its depth + the camera pose → 3D points → which surface
   they sit on (nearest wall/floor/ceiling plane) → area in m² and width/height on that surface.
4. **Merge** the same stain seen in several frames (same surface, centres within 30 cm).
5. **Rules** (plain if-then, each with an id) flag damage you can't see: e.g. a ceiling stain →
   "possible leak above".
6. **Scope:** each damage + each flag → repair line items for its surface (e.g. "stain-block
   primer + repaint wall room_2_w3: 11.2 m² [10.9, 11.5]").

**Model slot (swappable, as agreed):** detector = Grounding DINO (base), segmenter = SAM2.1
(small). Comparison candidate: OWLv2. All pretrained, weights fetched by
`scripts/fetch_weights.py`, disclosed in TRADEOFFS (name, version, licence, source).

**Ground truth:** the staged water stain (A4 sheet) and marker crack in the S23 benchmark room,
tape-measured (`data/ground_truth/own_rooms.csv`). Sample data has no damage → only a
"no false alarms" check there (count of detections on the sample flat, all reviewed by eye).

## Global Constraints
- Same input → same output: fixed frame selection, models in eval mode, no sampling.
- Damage measurements carry ranges (pixel quantisation + depth noise + tier scale term).
- Rules are data (a table in code), each with `rule_id`, `description`, `evidence`.
- Default test suite must not need the GPU (`model` marker for model tests).
- Plain commit messages, no co-author trailers.

## Review Focus
1. Shadows / dirt / wood grain detected as "stain" → false alarms. Expected: score threshold +
   minimum 2 views (LiDAR/video) or flagged `single_view` (photos).
2. Damage on furniture, not a building surface → dropped (not within 5 cm of a wall/floor/ceiling plane).
3. Mask spilling across a corner onto two walls → split per surface, not one region.
4. Photo tier with 1 view per stain → kept but range widened, `n_views: 1`.
5. A rule that fires must name its evidence (damage ids, surface ids).

---

### Task 1: Rules table and scope generator (pure logic, no models)

**In plain words:** first the part that needs no AI: given a list of damage records (class,
surface, size, height above floor), decide which hidden-damage rules fire and which repair
line items to list. Fully testable with made-up damage.

**Files:** Create `roomscan/damage_rules.py`, `roomscan/repair_scope.py`; Test `tests/test_damage_rules.py`, `tests/test_repair_scope.py`

**Interfaces:**
- Consumes: room records (`rooms[]` with walls/surface_id, floor_area, ceiling), adjacency
- Produces: `RULES: list[dict]` (id, description), `concealed_flags(damage, rooms, adjacency) -> list[dict]`
  each `{"rule_id", "description", "damage_ids", "surface_ids", "severity"}`
- Produces: `scope_items(damage, flags, rooms) -> list[dict]` each
  `{"id", "surface_id", "action", "quantity": Measure-dict, "unit", "reason": damage/flag id}`

Rules (first version):
| id | fires when | meaning |
|---|---|---|
| R1 | water_stain or mold on a **ceiling** | possible leak from above (roof, bathroom, pipe) |
| R2 | water_stain / mold on a wall with bottom edge < 0.3 m above floor | possible rising damp / leak at floor level |
| R3 | any mold | hidden moisture behind the surface likely; inspect the cavity |
| R4 | crack longer than 1.0 m, or starting within 0.2 m of a door/window corner | possible structural movement |
| R5 | water_stain / mold on a wall shared (via adjacency) with a room whose floor is ≥ 3 cm lower (wet room) | possible pipe leak inside the wall |

Scope actions (first version): water_stain → `stain_block_and_repaint` (whole surface area);
mold → `mold_remediation` (damage area × 1.5, min 1 m²) + `repaint`; crack → `fill_and_repaint`
(crack length; whole wall repaint); hole → `patch_and_repaint`; peeling_paint →
`scrape_prime_repaint` (whole surface); each flag → `inspect` (moisture meter / structural
engineer) on its surfaces.

- [ ] Step 1: failing tests for each rule firing / not firing with hand-made damage records;
  scope quantities for a 4 x 3 m room wall (area = length x ceiling height, with range).
- [ ] Step 2: run, see them fail. Step 3: implement as a table-driven module. Step 4: pass. Step 5: commit.

### Task 2: Damage detector wrapper (Grounding DINO + SAM2)
**In plain words:** one function: image in → list of (class, score, mask). Hidden behind an
interface so OWLv2 or another model can be swapped in later.

**Files:** Create `roomscan/damage_detect.py`; Test `tests/test_damage_detect.py` (`model` marker
+ a pure test of the post-processing: score threshold, class mapping, mask clean-up).
- Produces: `detect_damage(image_rgb, detector="gdino+sam2", threshold=0.35) -> list[dict]` each
  `{"class", "score", "mask": bool(H,W), "box": (x0,y0,x1,y1)}`; classes ∈
  {water_stain, crack, mold, peeling_paint, hole}.

### Task 3: Masks → measured damage on surfaces
**In plain words:** turn each mask into 3D using the frame's depth and pose, find which surface
it lies on, measure its area and size on that surface, split across corners, drop if not on a
building surface; merge the same damage seen in several frames.

**Files:** Create `roomscan/damage_measure.py`; Test `tests/test_damage_measure.py` (synthetic:
a known 0.3 x 0.2 m square mask on a wall at 2 m depth → area 0.06 m² ± 10%; a mask on a
floating table top → dropped; a mask across a corner → two regions).
- Produces: `measure_damage(detections_per_frame, frames, rooms_surfaces, tier) -> list[dict]` each
  `{"id", "class", "surface_id", "room_id", "area": Measure, "width": Measure, "height": Measure,
  "bottom_above_floor": Measure, "n_views", "score"}`

### Task 4: Wire into all three tiers + evaluate
**Files:** Modify `roomscan/pipeline.py`, video/photo pipelines, `roomscan/save_plan.py`,
`roomscan/draw_plan.py` (damage drawn as red hatched marks on its wall, flags listed under the plan);
Create `scripts/eval_damage.py` (staged damage vs tape GT; detections on sample flat for review).
- `run.py ... --no-damage` to skip (speed).
- Record: detection of both staged damages per tier, size error vs tape, false alarms on sample.
