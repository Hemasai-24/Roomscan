# Trade-offs and limitations

This document lists what we could not do, what the code assumes, and what we tried that did not work.

## What we could not do

| Brief asks for | What we did | Why |
|---|---|---|
| Our own LiDAR captures and LiDAR accuracy against tape | LiDAR tier tested on the provided sample only | no LiDAR iPhone available; our phone is a Samsung Galaxy M53 |
| Comparison with Polycam | not done | needs a LiDAR iPhone |
| All three tiers on the staged-damage room | photo and video only | same reason |
| Ground truth for the sample data | LiDAR used as the reference for video and photo | the hiring team confirmed none exists |
| Video and photo of the sample flat | simulated from the sample's `rgb.mp4` | the sample has no separate video or photos |
| Laser ground truth | tape measure (about ±0.5 cm) | no laser meter |

## Assumptions

**Geometry**
- Walls meet at right angles. Angled or curved walls are squared off.
- One floor level per capture. A stair void can be read as a very high ceiling (seen once: 6.14 m).
- A wall is vertical, flat and at least 1.4 m tall. Tall furniture such as a wardrobe counts as wall.
- An opening is where depth passes through a wall. Closed doors are invisible to depth alone.
- A room's real wall never lies between two of its own photo positions (fix loop rule).
- If the ceiling is not seen, its height is a wide range, never a confident number.

**Scale and cameras**
- Depth Anything's scale correction, fitted on the sample iPhone, holds for other phones. On the Galaxy M53 the
  scale came out 0 to 16 % large.
- The focal length in the photo's EXIF data is correct. Without EXIF the model guesses it, 8-22 % low on the
  sample.
- The phone is held roughly upright. Other orientations need `--rotate`.

**Our measurements**
- Rooms are rectangles; we taped two lengths per room.
- Window sizes were written as width then height; the order was not stated.
- Window sill heights were not measured, and the ceiling was measured at one spot per room.

## Experiments that did not work

Each was measured and then dropped. The code is kept on the named branch.

| Change | Result | Why it was dropped |
|---|---|---|
| Treat low furniture as floor (`low-furniture`) | bedroom −21 % → −6 % | hall got worse (+25 %), home total +18 % |
| Fill shallow furniture notches (`fill-furniture-notches`) | bedroom → −9 % | kitchen +74 %, home total +13 % |
| UniDepth V2 instead of Depth Anything (`exp-metric-depth`) | scale far better (0.998× LiDAR depth) | room rules were tuned for the old scale; home total +39 % |
| Find shared doorway photos automatically (`auto-doorway-share`) | 1 of 3 doorways found | the kitchen grew to +229 % |
| Merge short wall steps in outlines | fewer walls | area error 25 % → 27-29 % |
| OWLv2 as damage detector | 4 of 10 public photos | Grounding DINO finds 5 with fewer false alarms |
| YOLO11 crack model trained on crack-seg | 1 of 4 wall cracks | trained on pavement, not indoor walls |
| Microsoft building-damage model | not run | it rates whole buildings from satellite images |

The pattern in the first three rows: each change fixed the room it targeted and broke another. Building room
shapes from a few photos needs a different method, not more rules.

## Known gaps

- **Mirrors and glass.** Video poses flip in front of the sample bathroom mirror. Glass can look like an
  opening. We use only high-confidence LiDAR depth and drop flipped poses, but this is not solved.
- **Shiny floors.** Reflections and tile grout look like cracks, so on floors we only report mould.
- **Low light.** The 9 s bathroom clip with one small light gave +92 % area.
- **Video tracking.** VGGT runs on 20 frames at a time on an 8 GB GPU. The chunks drift apart, so only the
  longest consistent piece of a walk is used. A larger GPU or a global pose graph would help.
- **Doors and windows** are measured only when a photo shows the whole opening straight on. Our photos did not,
  so 0 of 9 were within 2 cm.
- **Damage seen in only one photo is dropped.** This removed all 12 false alarms on our home, but it also
  dropped the staged water stain.
- **Thin cracks** are hard for every detector we tried.
- **Drift correction** is off by default. It removes added drift on synthetic data but shows no gain on the real
  scans and changes how rooms are split.
- **Fix loop tuning.** The fix-loop rule was adjusted while looking at our own home, so it may be tuned to it.
- **Model licences.** VGGT-1B and Depth Anything V2 Large are non-commercial. A product would need the
  commercial VGGT checkpoint or other models.
