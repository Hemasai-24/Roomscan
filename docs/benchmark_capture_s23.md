# Own benchmark capture — Samsung S23 + tape measure (≈ 60-75 min)

Purpose: real photo/video captures of our own rooms with tape-measured ground truth.
The sample data has no ground truth, so these numbers are the only true accuracy we report.

## Before you start (5 min)
- **Pick the space:** 3 rooms + the corridor/hallway that connects them (the brief needs
  "three or more rooms plus a connector"). If possible include the **bathroom** (mirror,
  glass, wet-look tiles — the brief asks us to cover these).
- **Damage room:** in one of the rooms, stage 2 kinds of damage on a wall:
  1. *Water stain:* a sheet of paper (A4) with a brown stain made with strong tea/coffee,
     taped flat on the wall around 1 m high.
  2. *Crack:* a 30-50 cm jagged line drawn with a dark marker on masking tape stuck to the wall.
  Measure each one (width x height, in cm) and its height above the floor.
- **Lights:** all lights on, curtains open. Keep the doors between the rooms open.
- **Phone camera settings (important):**
  - Main camera, **1x**. Never zoom, never 0.6x ultra-wide.
  - Photo ratio **4:3**. Video **1080p, 30 fps**, ratio 16:9 is fine.
  - Settings → turn **off**: Scene optimizer, Auto HDR (if shown), Video stabilization "Super steady".
  - Hold the phone **upright (portrait)** for everything.

## Photos (per room, 2-3 min each)
Make one album/folder per room (or note which photos belong to which room).
1. Stand in **each corner**, point the phone at the **opposite corner**, tilted slightly down
   so the line where wall meets floor is visible. (4 photos)
2. One photo of **each door** from inside the room, the whole door frame in view.
3. Total **6-8 photos** per room. Don't move furniture; don't include people.
4. The corridor counts as a room: photos from both ends looking along it.

## Video (one walkthrough, 1-2 min)
1. Start in room 1, walk slowly (half walking speed) through **every room and the corridor**,
   going through the doors, and come back to where you started.
2. Keep the phone at chest height, pointing roughly forward and slightly down; slowly pan
   left/right so every wall is seen at least once. No fast turns.
3. Then record **room 1 again, on its own, twice** (two separate 30-s clips, same way) —
   this is the repeatability test (same room, same tier, twice).
4. One extra 30-s clip of the bathroom/mirror with the **lights off except one lamp**
   (low-light case).

## Measurements (30 min) — fill `data/ground_truth/own_rooms.csv`
Name walls of each room **W1, W2, ...** going clockwise starting from the wall with the
door you entered by. Measure at floor level, wall to wall (skirting board to skirting board).
- every wall length
- ceiling height in 2 different spots per room
- every door: width (inside the frame) and height
- every window: width, height, sill height above floor
- each staged damage: width, height, height above floor
Tape is fine; write cm to the nearest 0.5 cm.

## Hand-off
Copy everything to the laptop into `~/projects/floorplan-pipeline/data/raw/s23/` like this:
```
data/raw/s23/photos/<room_name>/*.jpg      one folder per room
data/raw/s23/video/walkthrough.mp4
data/raw/s23/video/room1_take1.mp4, room1_take2.mp4, lowlight.mp4
```
(USB cable, or upload to Google Drive and download.) Keep the original files — don't edit,
crop or compress them; the photo metadata (EXIF) tells us the lens focal length.
