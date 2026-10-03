# Own benchmark capture — Samsung S23 + tape (≈ 75 min)

Our only real ground truth. Follow in order.

## 1. Phone settings (5 min) — Camera app → gear icon
- Scene optimizer **OFF** · Video stabilisation / Super steady **OFF** · HDR10+ video **OFF** · Motion photos **OFF**
- Video size **FHD 1080p, 30 fps** · photos JPEG (default)
- Always the **1x** lens (never 0.6x, never zoom). Hold the phone **upright (portrait)** for everything.

## 2. Prepare (10 min)
- Choose **3 rooms + the corridor** joining them (include the bathroom if possible).
- All lights on, curtains open, **doors between them open**. Don't move furniture afterwards.
- Name them on paper: room1, room2, room3, corridor (+ which doorway joins which).
- **Staged damage** in one room, one wall, ~1 m high:
  - water stain: A4 sheet soaked in strong tea/coffee, slightly dried, taped **flat**;
  - crack: 40-50 cm masking-tape strip with a **thick dark zig-zag** marker line.

## 3. Photos (≈ 3 min per room)
In each room (corridor too), finish one room before the next:
1. **4 corner photos:** stand in each corner, aim at the opposite corner, tilted slightly down so the
   wall-floor line shows.
2. **1 photo of each door** from inside the room (whole frame visible).
3. **Doorway photos (important):** stand IN each doorway between two rooms and take **one photo looking
   into each of the two rooms**. Note them, e.g. "doorway room1-corridor: last 2 photos".
4. Damage room only: 1 photo of the damage from ~1.5 m.
5. Corridor: 2-3 photos from each end looking along it.
Write the count per room as you go ("room1: 8 photos, doorway room1-corridor: 2").

## 4. Videos
1. **Walkthrough (1-2 min):** start in room1, walk at **half speed** through every room and the corridor,
   through the doors, **end where you started**. Chest height, aim forward and slightly down, pan slowly so
   every wall is seen once, glance up at each ceiling.
2. **Room1 twice:** two separate 30-s clips of room1 only, recorded the same way (repeatability).
3. **Low light:** 30-s clip of the bathroom with only one small light on.

## 5. Tape measurements (≈ 30 min)
Walls of each room are **W1, W2, …** clockwise, starting with the wall holding the door you entered by.
Measure at floor level, to the nearest 0.5 cm:
- every wall length (corner to corner)
- ceiling height in 2 spots (C1, C2)
- every door: width inside the frame, height (D1, D2 …)
- every window: width, height, sill height above floor (N1 …)
- damage: stain width × height and height of its bottom edge above floor; crack length
Format: `room1 W1 412.5` · `room1 C1 268` · `room1 D1 width 82 height 205` · `damage stain 21x29.7 bottom 98`
(or photograph the paper sheet).

## 6. Optional: Polycam (20 min)
Install Polycam (free) on the S23, scan 2 of the same rooms in **Photo mode**, export. Labelled as a
photo-mode comparison (the brief asks for LiDAR; no LiDAR phone available).

## 7. Hand-off
Copy everything **unedited** (USB cable or Google Drive) to `~/projects/floorplan-pipeline/data/raw/s23/`:
`photos/` (all photos, unsorted is fine) and `video/` (all clips). Send the measurements + photo counts;
we sort photos into room folders together.
