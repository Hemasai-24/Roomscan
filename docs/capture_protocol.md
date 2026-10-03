# Capture protocol (one page): follow it literally

You need: an **iPhone 15 or newer**. LiDAR tier needs a **Pro / Pro Max**. About 10 minutes.

## 1. Install (1 minute)
- **LiDAR tier:** App Store → install **"Stray Scanner"** (free, by Stray Robots). Allow camera access.
- **Video / Photo tier:** nothing to install; use the built-in **Camera** app.

## 2. Prepare the space (2 minutes)
- Turn **all lights on**, open curtains, **open the doors** between the rooms you capture.
- Do not move furniture once you start. Keep people out of view.

## 3. Capture: choose ONE tier
**LiDAR tier (Stray Scanner, Pro iPhone)**
1. Open Stray Scanner, tap **record**.
2. Hold the phone **upright** at chest height, pointing **forward and slightly down** so the
   line where walls meet the floor is in view most of the time.
3. Walk **slowly (half walking speed)** through every room. In each room, turn slowly so
   **every wall is seen once**, then tilt up briefly to see the **ceiling**.
4. Walk through the doors (do not look into a room only from the doorway).
5. **Finish where you started.** Stop recording. Typical length: 1 minute per room.

**Video tier (Camera app)**
- Camera → **Video**, **1x** lens (never 0.5x, never zoom), 1080p or 4K, 30 fps.
- Same walk as the LiDAR tier (steps 2-5). One continuous clip.

**Photo tier (Camera app)**
- Camera → **Photo**, **1x** lens, no zoom, no Portrait/Panorama modes. Hold the phone
  **upright (portrait)** for every photo. Turn **Live** off (top of the camera screen).
  HEIC ("High Efficiency") and JPEG ("Most Compatible") both work.
- In **each room**: stand in each **corner** and photograph the **opposite corner**, tilted slightly
  down so the wall-floor line shows (4 photos); then **1 photo of each door** from inside the room.
  **Doorways:** stand in each doorway between two rooms and take **one photo looking into the room
  you are leaving**; put a **copy of that same photo in both rooms' folders** (this is how rooms get joined).
  **Each door and window:** one photo straight on, the whole frame visible (floor to top).
  **Damage:** photograph it **twice, from two different positions** (damage seen once is not reported).
  **2-8 photos per room** (plus the doorway copies). Keep each room's photos together (e.g. one album per room); a
  single room may also be passed as one plain folder of photos.

## 4. Avoid
- Fast turns or running; pointing at the floor or ceiling only; covering the camera.
- Standing in front of a **mirror** for long (it shows a fake room); with glass doors, look past them.
- Zoom, ultra-wide (0.5x), Live/Portrait/Panorama modes, filters.

## 5. Hand the files to the pipeline
- **LiDAR:** in Stray Scanner, open the recording list and export/share the recording, or use
  the iPhone **Files** app → *On My iPhone → Stray Scanner* and copy the recording folder.
  Copy it to the laptop (AirDrop, USB or cloud drive). The folder contains `rgb.mp4`,
  `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`.
- **Video:** copy the `.mp4`/`.mov` file unedited (Photos → Share → Save to Files / AirDrop).
- **Photos:** copy the original photos (no edits, no compression), **one folder per room**:
  `my_house/kitchen/*.HEIC`, `my_house/hall/*.HEIC`, ... (other files such as `.AAE` are ignored).

Then on the laptop, **one command**:
```
.venv/bin/python run.py <the recording folder | the video file | the folder of room folders>
```
Output: `outputs/<name>/plan.json`, `plan.svg`, `plan.png`.

*Note:* the Stray Scanner export menu wording may differ between app versions; the
Files-app route always works. (We could not test this page on an iPhone ourselves; see
`docs/TRADEOFFS.md`.)
