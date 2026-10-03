# How to capture your home

This takes about 10 minutes. Pick **one** of the three ways below. Each way gives you a floor plan; the first
is the most accurate.

| Way | Phone you need | App |
|---|---|---|
| A. Scan | iPhone 12 Pro or newer **Pro** model (it has a depth sensor) | Stray Scanner, free on the App Store |
| B. Video | any iPhone 15 or newer | the normal Camera app |
| C. Photos | any iPhone 15 or newer | the normal Camera app |

## Before you start

1. Turn on all the lights and open the curtains.
2. Open every door between the rooms.
3. Do not move furniture while you capture, and keep people out of the picture.

## A. Scan (Stray Scanner)

1. Install **Stray Scanner** from the App Store and allow it to use the camera.
2. Open the app and press the record button.
3. Hold the phone upright at chest height. Point it ahead and a little down, so you can see where the walls
   meet the floor.
4. Walk slowly, at half your normal speed, into every room.
5. In each room, turn slowly until you have seen every wall once. Then point up at the ceiling for a moment.
6. Walk through each door into the next room. Do not just look in from the doorway.
7. Finish where you started and press stop. Expect about 1 minute per room.

## B. Video (Camera app)

1. Open the Camera app and choose **Video**.
2. Use the normal lens (the "1x" button). Do not zoom and do not use "0.5x".
3. Press record and walk the home exactly as in steps 3 to 7 of the scan above, in one continuous video.

## C. Photos (Camera app)

1. Open the Camera app and choose **Photo**. Use the normal "1x" lens. Turn off **Live** (the round icon at
   the top). Do not use Portrait or Panorama.
2. Hold the phone upright for every photo.
3. **Corners:** in each room, stand in each corner and photograph the opposite corner, tilted a little down so
   the floor shows. That is 4 photos per room.
4. **Doors and windows:** one photo of each, taken straight on, with the whole door or window in the picture
   from bottom to top.
5. **Damage** (stains, cracks, mould): two photos of each, from two different spots.
6. **Between two rooms:** stand in the doorway and take one photo looking back into the room you are leaving.
   You will put this photo in the folders of **both** rooms. This is how the rooms are joined into one plan.
7. Keep each room's photos together, for example in one album per room. Aim for 2 to 8 photos per room.

## Things to avoid

- Turning quickly, or pointing only at the floor or only at the ceiling.
- Standing in front of a mirror. If there is a glass door, look past it rather than at it.
- Zoom, "0.5x", filters, Portrait, Panorama or Live photos.
- Editing or cropping the photos or video afterwards.

## Sending the files

- **Scan:** in Stray Scanner, open the list of recordings and share the recording. You can also open the
  **Files** app, go to *On My iPhone > Stray Scanner*, and copy the recording's folder.
- **Video:** in the Photos app, share the video unedited (AirDrop or Save to Files).
- **Photos:** send the original photos with one folder per room, named after the room, for example
  `kitchen`, `hall`, `bedroom`. Put the doorway photo from step 6 in both rooms' folders.

Send the files by AirDrop, USB cable or a cloud drive to the person who runs Roomscan.

---

*For the person running Roomscan:* run `.venv/bin/python run.py <recording folder | video file | folder of room
folders>`. The plan is written to `outputs/<name>/`. We could not test these steps on an iPhone ourselves
(no iPhone was available), and the wording of Stray Scanner's share menu may differ between versions; the
Files app route always works.
