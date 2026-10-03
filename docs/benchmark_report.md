# Benchmark report

## What was measured

| Capture | Phone | Tiers | Ground truth |
|---|---|---|---|
| Our home: bedroom, hall, kitchen, bathroom (the hall connects all three) | Samsung Galaxy M53 | photo (24 photos), video (4 clips) | tape: every wall, door, window and the ceiling |
| Sample data from the brief: 3 captures of one flat | iPhone Pro | LiDAR; video and photo simulated from it | none; LiDAR is used as the reference |

We had no LiDAR iPhone, so there is no LiDAR result on our home and no comparison with Polycam.

To regenerate the numbers, run `bash scripts/reproduce_all.sh`. Saved results are in `results/`.

## 1. Photo tier on our home

Photos taken as the capture protocol asks (each doorway photo in both rooms' folders), with the current code.
Saved result: `results/m53_benchmark.json`, key `photo_protocol`.

| Room | Tape | Photo tier | Error | Tape inside 95 % range |
|---|---|---|---|---|
| bedroom | 8.37 m² | 6.62 m² | −21 % | yes |
| hall | 15.56 m² | 17.08 m² | +10 % | yes |
| kitchen | 2.86 m² | 4.07 m² | +42 % | yes |
| bathroom | 2.87 m² | 3.65 m² | +27 % | yes |
| whole home | 29.65 m² | 31.41 m² | +6 % | - |

| Gate | Result | Pass |
|---|---|---|
| Whole-home footprint within ±8 % | +6 % | yes |
| All rooms placed, correct connections, no overlap | 3 of 3 connections, largest overlap 0.06 m² | yes |
| Wall lengths within ±8 % | 2 of 16 walls; mean error 74 cm | no |
| Door and window widths within 2 cm on 85 % | 0 of 9, plus 6 false openings | no |
| Ceiling height within 1.5 cm | ceiling not in the photos; values 3.3-4.3 m against 2.74 m | no |
| Truth inside the 95 % range | 4 of 4 areas, 15 of 16 walls, but the ranges are very wide | yes |

The footprint passes partly because errors cancel: the bedroom is too small and the other rooms too big. The
summed per-room error is about 19 %. Without the doorway photo in both folders, only 1 of 3 connections is
found.

## 2. Video tier on our home

| Clip | Result | Tape |
|---|---|---|
| walkthrough, 68 s | 2 rooms, 21.1 m² | 29.65 m² for the whole home |
| bedroom, take 1, 21 s | 5.76 m² (−31 %) | 8.37 m² |
| bedroom, take 2, 14 s | split into 3 pieces, 9.1 m² | 8.37 m² |
| bathroom, low light, 9 s | 5.50 m² (+92 %) | 2.87 m² |

The video gate (walls within ±3 %) fails. Repeatability also fails: the bedroom's walls differ by 61, 99 and
113 cm between the two takes. The ceiling was not filmed; the true 2.74 m is inside every reported range, but
the ranges are wide. The cause is camera tracking: on the sample, giving the video tier the true camera poses
brings the footprint to +3.6 %.

## 3. Damage on our home

| Item | Result |
|---|---|
| Staged water stain, 9 × 6 cm | found and measured at 16 × 24 cm, but it was in only one photo, so the two-view rule drops it |
| Staged crack, 8 cm | not found |
| False alarms, photo tier, whole home | 12 before the damage rules, 0 after |
| Public damage photos (10) | 5 found with Grounding DINO, 4 with OWLv2 |

## 4. LiDAR tier on the sample (no ground truth)

| Capture | Rooms | Connections | Runtime |
|---|---|---|---|
| single_room | 2 | 1 | 20 s |
| floor_only | 13 | 9 | 56 s |
| with_ceiling | 9 | 3 | 82 s |

- Repeatability: the two whole-floor scans are of the same flat. 0 of 14 matched walls agree within 1 cm.
- Ceilings observed in with_ceiling: 2.34-3.21 m.
- Damage false alarms on the undamaged flat: 0, 2 and 3 regions.
- Drift ablation: see section 5 of the technical report.

## 5. Video and photo against LiDAR on the sample

| Test | Result |
|---|---|
| Video, floor_only | covers 5.1 of 41.0 m² |
| Video with the true camera poses | 42.5 m² (+3.6 %); poses are the main video error |
| Photo, floor_only | footprint −12 % |
| Photo with the true room positions | within ±8 %; room placement is the main photo error |

## 6. Comparison with a consumer app

Not done. It needs a LiDAR iPhone to run both our LiDAR tier and Polycam.

## 7. Timing

Laptop with an RTX 2000 Ada (8 GB) and 20 CPU threads.

| Run | Time |
|---|---|
| LiDAR, one room / whole floor | 20 s / 56-82 s, plus about 80 s for damage |
| Photo, 4 rooms, 24 photos, with damage | 113 s |
| Video, 68 s walkthrough | 35 s back end plus 102 s damage; VGGT takes 5.4 s per 20 frames |
| Full reproduction, CPU part | about 17 min |
