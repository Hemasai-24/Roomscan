# Device matrix

Which input tier runs on which phone, and the accuracy each tier honestly delivers.
Accuracy columns are filled only from measured benchmark runs (see `docs/benchmark_report.md`);
"not measured" means we have no ground-truth run for that cell yet.

| Tier | Phones | Capture tool | What the phone gives us | Expected accuracy (wall length) | Measured |
|---|---|---|---|---|---|
| LiDAR | iPhone 12 Pro / Pro Max and later **Pro** models (incl. 15 Pro, 16 Pro, 17 Pro) | Stray Scanner (free) | colour video + LiDAR depth (256x192) + ARKit camera poses + lens intrinsics | ~1-2 cm on walls seen well | consistency only (sample has no ground truth) |
| Video | any iPhone 15 or newer (Pro not needed) | Camera app | colour video only | gate is ±3% | **measured (Galaxy M53, tape): bedroom −31 %, bathroom +92 %; repeatability 0/3 walls, fails** |
| Photo | any iPhone 15 or newer | Camera app | 2-8 photos per room, EXIF focal length | gate is ±8% | **measured (Galaxy M53, tape): whole home +7.5 %; rooms −21 % … +42 %** |

Non-iPhone: our own benchmark photos/videos were taken with a **Samsung Galaxy M53** main
camera (no iPhone was available, see `docs/TRADEOFFS.md`). Nothing in the video/photo tiers
is iPhone-specific; the LiDAR tier requires an iPhone/iPad with LiDAR.

Laptop used for all runs: Linux, NVIDIA RTX 2000 Ada (8 GB), 20 CPU threads, 30 GB RAM.
LiDAR tier runs on CPU; video/photo tiers need a CUDA GPU with ≥ 8 GB for the 3D model.
