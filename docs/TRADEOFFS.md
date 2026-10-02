# Trade-offs and limitations

Read this first: it lists every place this submission departs from the ideal case in the
brief, why, and what that does to the reported numbers. Kept up to date as work proceeds.

## Constraints we worked under
| Constraint | Consequence |
|---|---|
| No iPhone available (author's phone is a Samsung Galaxy S23, no LiDAR) | Could not record our own LiDAR captures, a Polycam LiDAR export, or iPhone photo/video |
| No ground truth exists for the provided sample data (confirmed by the hiring team, 2026-10-02) | Sample-data numbers are consistency checks, not accuracy |
| ~42 hours from brief to submission | Scope triaged by score weight; see "Not done" |

## What we did instead
| Brief requirement | What we did | Effect on the numbers |
|---|---|---|
| Capture route | Route 2: Stray Scanner (free; same format as the sample data) + one-page protocol | None; the protocol was not tested by us on an iPhone |
| LiDAR-tier accuracy vs laser ground truth | Planned: ARKitScenes scenes (iPad Pro LiDAR, laser-scanned reference) | iPad Pro, not iPhone; disclosed dataset |
| Video and photo tiers on the sample apartment | Simulated from the sample `rgb.mp4` (video: RGB only, no depth/poses; photos: 2-8 sharp frames per room) | Video frames are sharper and more evenly spaced than a typical handheld clip; photos are video frames, not stills |
| Our own photo/video benchmark with tape ground truth | Planned: author's rooms, Samsung S23 main camera (1x), tape measure | Android camera instead of iPhone 15; tape ±2-3 mm, not laser |
| Repeatability (same room twice, same tier) | The two whole-floor sample scans of the same apartment | Different walk paths, so this is a harder test than two identical captures |

## Not done (and why)
| Requirement | Status | Why |
|---|---|---|
| Staged damage room captured at all tiers | Pending | Needs a physical room + phone; S23 attempt planned |
| Head-to-head vs Polycam at the LiDAR tier | Pending | Needs a LiDAR iPhone |

## Engineering decisions with known costs
- **Own RANSAC instead of Open3D's.** Open3D's threaded `segment_plane` gave different planes on
  repeated runs of the same input despite a fixed seed. We use a seeded single-threaded NumPy
  RANSAC so the same capture always yields the same plan (repeatability gate). Cost: slower.
- **Manhattan (right-angle) room footprints.** Wall outlines are snapped to two perpendicular
  directions. Rooms with angled walls will be squared off.
