# Experiment: focal-aware metric depth (UniDepth V2) for the video/photo scale

**Why:** on our tape-measured home (Galaxy M53) the kitchen and bathroom photo rooms came out ~13-20 %
too large linearly. Depth Anything V2 Metric-Indoor reads depth ~41 % too far and was bias-calibrated on the
sample iPhone; it does not use the camera's focal length.

**Candidate:** UniDepth V2 ViT-S (`lpiccinelli/unidepth-v2-vits14`, code `lpiccinelli-eth/UniDepth` @
`8d8cfe4`, **CC BY-NC 4.0**). Takes the camera intrinsics (EXIF focal for photos) or estimates them. Runs from
source without installing anything into the main environment (0.4 GB GPU). Option:
`ROOMSCAN_METRIC_DEPTH=unidepth` (default stays `depth_anything`); fetch with
`python scripts/fetch_weights.py --unidepth`.

## 1. Scale vs LiDAR on the sample captures (no calibration for either model)
Median over frames of median(metric depth / LiDAR depth); 185 frames over 3 captures.

| Model | Ratio | Per-frame std |
|---|---|---|
| Depth Anything V2 Metric-Indoor (raw) | 1.425 | 0.300 |
| Depth Anything after our bias 1.408 | ≈ 1.01 | ≈ 0.21 (relative) |
| **UniDepth V2, true focal** | **0.998** | **0.111** |
| UniDepth V2, its own focal estimate | 1.008 | 0.109 |

UniDepth needs no calibration and has about half the scatter of calibrated Depth Anything.

## 2. Our home, photo tier (protocol photos `data/raw/m53_doorway_shared`; nothing fitted on this data)

| Room (tape area) | Depth Anything | UniDepth |
|---|---|---|
| bedroom (8.37 m²) | 6.62 (−21 %) | 9.68 (+16 %) |
| hall (15.56 m²) | 17.08 (+10 %) | 25.04 (+61 %) |
| kitchen (2.86 m²) | 4.07 (+42 %) | 3.77 (+32 %) |
| bathroom (2.87 m²) | 3.65 (+27 %) | **2.71 (−5 %)** |
| **total (29.66 m²)** | **31.41 (+6 %)** | **41.21 (+39 %)** |

Per-room metric scale went down 7-14 % with UniDepth, as the overscale diagnosis predicted, and the bathroom
lands inside ±8 %. But the room **shapes** changed too (hall box 7.1 × 4.6 → 8.4 × 6.7 m; bedroom 3.6 × 2.6 →
5.3 × 3.4 m): the plane finder and the wall-bounding step use thresholds in metres, so a different scale
changes which walls are found and where a room is cut. The room-shape step from few photos is unstable,
and it dominates the result.

## Recommendation
**Not made the default.** It is clearly better on the sample (scale), but not on our home end to end (total
+6 % → +39 %). The scale fix is real; adopting it needs the room-shape step made scale-robust first (thresholds
relative to room size, or the wall-bound rule from the fix loop re-validated at the new scale).
