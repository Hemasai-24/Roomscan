# Experiments (not part of the pipeline)

**YOLO11n-seg crack specialist** (Ultralytics 8.3.40, separate venv `.venv-yolo`):
`yolo segment train data=scripts/experiments/yolo_crack_data.yaml model=yolo11n-seg.pt epochs=15 imgsz=640 seed=0`
on the Ultralytics crack-seg dataset (3,717 train / 200 val), then `python scripts/experiments/yolo_crack_eval.py`.
Result: val mAP50 (box) 0.76 in-domain, but 1/4 public wall-crack photos found — no better than Grounding DINO —
so it was not adopted (domain gap: pavement/concrete cracks vs indoor walls). See `docs/TRADEOFFS.md`.
