import glob
from ultralytics import YOLO
m = YOLO("/home/hemasai/projects/floorplan-pipeline/outputs/yolo/crack/weights/best.pt")
crack_photos = [f"/home/hemasai/projects/floorplan-pipeline/data/public_damage/{n}" for n in
                ("01_crack_in_wall.jpg", "03_crack_in_wall.jpg", "04_crack_in_wall.jpg", "07_water_damage_ceiling.jpg")]
frames = sorted(glob.glob("/home/hemasai/projects/floorplan-pipeline/outputs/yolo/eval_frames/*.jpg"))
for conf in (0.25, 0.4, 0.5, 0.6):
    hits = sum(len(r.boxes) > 0 for r in m.predict(crack_photos, conf=conf, verbose=False, imgsz=640))
    fa = [len(r.boxes) for r in m.predict(frames, conf=conf, verbose=False, imgsz=640, stream=True)]
    print(f"yolo11n-seg crack conf {conf}: public crack photos found {hits}/4 | clean frames with a crack {sum(1 for x in fa if x)}/{len(fa)} ({sum(fa)} detections)", flush=True)
