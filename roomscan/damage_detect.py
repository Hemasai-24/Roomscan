"""Damage detection on one colour image: open-vocabulary boxes (Grounding DINO) -> exact outlines (SAM2).

Behind a small registry so another detector (e.g. OWLv2) can be swapped in. The post-processing
(score threshold, label -> class, size limits, de-duplication, clipping the mask to its box) is plain code
and tested without a GPU."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"water stain": "water_stain", "water damage": "water_stain", "peeling paint": "peeling_paint",
           "crack": "crack", "mold": "mold", "hole": "hole"}
PROMPT = "water stain. crack. mold. peeling paint. hole."
THRESHOLD = 0.35
MIN_FRAC = 0.0005       # mask smaller than this share of the image: noise
MAX_FRAC = 0.25         # larger: a whole wall / floor, not a defect
NMS_IOU = 0.6
BOX_MARGIN = 2          # px a mask may extend past its box


def label_to_class(text):
    text = text.lower()
    for key in sorted(CLASSES, key=len, reverse=True):
        if key in text:
            return CLASSES[key]
    return None


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def postprocess(raw, image_shape, threshold=THRESHOLD):
    """raw: [{"label": text, "score", "box": (x0,y0,x1,y1), "mask": bool(H,W)}] -> kept detections."""
    h, w = image_shape[:2]
    cands = []
    for d in raw:
        cls = label_to_class(d["label"])
        if cls is None or d["score"] < threshold:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        keep = np.zeros((h, w), bool)
        keep[max(0, y0 - BOX_MARGIN):y1 + BOX_MARGIN, max(0, x0 - BOX_MARGIN):x1 + BOX_MARGIN] = True
        mask = np.asarray(d["mask"], bool) & keep
        frac = mask.sum() / (h * w)
        if frac < MIN_FRAC or frac > MAX_FRAC:
            continue
        cands.append({"class": cls, "score": float(d["score"]), "box": (x0, y0, x1, y1), "mask": mask})
    cands.sort(key=lambda c: (-c["score"], c["box"]))
    out = []
    for c in cands:
        if all(o["class"] != c["class"] or _iou(o["box"], c["box"]) < NMS_IOU for o in out):
            out.append(c)
    return out


def upright_turns(T_wc):
    """Number of 90-degree counter-clockwise image turns (np.rot90 k) that make the image upright,
    from where world-up appears in the camera (camera axes: x right, y down)."""
    up = T_wc[:3, :3].T @ np.array([0.0, 1.0, 0.0])
    v = np.array([up[0], up[1]])
    best, best_k = None, 0
    for k in range(4):
        score = -v[1]                       # pointing to the image top
        if best is None or score > best + 1e-9:
            best, best_k = score, k
        v = np.array([v[1], -v[0]])         # effect of one CCW image turn on a direction
    return best_k


class GDinoSam2:
    """Grounding DINO (base) boxes from the text prompt, then SAM2.1 (small) masks inside each box."""

    def __init__(self, root=ROOT, device="cuda"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.torch, self.device = torch, device
        w = Path(root) / "weights"
        self.gp = AutoProcessor.from_pretrained(w / "gdino-base")
        self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(w / "gdino-base").to(device).eval()
        self.sp = Sam2Processor.from_pretrained(w / "sam2.1-small")
        self.sm = Sam2Model.from_pretrained(w / "sam2.1-small").to(device).eval()

    def __call__(self, image, box_threshold=0.2):
        torch = self.torch
        with torch.no_grad():
            inp = self.gp(images=image, text=PROMPT, return_tensors="pt").to(self.device)
            res = self.gp.post_process_grounded_object_detection(
                self.gm(**inp), inp.input_ids, threshold=box_threshold, text_threshold=box_threshold,
                target_sizes=[image.shape[:2]])[0]
            boxes = [b.tolist() for b in res["boxes"]]
            if not boxes:
                return []
            si = self.sp(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            so = self.sm(**si)
            masks = self.sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0]
            best = so.iou_scores[0].argmax(-1).cpu()
        labels = res.get("text_labels", res["labels"])
        return [{"label": str(l), "score": float(s), "box": tuple(b),
                 "mask": masks[i, int(best[i])].numpy().astype(bool)}
                for i, (l, s, b) in enumerate(zip(labels, res["scores"].tolist(), boxes))]

    def boxes(self, image, prompt, threshold=0.3):
        """Boxes only (no masks) for a free-text prompt, e.g. "door. window."."""
        torch = self.torch
        with torch.no_grad():
            inp = self.gp(images=image, text=prompt, return_tensors="pt").to(self.device)
            res = self.gp.post_process_grounded_object_detection(
                self.gm(**inp), inp.input_ids, threshold=threshold, text_threshold=threshold,
                target_sizes=[image.shape[:2]])[0]
        labels = res.get("text_labels", res["labels"])
        return [{"label": str(l), "score": float(sc), "box": tuple(b.tolist())}
                for l, sc, b in zip(labels, res["scores"].tolist(), res["boxes"])]

    def close(self):
        del self.gm, self.sm
        self.torch.cuda.empty_cache()


OWL_PHRASES = ["a crack in a wall", "a water stain", "water damage", "mold", "peeling paint", "a hole in a wall"]


class Owlv2Sam2(GDinoSam2):
    """OWLv2 (base, ensemble) boxes from text phrases, then SAM2.1 masks (same SAM as GDinoSam2)."""

    def __init__(self, root=ROOT, device="cuda"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor
        self.torch, self.device = torch, device
        w = Path(root) / "weights"
        self.op = Owlv2Processor.from_pretrained(w / "owlv2-base")
        self.om = Owlv2ForObjectDetection.from_pretrained(w / "owlv2-base").to(device).eval()
        self.sp = Sam2Processor.from_pretrained(w / "sam2.1-small")
        self.sm = Sam2Model.from_pretrained(w / "sam2.1-small").to(device).eval()
        self.gm = self.om                         # so close() frees it

    def __call__(self, image, box_threshold=0.1):
        torch = self.torch
        h, w = image.shape[:2]
        with torch.no_grad():
            inp = self.op(text=[OWL_PHRASES], images=image, return_tensors="pt").to(self.device)
            out = self.om(**inp)
            side = max(h, w)                      # OWLv2 pads to a square
            res = self.op.post_process_object_detection(out, threshold=box_threshold,
                                                        target_sizes=torch.tensor([[side, side]]).to(self.device))[0]
            boxes = [[max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]
                     for b in res["boxes"].tolist()]
            if not boxes:
                return []
            si = self.sp(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            so = self.sm(**si)
            masks = self.sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0]
            best = so.iou_scores[0].argmax(-1).cpu()
        return [{"label": OWL_PHRASES[int(l)], "score": float(sc), "box": tuple(b),
                 "mask": masks[i, int(best[i])].numpy().astype(bool)}
                for i, (l, sc, b) in enumerate(zip(res["labels"].tolist(), res["scores"].tolist(), boxes))]


DETECTORS = {"gdino+sam2": GDinoSam2, "owlv2+sam2": Owlv2Sam2}
THRESHOLDS = {"gdino+sam2": THRESHOLD, "owlv2+sam2": 0.2}   # OWLv2 scores run lower
DAMAGE_DETECTOR = os.environ.get("ROOMSCAN_DAMAGE_DETECTOR", "gdino+sam2")
_LOADED = {}


def get_detector(name="gdino+sam2", root=ROOT):
    if name not in _LOADED:
        _LOADED[name] = DETECTORS[name](root)
    return _LOADED[name]


def release_detectors():
    for d in _LOADED.values():
        d.close()
    _LOADED.clear()


def detect_damage(image_rgb, detector=None, threshold=None, root=ROOT):
    """image_rgb: upright uint8 (H,W,3) -> [{"class", "score", "mask", "box"}]."""
    detector = detector or DAMAGE_DETECTOR
    threshold = THRESHOLDS[detector] if threshold is None else threshold
    return postprocess(get_detector(detector, root)(np.ascontiguousarray(image_rgb)), image_rgb.shape, threshold)


OPENING_PROMPT = "door. window."
OPENING_THRESHOLD = 0.4


def detect_openings(image_rgb, detector="gdino+sam2", root=ROOT):
    return get_detector(detector, root).boxes(np.ascontiguousarray(image_rgb), OPENING_PROMPT, OPENING_THRESHOLD)
