"""Photo tier, step P1: recognise a set of per-room photo folders, and load each room's photos upright at
the 3D model's input size, with the lens focal length from EXIF when the phone wrote it."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:                                    # iPhone default "High Efficiency" photos are HEIC
    import pillow_heif
    pillow_heif.register_heif_opener()
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
except ImportError:                     # pragma: no cover - pillow-heif is in requirements-models.txt
    IMAGE_EXT = {".jpg", ".jpeg", ".png"}
FULL_FRAME_DIAGONAL_MM = 43.27     # 35 mm-equivalent focal lengths are defined on this diagonal
EXIF_IFD, FOCAL_35MM = 0x8769, 0xA405


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def _subfolders(path):
    return [p for p in Path(path).iterdir() if p.is_dir() and not p.name.startswith(".")]


def is_photo_set(path):
    """One subfolder of photos per room, or a single folder of photos (one room). Other files
    (iPhone .AAE edit sidecars, Live Photo .MOV clips, notes) are ignored."""
    path = Path(path)
    if not path.is_dir():
        return False
    return bool(room_folders(path))


def room_folders(path):
    path = Path(path)
    subs = sorted((p for p in _subfolders(path) if _images(p)), key=lambda p: p.name)
    if subs:
        return subs
    return [path] if _images(path) else []


def focal_px_from_35mm(f35, w, h):
    return float(f35) / FULL_FRAME_DIAGONAL_MM * float(np.hypot(w, h))


def load_room_images(folder, size=(392, 518)):
    """(images uint8 [N, H, W, 3], focal in pixels at that size or None, file names, dropped names).
    EXIF orientation is applied and photos are never rotated further: turning an upright landscape photo
    would make the model see a wall as the floor. The room uses the majority orientation (`size` is the
    portrait (W, H); landscape rooms use (H, W)); photos in the other orientation are dropped."""
    loaded = []
    for p in _images(folder):
        im = Image.open(p)
        f35 = im.getexif().get_ifd(EXIF_IFD).get(FOCAL_35MM)
        im = ImageOps.exif_transpose(im).convert("RGB")
        loaded.append((p.name, im, f35))
    landscape = [im.width > im.height for _, im, _ in loaded]
    use_landscape = sum(landscape) > len(loaded) / 2
    w, h = (size[1], size[0]) if use_landscape else size
    imgs, focals, names, dropped = [], [], [], []
    for (name, im, f35), land in zip(loaded, landscape):
        if land != use_landscape:
            dropped.append(name)
            continue
        W0, H0 = im.size
        cw, ch = (W0, round(W0 * h / w)) if W0 * h / w <= H0 else (round(H0 * w / h), H0)
        x0, y0 = (W0 - cw) // 2, (H0 - ch) // 2                 # centre-crop to the model's shape: never stretch
        imgs.append(np.asarray(im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)))
        if f35:   # focal in original pixels (35 mm-equivalent is defined on the full frame), then rescaled
            focals.append(focal_px_from_35mm(f35, W0, H0) * w / cw)
        names.append(name)
    fx = float(np.median(focals)) if focals else None
    return np.stack(imgs), fx, names, dropped
