"""Photo tier, step P1: recognise a set of per-room photo folders, and load each room's photos upright at
the 3D model's input size, with the lens focal length from EXIF when the phone wrote it."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
FULL_FRAME_DIAGONAL_MM = 43.27     # 35 mm-equivalent focal lengths are defined on this diagonal
EXIF_IFD, FOCAL_35MM = 0x8769, 0xA405


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def is_photo_set(path):
    """A folder whose subfolders (>= 1) hold only images, and at least one image in total."""
    path = Path(path)
    if not path.is_dir():
        return False
    subs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subs:
        return False
    for s in subs:
        files = [p for p in s.iterdir() if p.is_file() and not p.name.startswith(".")]
        if any(p.suffix.lower() not in IMAGE_EXT for p in files):
            return False
    return any(_images(s) for s in subs)


def room_folders(path):
    return sorted((p for p in Path(path).iterdir() if p.is_dir() and _images(p)), key=lambda p: p.name)


def focal_px_from_35mm(f35, w, h):
    return float(f35) / FULL_FRAME_DIAGONAL_MM * float(np.hypot(w, h))


def load_room_images(folder, size=(392, 518)):
    """(images uint8 [N, H, W, 3] at size=(W, H), focal in pixels at that size or None, file names).
    EXIF orientation is applied; a photo still in landscape is turned to portrait (the protocol asks for
    portrait), so all photos share one size."""
    w, h = size
    imgs, focals, names = [], [], []
    for p in _images(folder):
        im = Image.open(p)
        f35 = im.getexif().get_ifd(EXIF_IFD).get(FOCAL_35MM)
        im = ImageOps.exif_transpose(im).convert("RGB")
        if im.width > im.height:
            im = im.rotate(90, expand=True)
        imgs.append(np.asarray(im.resize((w, h), Image.BILINEAR)))
        if f35:
            focals.append(focal_px_from_35mm(f35, w, h))
        names.append(p.name)
    fx = float(np.median(focals)) if focals else None
    return np.stack(imgs), fx, names
