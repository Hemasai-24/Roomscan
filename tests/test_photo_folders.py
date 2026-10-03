import numpy as np
from PIL import Image

from roomscan.photo_folders import focal_px_from_35mm, is_photo_set, load_room_images, room_folders
from roomscan.pipeline import detect_tier


def _jpg(path, w=300, h=400, f35=None, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8))
    ex = Image.Exif()
    if f35:
        ex.get_ifd(0x8769)[0xA405] = f35
    img.save(path, quality=90, exif=ex)


def _set(tmp_path):
    _jpg(tmp_path / "kitchen" / "a.jpg", f35=26)
    _jpg(tmp_path / "kitchen" / "b.jpg", f35=26, seed=1)
    _jpg(tmp_path / "bedroom" / "a.JPG", w=400, h=300, seed=2)
    return tmp_path


def test_photo_set_detected(tmp_path):
    assert is_photo_set(_set(tmp_path))
    assert detect_tier(tmp_path) == ("photo", tmp_path)


def test_iphone_side_files_are_ignored(tmp_path):
    _set(tmp_path)
    (tmp_path / "kitchen" / "IMG_0001.AAE").write_text("<plist/>")      # iPhone edit sidecar
    (tmp_path / "kitchen" / "IMG_0001.MOV").write_bytes(b"\0" * 10)    # Live Photo clip
    assert is_photo_set(tmp_path)
    imgs, _, names, _ = load_room_images(tmp_path / "kitchen")
    assert names == ["a.jpg", "b.jpg"]


def test_empty_folder_is_not_photo_set(tmp_path):
    assert not is_photo_set(tmp_path)


def test_flat_folder_of_photos_is_one_room(tmp_path):
    _jpg(tmp_path / "a.jpg")
    _jpg(tmp_path / "b.jpg", seed=1)
    assert is_photo_set(tmp_path)
    assert room_folders(tmp_path) == [tmp_path]


def test_heic_photos_load(tmp_path):
    import pillow_heif
    pillow_heif.register_heif_opener()
    (tmp_path / "hall").mkdir()
    for i in range(2):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (400, 300, 3), dtype=np.uint8)).save(
            tmp_path / "hall" / f"IMG_{i}.HEIC", format="HEIF")
    assert is_photo_set(tmp_path)
    imgs, _, names, _ = load_room_images(tmp_path / "hall")
    assert imgs.shape == (2, 518, 392, 3) and names == ["IMG_0.HEIC", "IMG_1.HEIC"]


def test_room_folders_sorted(tmp_path):
    assert [p.name for p in room_folders(_set(tmp_path))] == ["bedroom", "kitchen"]


def test_images_resized_and_focal_read(tmp_path):
    imgs, fx, names, dropped = load_room_images(_set(tmp_path) / "kitchen", size=(392, 518))
    assert dropped == []
    assert imgs.shape == (2, 518, 392, 3) and imgs.dtype == np.uint8
    np.testing.assert_allclose(fx, focal_px_from_35mm(26, 300, 400) * 392 / 300)   # 300x400 -> crop 300x396 -> 392x518
    assert names == ["a.jpg", "b.jpg"]


def test_upright_landscape_photos_stay_landscape_and_no_exif_gives_none(tmp_path):
    # rotating an upright landscape photo would make a wall look like the floor
    imgs, fx, _, _ = load_room_images(_set(tmp_path) / "bedroom", size=(392, 518))
    assert imgs.shape == (1, 392, 518, 3)
    assert fx is None


def test_mixed_orientation_keeps_the_majority(tmp_path):
    for i in range(3):
        _jpg(tmp_path / "den" / f"p{i}.jpg", seed=i)
    _jpg(tmp_path / "den" / "l0.jpg", w=400, h=300, seed=9)
    imgs, _, names, dropped = load_room_images(tmp_path / "den")
    assert imgs.shape == (3, 518, 392, 3) and dropped == ["l0.jpg"]


def test_focal_from_35mm():
    # 35 mm-equivalent focal is defined on the 43.27 mm full-frame diagonal
    np.testing.assert_allclose(focal_px_from_35mm(43.27, 300, 400), 500.0)


def test_different_aspect_photo_is_centre_cropped_not_stretched(tmp_path):
    # a 9:16 photo among 3:4 photos must keep its geometry (crop to 3:4), not be squashed
    _jpg(tmp_path / "bed" / "a.jpg", w=300, h=400)
    img = np.zeros((1600, 900, 3), np.uint8)
    img[700:900, 350:550] = 255                            # 200 x 200 px white square at the centre
    Image.fromarray(img).save(tmp_path / "bed" / "b.jpg", quality=95)
    imgs, fx, names, _ = load_room_images(tmp_path / "bed", size=(392, 518))
    b = imgs[names.index("b.jpg")][:, :, 0] > 128
    ys, xs = np.nonzero(b)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert abs(w - h) <= 2                                 # still a square (a stretch gives ~87 x 65)
