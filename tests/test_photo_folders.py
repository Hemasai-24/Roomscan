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


def test_folder_with_non_images_is_not_photo_set(tmp_path):
    _set(tmp_path)
    (tmp_path / "kitchen" / "notes.txt").write_text("x")
    assert not is_photo_set(tmp_path)


def test_empty_or_flat_folder_is_not_photo_set(tmp_path):
    assert not is_photo_set(tmp_path)
    _jpg(tmp_path / "a.jpg")
    assert not is_photo_set(tmp_path)


def test_room_folders_sorted(tmp_path):
    assert [p.name for p in room_folders(_set(tmp_path))] == ["bedroom", "kitchen"]


def test_images_resized_and_focal_read(tmp_path):
    imgs, fx, names = load_room_images(_set(tmp_path) / "kitchen", size=(392, 518))
    assert imgs.shape == (2, 518, 392, 3) and imgs.dtype == np.uint8
    np.testing.assert_allclose(fx, focal_px_from_35mm(26, 392, 518))
    assert names == ["a.jpg", "b.jpg"]


def test_landscape_photo_turned_upright_and_no_exif_gives_none(tmp_path):
    imgs, fx, _ = load_room_images(_set(tmp_path) / "bedroom", size=(392, 518))
    assert imgs.shape == (1, 518, 392, 3)
    assert fx is None


def test_focal_from_35mm():
    # 35 mm-equivalent focal is defined on the 43.27 mm full-frame diagonal
    np.testing.assert_allclose(focal_px_from_35mm(43.27, 300, 400), 500.0)
