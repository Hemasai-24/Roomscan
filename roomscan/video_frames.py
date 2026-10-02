"""Video tier, step V1: decode a walkthrough video exactly (no padded frames), turn it upright, and keep
the sharpest frame in every 1/per_sec window."""
import json
import subprocess

import cv2
import numpy as np

ROTATE_FILTERS = {0: None, 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}   # clockwise degrees


def video_info(path):
    """(number of decoded frames, average frames per second)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    n = int(info["streams"][0]["nb_read_packets"])
    dur = float(info["format"]["duration"])
    return n, n / dur


PORTRAIT, LANDSCAPE = (392, 518), (518, 392)    # (W, H) at the 3D model's input; multiples of 14 px


def display_size(path):
    """(width, height) as the video is shown, i.e. after the phone's rotation metadata."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True)    # works on ffprobe 4.x-7.x
    st = json.loads(out.stdout)["streams"][0]
    rot = int(float(st.get("tags", {}).get("rotate", 0) or 0))
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    w, h = int(st["width"]), int(st["height"])
    return (h, w) if rot % 180 else (w, h)


def model_size(path, rotate=0):
    """Model input size keeping the video's shape: landscape clips stay landscape (never squashed)."""
    w, h = display_size(path)
    if rotate % 180:
        w, h = h, w
    return LANDSCAPE if w > h else PORTRAIT


def read_frames(path, size, rotate=0):
    """Yield (index, RGB uint8 image of `size`=(w, h)) for every frame. `rotate` is applied before resizing;
    phone videos with rotation metadata are auto-rotated by ffmpeg anyway."""
    w, h = size
    filters = [f for f in (ROTATE_FILTERS[rotate], f"scale={w}:{h}") if f]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", ",".join(filters),
                          "-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def sharpness(img):
    """Variance of the Laplacian: high = lots of crisp edges, low = motion blur."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def pick_sharpest(path, per_sec=2.0, size=(392, 518), rotate=0):
    _, fps = video_info(path)
    win = max(1, int(round(fps / per_sec)))
    best = {}
    for i, img in read_frames(path, size, rotate):
        s = sharpness(img)
        k = i // win
        if k not in best or s > best[k][0]:
            best[k] = (s, i, img)
    picked = [best[k] for k in sorted(best)]
    return [i for _, i, _ in picked], np.stack([img for _, _, img in picked])
