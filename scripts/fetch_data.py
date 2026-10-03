"""Download our own raw benchmark data (Samsung S23 photos/videos) into data/raw/s23/.

usage: python scripts/fetch_data.py
The archive is too large for git; it is hosted at the URL in data/DATA_URL (one line). The tape measurements
(data/ground_truth/) are in git. The provided Stray Scanner sample data is not redistributed: place it in
sample_data/ as received."""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    url_file = ROOT / "data" / "DATA_URL"
    if not url_file.exists() or not url_file.read_text().strip():
        sys.exit("data/DATA_URL is empty: the raw-data archive link has not been published yet")
    url = url_file.read_text().strip()
    dest = ROOT / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "s23.zip"
    print(f"downloading {url} ...")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    zpath.unlink()
    print(f"extracted to {dest}")


if __name__ == "__main__":
    main()
