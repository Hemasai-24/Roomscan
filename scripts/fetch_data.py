"""Download our own raw benchmark data (Samsung Galaxy M53 photos/videos) into data/raw/.

usage: python scripts/fetch_data.py
The photos are in git (data/raw/m53, data/raw/m53_doors). The videos are too large for git (one is 140 MB); they
are in a Google Drive folder whose link is in data/DATA_URL. Downloaded with
gdown (pip install gdown==5.2.0). The tape measurements (data/ground_truth/) are in git. The provided Stray
Scanner sample data is not redistributed: place it in sample_data/ as received.
If gdown is blocked, open the link in a browser, download the folder and unzip it into data/raw/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    url_file = ROOT / "data" / "DATA_URL"
    url = url_file.read_text().strip() if url_file.exists() else ""
    if not url:
        sys.exit("data/DATA_URL is empty")
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown==5.2.0   (or download the folder in a browser: " + url + ")")
    dest = ROOT / "data" / "raw" / "m53"          # the Drive folder holds the room folders and video/
    dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url, output=str(dest), quiet=False, remaining_ok=True)
    print(f"downloaded into {dest}")


if __name__ == "__main__":
    main()
