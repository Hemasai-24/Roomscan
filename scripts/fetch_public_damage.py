"""Download the public damage photos (Wikimedia Commons, open licences) listed in data/public_damage/credits.txt.

usage: python scripts/fetch_public_damage.py   then   python scripts/eval_public_damage.py [detector]"""
import urllib.request
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "public_damage"


def main():
    for line in (DIR / "credits.txt").read_text().splitlines():
        name, cls, lic, title, url = line.split("|")
        if not (DIR / name).exists():
            req = urllib.request.Request(url, headers={"User-Agent": "roomscan-eval/1.0"})
            (DIR / name).write_bytes(urllib.request.urlopen(req, timeout=60).read())
        print(f"{name}  ({lic})  {title}")


if __name__ == "__main__":
    main()
