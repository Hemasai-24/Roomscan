"""Download model weights and third-party code used by the video and photo tiers.

Usage: python scripts/fetch_weights.py        (idempotent; skips what is already there)
Weights land in weights/ and code in third_party/, both gitignored.
"""
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
THIRD = ROOT / "third_party"

# VGGT (Meta, 2025): camera poses + depth from many images in one forward pass.
VGGT_REPO = "https://github.com/facebookresearch/vggt.git"
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"  # pinned: the commit every reported number used
VGGT_WEIGHTS = ("facebook/VGGT-1B", "model.pt")
# Depth Anything V2, metric indoor (metres) - used only to fix VGGT's unknown scale.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
# Damage: Grounding DINO (open-vocabulary boxes from text) + SAM2.1 small (exact outline inside each box).
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-small"
HF_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.md"]


# Optional (--unidepth): UniDepth V2 ViT-S, focal-aware metric depth (ROOMSCAN_METRIC_DEPTH=unidepth).
# Code: CC BY-NC 4.0, pinned commit; weights from Hugging Face.
UNIDEPTH_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
UNIDEPTH_MODEL = "lpiccinelli/unidepth-v2-vits14"


def fetch_unidepth():
    import io
    import tarfile
    import urllib.request
    dest = THIRD / "UniDepth"
    if not dest.exists():
        url = f"https://codeload.github.com/lpiccinelli-eth/UniDepth/tar.gz/{UNIDEPTH_COMMIT}"
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz") as t:
            t.extractall(THIRD)
        (THIRD / f"UniDepth-{UNIDEPTH_COMMIT}").rename(dest)
    print(snapshot_download(UNIDEPTH_MODEL, local_dir=WEIGHTS / "unidepth-v2-vits14"))


def main():
    WEIGHTS.mkdir(exist_ok=True)
    THIRD.mkdir(exist_ok=True)
    if not (THIRD / "vggt").exists():
        subprocess.run(["git", "clone", "-q", VGGT_REPO, str(THIRD / "vggt")], check=True)
        subprocess.run(["git", "-C", str(THIRD / "vggt"), "checkout", "-q", VGGT_COMMIT], check=True)
    print(hf_hub_download(*VGGT_WEIGHTS, local_dir=WEIGHTS / "vggt-1b"))
    print(snapshot_download(DEPTH_MODEL, local_dir=WEIGHTS / "da2-metric-indoor-large"))
    print(snapshot_download(GDINO_MODEL, local_dir=WEIGHTS / "gdino-base", allow_patterns=HF_PATTERNS))
    print(snapshot_download(SAM2_MODEL, local_dir=WEIGHTS / "sam2.1-small", allow_patterns=HF_PATTERNS))
    if "--unidepth" in sys.argv:
        fetch_unidepth()


if __name__ == "__main__":
    sys.exit(main())
