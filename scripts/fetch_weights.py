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
VGGT_COMMIT = "main"
VGGT_WEIGHTS = ("facebook/VGGT-1B", "model.pt")
# Depth Anything V2, metric indoor (metres) - used only to fix VGGT's unknown scale.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
# Damage: Grounding DINO (open-vocabulary boxes from text) + SAM2.1 small (exact outline inside each box).
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-small"
HF_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.md"]


def main():
    WEIGHTS.mkdir(exist_ok=True)
    THIRD.mkdir(exist_ok=True)
    if not (THIRD / "vggt").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", VGGT_REPO, str(THIRD / "vggt")], check=True)
    print(hf_hub_download(*VGGT_WEIGHTS, local_dir=WEIGHTS / "vggt-1b"))
    print(snapshot_download(DEPTH_MODEL, local_dir=WEIGHTS / "da2-metric-indoor-large"))
    print(snapshot_download(GDINO_MODEL, local_dir=WEIGHTS / "gdino-base", allow_patterns=HF_PATTERNS))
    print(snapshot_download(SAM2_MODEL, local_dir=WEIGHTS / "sam2.1-small", allow_patterns=HF_PATTERNS))


if __name__ == "__main__":
    sys.exit(main())
