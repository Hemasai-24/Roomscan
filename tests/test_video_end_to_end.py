"""Video tier end to end on the sample video (model test: needs GPU weights)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from roomscan.save_plan import validate
from tests.test_video_poses_model import ROOT, SINGLE

pytestmark = pytest.mark.model


def test_run_py_on_sample_video(tmp_path):
    if SINGLE is None or not (ROOT / "weights" / "vggt-1b" / "model.pt").exists():
        pytest.skip("needs sample data and weights")
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), str(SINGLE / "rgb.mp4"), "--rotate", "90",
                        "--out", str(tmp_path)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    plan = json.loads((tmp_path / "plan.json").read_text())
    validate(plan)
    assert plan["capture"]["tier"] == "video"
    assert len(plan["rooms"]) >= 1
