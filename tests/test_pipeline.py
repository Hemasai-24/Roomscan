import json
import subprocess
import sys
from pathlib import Path

import pytest
from roomscan.save_plan import build_plan, validate
from tests.conftest import SINGLE_ROOM, need_sample

ROOT = Path(__file__).resolve().parents[1]


def test_empty_plan_validates():
    validate(build_plan("x", "lidar", [], {}))


@pytest.mark.sample
def test_run_single_room_end_to_end(tmp_path):
    need_sample()
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), str(SINGLE_ROOM), "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    plan = json.loads((tmp_path / "plan.json").read_text())
    validate(plan)
    room = plan["rooms"][0]
    assert 2.0 < room["ceiling_height"]["value"] < 3.2
    assert 3.0 < room["floor_area"]["value"] < 30.0
    assert len(room["walls"]) >= 4
    assert (tmp_path / "plan.svg").stat().st_size > 1000


@pytest.mark.sample
def test_deterministic_output(tmp_path):
    need_sample()
    outs = []
    for k in range(2):
        d = tmp_path / str(k)
        subprocess.run([sys.executable, str(ROOT / "run.py"), str(SINGLE_ROOM), "--out", str(d)], check=True,
                       capture_output=True, timeout=600)
        p = json.loads((d / "plan.json").read_text())
        p["meta"].pop("runtime_s", None)
        outs.append(p)
    assert outs[0] == outs[1]
