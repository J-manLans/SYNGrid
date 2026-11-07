import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Fixtures      #
# ================= #

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "scripts" / "replay" / "probe.py"
BASELINES = Path(__file__).parent / "baselines"
LOCKED = [name for name, entry in SCENARIOS.items() if entry.lock is not None]


def _record(name: ScenarioName, out: Path) -> dict:
    # PYTHONPATH keeps the probe on this tree, wherever the editable install points
    result = subprocess.run(
        [sys.executable, str(PROBE), "--scenario", name.value, "--label", "now"]
        + ["--out", str(out)],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    return json.loads((out / "now.json").read_text())


# ================= #
#      Tests        #
# ================= #


class TestReplay:
    """A locked scenario's recording against its baseline."""

    @pytest.mark.parametrize("name", LOCKED, ids=lambda name: name.value)
    def test_the_tape_plays_as_the_baseline_recorded_it(
        self, name: ScenarioName, tmp_path: Path
    ):
        baseline_path = BASELINES / f"{name.value}.json"
        assert baseline_path.exists(), (
            f"{name.value} is locked and has no baseline. Record it with: "
            f"PYTHONPATH=src python scripts/replay/probe.py --baseline --scenario {name.value}"
        )
        baseline = json.loads(baseline_path.read_text())

        now = _record(name, tmp_path)

        assert now["stream_digest"] == baseline["stream_digest"], (
            f"{name.value} no longer plays as its baseline. "
            f"The config {'is the same' if now['config_digest'] == baseline['config_digest'] else 'changed'}. "
            f"See where with: python scripts/replay/diff.py "
            f"{baseline_path.with_suffix('.npz')} {tmp_path / 'now.npz'}"
        )
