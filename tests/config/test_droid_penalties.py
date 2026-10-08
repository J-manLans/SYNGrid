"""Covers DroidConf's penalty validation and the shipped spatial-scenario pair.

The two penalties are validated together and must both be 0 or negative. They are
also deliberately independent fields: the ratio between "broke a chain" and "ran
out of steps" is what shapes the incentive to finish rather than expire, and that
ratio used to be pinned at 1:1 because both names resolved to the same value.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from syn_grid.legacy.config_legacy.models import DroidConf

SHIPPED_CONFIG = "src/syn_grid/config/configs.yaml"

# The pair validated on seed 3 RPPO in the spatial scenario. Changing either
# invalidates the thesis figures, so they are pinned here rather than left to a
# sweep. See docs/dev/rppo-regression.md.
SPATIAL_CHAIN_BREAK = -0.01
SPATIAL_TIMEOUT = -1.0


def _droid_kwargs(**overrides) -> dict:
    kwargs = {
        "grid_rows": 5,
        "grid_cols": 5,
        "starting_score": 50,
        "step_penalty": -0.001,
        "boundary_penalty": -1.0,
        "chain_break_penalty": -0.01,
        "tier_consumption_penalty": -0.001,
        "reward_multiplier": 1.0,
        "timeout_penalty": -1.0,
    }
    kwargs.update(overrides)
    return kwargs


class TestDroidConfValidators:
    def test_valid_penalties_pass(self):
        conf = DroidConf(**_droid_kwargs())

        assert conf.timeout_penalty == -1.0

    @pytest.mark.parametrize(
        "field",
        [
            "step_penalty",
            "boundary_penalty",
            "chain_break_penalty",
            "tier_consumption_penalty",
            "timeout_penalty",
        ],
    )
    def test_positive_penalty_raises(self, field: str):
        with pytest.raises(ValidationError, match=field):
            DroidConf(**_droid_kwargs(**{field: 0.5}))

    @pytest.mark.parametrize("chain_break", [0.0, -0.01, -0.1])
    @pytest.mark.parametrize("timeout", [-0.1, -1.0])
    def test_the_two_penalties_are_independently_settable(
        self, chain_break: float, timeout: float
    ):
        """Guards two cheap things: both fields still exist, and neither is
        derived from or overwritten by the other.

        They used to be one field, which pinned their ratio at 1:1. No magnitude
        arithmetic is needed to guard that -- each value round-tripping unchanged
        is the whole property. The (-0.1, -0.1) case is the old forced-equality
        case and is included on purpose.
        """

        conf = DroidConf(
            **_droid_kwargs(chain_break_penalty=chain_break, timeout_penalty=timeout)
        )

        assert conf.chain_break_penalty == chain_break
        assert conf.timeout_penalty == timeout

    def test_zero_chain_break_leaves_the_timeout_untouched(self):
        """chain_break_penalty 0.0 was the thesis-era value; the timeout must
        still be applied in full rather than scaled away."""

        conf = DroidConf(**_droid_kwargs(chain_break_penalty=0.0, timeout_penalty=-1.0))

        assert conf.chain_break_penalty == 0.0
        assert conf.timeout_penalty == -1.0


class TestShippedSpatialConfig:
    @staticmethod
    def _raw() -> dict:
        return yaml.safe_load(Path(SHIPPED_CONFIG).read_text())

    def test_configs_yaml_pins_the_validated_pair(self):
        droid = self._raw()["world"]["droid_conf"]

        assert droid["chain_break_penalty"] == SPATIAL_CHAIN_BREAK
        assert droid["timeout_penalty"] == SPATIAL_TIMEOUT

    def test_step_penalty_anchor_is_not_reused_for_the_new_field(self):
        """AGENTS.md: breaking the &step_penalty alias matters because changing one
        penalty silently rescales the others. The new field must be a literal."""

        text = Path(SHIPPED_CONFIG).read_text()

        assert "timeout_penalty: *step_penalty" not in text
        assert self._raw()["world"]["droid_conf"]["timeout_penalty"] == SPATIAL_TIMEOUT
