from typing import Any

import pytest

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Fixtures      #
# ================= #

SPATIAL = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
DELAY = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY


def _run_conf(name: ScenarioName) -> ScenarioConf:
    manager = ConfigManager()
    return SCENARIOS[name].full_conf(
        manager.load_raw(manager.scenario_config_path(name))
    )


# ================= #
#      Tests        #
# ================= #


class TestBuild:
    """The checks an entry makes before it composes a scenario."""

    def test_a_config_of_another_class_is_rejected(self):
        with pytest.raises(TypeError, match="configured by"):
            SCENARIOS[DELAY].build(DELAY.value, _run_conf(SPATIAL))

    def test_a_perception_without_the_scenarios_visibility_is_rejected(self):
        conf = _run_conf(SPATIAL)
        obs: dict[str, Any] = {
            "observation_handler_conf": {"perception": "vector_markovian"}
        }
        seeing_all = conf.model_copy(update={"obs_conf": type(conf.obs_conf)(**obs)})

        with pytest.raises(ValueError, match="does not allow the perception"):
            SCENARIOS[SPATIAL].build(SPATIAL.value, seeing_all)

    def test_a_config_that_passes_is_composed_under_its_name(self):
        scenario = SCENARIOS[SPATIAL].build(SPATIAL.value, _run_conf(SPATIAL))

        assert scenario.scenario_name == SPATIAL.value
