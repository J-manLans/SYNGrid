"""The checks a scenario entry makes before it composes a scenario."""

from typing import Any

import pytest

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Constants     #
# ================= #


SPATIAL = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL


DELAY = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY


# ================= #
#      Helpers      #
# ================= #


def _run_conf(name: ScenarioName) -> ScenarioConf:
    manager = ConfigManager()
    return SCENARIOS[name].full_conf(
        manager.load_raw(manager.scenario_config_path(name))
    )


# ================= #
#       Tests       #
# ================= #


class TestBuild:
    """The checks an entry makes before it composes a scenario."""

    def test_a_config_of_another_class_is_rejected(self):
        with pytest.raises(
            TypeError, match="configured by .* given a TierScenarioConf"
        ):
            SCENARIOS[DELAY].build(DELAY.value, _run_conf(SPATIAL))

    def test_a_perception_without_the_scenarios_visibility_is_rejected(self):
        conf = _run_conf(SPATIAL)
        obs: dict[str, Any] = {
            "observation_handler_conf": {"perception": "vector_markovian"}
        }
        seeing_all = conf.model_copy(update={"obs_conf": type(conf.obs_conf)(**obs)})

        with pytest.raises(
            ValueError,
            match="does not allow the perception 'vector_markovian'. Allowed: vector_fog_of_war$",
        ):
            SCENARIOS[SPATIAL].build(SPATIAL.value, seeing_all)
