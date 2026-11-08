"""The checks a scenario entry makes before it composes a scenario, and the base config it hands out."""

from dataclasses import replace
from typing import Any

import pytest
import yaml

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Constants     #
# ================= #


SPATIAL = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL


DELAY = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY


# ================= #
#       Tests       #
# ================= #


class TestBuild:
    """The checks an entry makes before it composes a scenario."""

    def test_a_config_of_another_class_is_rejected(self):
        with pytest.raises(
            TypeError, match="configured by .* given a TierScenarioConf"
        ):
            SCENARIOS[DELAY].build(DELAY.value, SCENARIOS[SPATIAL].base_conf())

    def test_a_perception_without_the_scenarios_visibility_is_rejected(self):
        conf = SCENARIOS[SPATIAL].base_conf()
        obs: dict[str, Any] = {
            "observation_handler_conf": {"perception": "vector_markovian"}
        }
        seeing_all = conf.model_copy(update={"obs_conf": type(conf.obs_conf)(**obs)})

        with pytest.raises(
            ValueError,
            match="does not allow the perception 'vector_markovian'. Allowed: vector_fog_of_war$",
        ):
            SCENARIOS[SPATIAL].build(SPATIAL.value, seeing_all)


class TestBaseConf:
    """The config a locked scenario was run with."""

    def test_it_is_the_base_config_file_as_the_full_model(self):
        entry = SCENARIOS[DELAY]
        assert entry.lock is not None
        written = yaml.safe_load(entry.lock.base_config_path.read_text())

        assert entry.base_conf() == entry.conf_class(**written)

    def test_a_scenario_that_is_not_locked_has_none(self):
        unlocked = replace(SCENARIOS[SPATIAL], lock=None)

        with pytest.raises(ValueError, match="not locked"):
            unlocked.base_conf()
