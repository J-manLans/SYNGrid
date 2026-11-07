"""The observation handler hands out observations a caller can keep."""

import numpy as np
import pytest

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.gymnasium.environment import SYNGridEnv
from syn_grid.gymnasium.observation_space.observation_handler import ObservationHandler
from syn_grid.scenario.registry import SCENARIOS, build_scenario
from syn_grid.scenario.scenario import Scenario

# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def scenario() -> Scenario:
    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
    manager = ConfigManager()
    conf = SCENARIOS[name].full_conf(
        manager.load_raw(manager.scenario_config_path(name))
    )
    return build_scenario(name, conf)


@pytest.fixture
def env(scenario: Scenario) -> SYNGridEnv:
    return SYNGridEnv(scenario, render_mode=None)


# ================= #
#       Tests       #
# ================= #


class TestGetObservation:
    """Each observation is an array of its own, so the next one does not overwrite it."""

    def test_an_observation_keeps_its_values_after_the_world_moves(
        self, scenario: Scenario, env: SYNGridEnv
    ):
        env.reset(seed=0)
        handler = ObservationHandler(
            scenario.observation, env.world.grid_dimensions, env.world.max_identity
        )
        handler.setup_obs_space()
        handler.reset()

        first = handler.get_observation(env.world)
        before = first.copy()
        env.step(0)
        second = handler.get_observation(env.world)

        # The step moved what the perception sees, so a shared buffer would show it in `first`
        assert not np.array_equal(second, before)
        assert np.array_equal(first, before)
