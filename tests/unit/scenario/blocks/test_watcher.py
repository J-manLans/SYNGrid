import numpy as np
import pytest

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.core.grid_world import GridWorld, StepResult
from syn_grid.scenario.blocks.watcher import NoWatcher
from syn_grid.scenario.registry import SCENARIOS, build_scenario
from syn_grid.scenario.scenario import Scenario

# ================= #
#      Helpers      #
# ================= #


def _field_state(world: GridWorld) -> list[tuple[bool, list[int], int | None]]:
    return [
        (orb.is_active, list(orb.position), orb.TIMER.remaining)
        for orb in world.ALL_ORBS
    ]


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def scenario() -> Scenario:
    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
    conf: ScenarioConf = SCENARIOS[name].base_conf()

    return build_scenario(name, conf)


@pytest.fixture
def world(scenario: Scenario) -> GridWorld:
    world = scenario.build_world()
    world.reset(np.random.default_rng(0))

    return world


# ================= #
#       Tests       #
# ================= #


class TestNoWatcher:
    """The watcher of a scenario that states none."""

    def test_react_leaves_the_world_as_it_was(self, world: GridWorld):
        orb = world.orbs_on_field()[0]
        before = _field_state(world)
        droid_before = (
            list(world.droid.position),
            world.droid.energy,
            world.droid.score,
        )

        NoWatcher().react(world, StepResult(reward=0.0, consumed=orb.META))

        assert _field_state(world) == before
        assert (
            list(world.droid.position),
            world.droid.energy,
            world.droid.score,
        ) == droid_before
