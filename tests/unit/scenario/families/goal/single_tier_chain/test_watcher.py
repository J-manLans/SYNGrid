import numpy as np
import pytest

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.core.grid_world import GridWorld, StepResult
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectKind, SynergyKind
from syn_grid.scenario.families.goal.single_tier_chain.watcher import (
    DelayWatcher,
    DelayWatcherRecipe,
)
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Fixtures      #
# ================= #

DELAY = 3


@pytest.fixture
def world() -> GridWorld:
    """Delay's world, with the negative orb on, at the start of an episode."""

    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY
    manager = ConfigManager()
    raw = manager.load_raw(manager.scenario_config_path(name))
    raw["world_conf"]["neg_orb_conf"] = {"enabled": True}
    entry = SCENARIOS[name]
    world = entry.build(name.value, entry.full_conf(raw)).build_world()
    world.reset(np.random.default_rng(0))

    return world


@pytest.fixture
def watcher() -> DelayWatcher:
    watcher = DelayWatcher(DELAY)
    watcher.reset()

    return watcher


def _tier_orbs(world: GridWorld) -> list[BaseOrb]:
    return [o for o in world.ALL_ORBS if o.META.KIND == SynergyKind.TIER]


def _negative_orb(world: GridWorld) -> BaseOrb:
    return next(o for o in world.ALL_ORBS if o.META.KIND == DirectKind.NEGATIVE)


def _eat(world: GridWorld, orb: BaseOrb) -> StepResult:
    """What the world reports when the droid eats `orb`: the orb leaves the field."""

    orb.consume()

    return StepResult(reward=0.0, consumed=orb.META)


def _idle(watcher: DelayWatcher, world: GridWorld, steps: int) -> None:
    for _ in range(steps):
        watcher.react(world, StepResult(reward=0.0, consumed=None))


def _off_the_field(world: GridWorld) -> list[BaseOrb]:
    return [o for o in _tier_orbs(world) if not o.is_active]


# ================= #
#      Tests        #
# ================= #


class TestFreeze:
    """What eating an orb freezes."""

    def test_eating_a_tier_orb_freezes_the_other_tier_orbs_in_their_cells(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        eaten, *others = _tier_orbs(world)
        cells = [list(o.position) for o in others]

        watcher.react(world, _eat(world, eaten))

        assert all(not o.is_active for o in others)
        assert [o.position for o in others] == cells

    def test_eating_the_negative_orb_freezes_nothing(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        watcher.react(world, _eat(world, _negative_orb(world)))

        assert _off_the_field(world) == []

    def test_an_orb_eaten_while_frozen_does_not_restart_the_delay(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        eaten, *others = _tier_orbs(world)
        watcher.react(world, _eat(world, eaten))
        _idle(watcher, world, DELAY - 1)

        watcher.react(world, _eat(world, _negative_orb(world)))

        assert all(o.is_active for o in others)


class TestRelease:
    """When frozen orbs come back."""

    def test_frozen_orbs_return_after_the_delay_in_their_cells(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        eaten, *others = _tier_orbs(world)
        cells = [list(o.position) for o in others]
        watcher.react(world, _eat(world, eaten))

        _idle(watcher, world, DELAY - 1)
        assert all(not o.is_active for o in others)

        _idle(watcher, world, 1)
        assert all(o.is_active for o in others)
        assert [o.position for o in others] == cells

    def test_an_orb_whose_cell_is_taken_waits_for_it_to_clear(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        eaten, blocked, *_ = _tier_orbs(world)
        watcher.react(world, _eat(world, eaten))
        away = list(world.droid.position)
        world.droid.position = list(blocked.position)

        _idle(watcher, world, DELAY + 2)
        assert not blocked.is_active

        world.droid.position = away
        _idle(watcher, world, 1)
        assert blocked.is_active

    def test_after_every_orb_returns_the_next_eat_freezes_again(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        first, second, *rest = _tier_orbs(world)
        watcher.react(world, _eat(world, first))
        _idle(watcher, world, DELAY)

        watcher.react(world, _eat(world, second))

        assert all(not o.is_active for o in rest)


class TestReset:
    """A new episode."""

    def test_after_a_reset_the_next_eat_freezes(
        self, world: GridWorld, watcher: DelayWatcher
    ):
        watcher.react(world, _eat(world, _tier_orbs(world)[0]))

        watcher.reset()
        world.reset(np.random.default_rng(1))
        eaten, *others = _tier_orbs(world)
        watcher.react(world, _eat(world, eaten))

        assert all(not o.is_active for o in others)


class TestRecipe:
    """The recipe delay's scenario holds."""

    def test_each_call_makes_a_new_watcher(self):
        recipe = DelayWatcherRecipe(DELAY)

        assert recipe() is not recipe()
