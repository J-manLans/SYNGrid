"""When a single tier chain episode ends, and what the ending costs."""

from collections.abc import Callable

import numpy as np
import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.max_tier_digester import (
    MaxTierDigester,
)
from syn_grid.core.droid.synergy_droid import DroidAction, SynergyDroid
from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)
from syn_grid.scenario.families.goal.single_tier_chain.termination import (
    SingleTierChainTermination,
)

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 3


MAX_STEPS = 5


TIMEOUT_PENALTY = -0.5


# ================= #
#      Helpers      #
# ================= #


def _eat_tiers(world: GridWorld, *tiers: int) -> None:
    """The droid eats a tier orb of each tier, in the order given."""

    for tier in tiers:
        orb = next(o for o in world.ALL_ORBS if o.META.TIER == tier)
        world.droid.consume_orb(orb)


def _tick(world: GridWorld, ticks: int) -> None:
    """Run the world's clock, the droid stepping between two cells away from the edge."""

    for i in range(ticks):
        world.perform_droid_action(
            DroidAction.LEFT if i % 2 == 0 else DroidAction.RIGHT
        )


# === Truncation === #


def _out_of_energy(world: GridWorld) -> None:
    world.droid.energy = 0.0


def _chain_broken(world: GridWorld) -> None:
    _eat_tiers(world, 2)


def _chain_completed(world: GridWorld) -> None:
    _eat_tiers(world, *range(1, MAX_TIER + 1))


def _out_of_steps(world: GridWorld) -> None:
    _tick(world, MAX_STEPS)


def _going_on(world: GridWorld) -> None:
    _tick(world, MAX_STEPS - 1)


def _completed_when_the_steps_are_out(world: GridWorld) -> None:
    _tick(world, MAX_STEPS)
    _chain_completed(world)


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def world() -> GridWorld:
    """A 3x3 world with a three-tier chain and an empty field, at the start of an episode.

    With no orb on the field, the clock runs without the droid eating anything.
    """

    grid = (3, 3)
    orbs = TierChainPopulation(
        max_tier=MAX_TIER,
        base_reward=1.0,
        growth_factor=1.0,
        cool_down=None,
        life_span=None,
    ).create()
    droid = SynergyDroid(
        [MaxTierDigester(max_tier=MAX_TIER, chain_break_penalty=-0.1)],
        grid,
        max_energy=50.0,
        step_penalty=-0.001,
        boundary_penalty=-1.0,
    )
    world = GridWorld(grid, 0, MAX_STEPS, orbs, droid)
    world.reset(np.random.default_rng(0))

    return world


@pytest.fixture
def termination() -> SingleTierChainTermination:
    return SingleTierChainTermination(TIMEOUT_PENALTY)


# ================= #
#       Tests       #
# ================= #


class TestDroidEndsTheTask:
    """The droid ends the task, and the ending costs nothing."""

    def test_out_of_energy_terminates_at_no_cost(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        _out_of_energy(world)

        outcome = termination.evaluate(world)

        assert outcome.terminated
        assert outcome.terminal_reward == 0.0

    def test_a_broken_chain_terminates_at_no_cost(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        _chain_broken(world)

        outcome = termination.evaluate(world)

        assert outcome.terminated
        assert outcome.terminal_reward == 0.0

    def test_a_completed_chain_terminates_at_no_cost(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        _chain_completed(world)

        outcome = termination.evaluate(world)

        assert outcome.terminated
        assert outcome.terminal_reward == 0.0


class TestTimeout:
    """The steps run out."""

    def test_running_out_of_steps_terminates_at_the_timeout_penalty(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        _out_of_steps(world)

        outcome = termination.evaluate(world)

        assert outcome.terminated
        assert outcome.terminal_reward == TIMEOUT_PENALTY

    def test_a_completed_chain_when_the_steps_are_out_terminates_at_no_cost(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        _completed_when_the_steps_are_out(world)

        outcome = termination.evaluate(world)

        assert outcome.terminated
        assert outcome.terminal_reward == 0.0


class TestGoingOn:
    """Nothing has ended the episode."""

    def test_the_episode_goes_on_at_no_cost(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        _going_on(world)

        outcome = termination.evaluate(world)

        # Gymnasium's terminated is a bool, so False and not only falsy
        assert outcome.terminated is False
        assert outcome.terminal_reward == 0.0

    def test_a_droid_with_less_than_one_energy_left_goes_on(
        self, world: GridWorld, termination: SingleTierChainTermination
    ):
        world.droid.energy = 0.5

        assert termination.evaluate(world).terminated is False


class TestTruncation:
    """A goal scenario never truncates."""

    @pytest.mark.parametrize(
        "situation",
        [
            _out_of_energy,
            _chain_broken,
            _chain_completed,
            _out_of_steps,
            _going_on,
            _completed_when_the_steps_are_out,
        ],
    )
    def test_truncation_is_never_set(
        self,
        world: GridWorld,
        termination: SingleTierChainTermination,
        situation: Callable[[GridWorld], None],
    ):
        situation(world)

        assert not termination.evaluate(world).truncated
