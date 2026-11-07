"""What one tick of the world does, how it fills its field, and which cells are free."""

import numpy as np
import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.max_tier_digester import (
    MaxTierDigester,
)
from syn_grid.core.droid.synergy_droid import DroidAction, SynergyDroid
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

STEP_PENALTY = -0.001
CHAIN_BREAK_PENALTY = -0.1
MAX_STEPS = 100

# ================= #
#     Fixtures      #
# ================= #


def _make_world(
    grid: tuple[int, int], max_active_orbs: int, life_span: int | None = None
) -> GridWorld:
    """A world with a three-tier chain, before its first reset."""

    orbs = TierChainPopulation(
        max_tier=3,
        base_reward=1.0,
        growth_factor=1.5,
        cool_down=None,
        life_span=life_span,
    ).create()
    droid = SynergyDroid(
        [MaxTierDigester(max_tier=3, chain_break_penalty=CHAIN_BREAK_PENALTY)],
        grid,
        max_energy=50.0,
        step_penalty=STEP_PENALTY,
        boundary_penalty=-1.0,
    )
    return GridWorld(grid, max_active_orbs, MAX_STEPS, orbs, droid)


@pytest.fixture
def world() -> GridWorld:
    """A 5x5 world with a three-tier chain. The droid starts in the centre, at (2, 2), with tier 1
    just left of it and the other tiers far away."""

    world = _make_world((5, 5), 3)
    world.reset(np.random.default_rng(0))

    # Place the orbs by hand, so the next step's outcome is known
    cells = {1: [2, 1], 2: [0, 0], 3: [4, 4]}
    for orb in world.ALL_ORBS:
        tier = orb.META.TIER
        assert tier is not None
        orb.position = cells[tier]

    return world


def _tier(world: GridWorld, tier: int) -> BaseOrb:
    return next(orb for orb in world.ALL_ORBS if orb.META.TIER == tier)


def _tier_1(world: GridWorld) -> BaseOrb:
    return _tier(world, 1)


# ================= #
#      Tests        #
# ================= #


class TestPerformDroidAction:
    """What one tick of the world reports: the orb eaten, and the reward."""

    def test_eating_reports_the_orb_and_what_the_droid_took_in(self, world: GridWorld):
        # Tier 2 before tier 1 breaks the chain, so the orb pays something: the break's penalty
        tier_1, tier_2 = _tier_1(world), _tier(world, 2)
        tier_1.position, tier_2.position = tier_2.position, tier_1.position

        result = world.perform_droid_action(DroidAction.LEFT)

        assert result.consumed is tier_2.META
        assert result.reward == pytest.approx(STEP_PENALTY + CHAIN_BREAK_PENALTY)

    def test_an_empty_step_reports_nothing_and_costs_its_penalty(
        self, world: GridWorld
    ):
        result = world.perform_droid_action(DroidAction.RIGHT)

        assert result.consumed is None
        assert result.reward == pytest.approx(STEP_PENALTY)


class TestLifeSpan:
    """An orb with a life span leaves the field when it runs out."""

    def test_it_leaves_unless_the_droid_eats_it_on_its_last_tick(self):
        world = _make_world((5, 5), 3, life_span=2)
        world.reset(np.random.default_rng(0))
        cells = {1: [2, 4], 2: [0, 0], 3: [4, 4]}
        for orb in world.ALL_ORBS:
            assert orb.META.TIER is not None
            orb.position = cells[orb.META.TIER]

        world.perform_droid_action(DroidAction.RIGHT)
        assert len(world.orbs_on_field()) == 3

        result = world.perform_droid_action(DroidAction.RIGHT)
        assert result.consumed is _tier_1(world).META
        assert world.orbs_on_field() == []


class TestReset:
    """Filling the field at the start of an episode."""

    def test_every_orb_lands_inside_a_grid_with_more_columns_than_rows(self):
        world = _make_world((2, 6), 3)

        for seed in range(50):
            world.reset(np.random.default_rng(seed))

            assert world.grid_dimensions == (2, 6)
            for orb in world.orbs_on_field():
                assert 0 <= orb.position[0] < 2 and 0 <= orb.position[1] < 6

    def test_a_full_field_takes_every_cell_but_the_droids(self):
        world = _make_world((2, 2), 3)

        for seed in range(20):
            world.reset(np.random.default_rng(seed))

            cells = [tuple(o.position) for o in world.orbs_on_field()]
            cells.append(tuple(world.droid.position))
            assert sorted(cells) == [(0, 0), (0, 1), (1, 0), (1, 1)]


class TestStepClock:
    """The steps an episode has left, which termination reads to end it."""

    def test_each_tick_takes_one_step_and_a_reset_restores_them(self, world: GridWorld):
        world.perform_droid_action(DroidAction.RIGHT)
        world.perform_droid_action(DroidAction.RIGHT)

        assert world.steps_left == MAX_STEPS - 2

        world.reset(np.random.default_rng(0))

        assert world.steps_left == MAX_STEPS


class TestFieldLimit:
    """The field holds no more than `max_active_orbs` at once."""

    def test_a_ready_orb_waits_while_the_field_is_full(self):
        world = _make_world((5, 5), 2)
        world.reset(np.random.default_rng(0))
        for orb, cell in zip(world.orbs_on_field(), ([0, 0], [4, 4])):
            orb.position = cell

        world.perform_droid_action(DroidAction.RIGHT)

        assert len(world.orbs_on_field()) == 2


class TestIsCellFree:
    """Whether an orb can be placed in a cell: no droid there, and no orb on the field."""

    def test_free_only_without_the_droid_or_an_orb_on_the_field(self, world: GridWorld):
        assert world.is_cell_free([1, 1])
        assert not world.is_cell_free([2, 2])
        assert not world.is_cell_free([2, 1])

        _tier_1(world).de_spawn()

        assert world.is_cell_free([2, 1])
