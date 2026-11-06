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

# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def world() -> GridWorld:
    """A 5x5 world with a three-tier chain. The droid starts in the centre, at (2, 2), with tier 1
    just left of it and the other tiers far away."""

    orbs = TierChainPopulation(
        max_tier=3, base_reward=1.0, growth_factor=1.5, cool_down=None, life_span=None
    ).create()
    droid = SynergyDroid(
        [MaxTierDigester(max_tier=3, chain_break_penalty=-0.1)],
        (5, 5),
        max_energy=50.0,
        step_penalty=STEP_PENALTY,
        boundary_penalty=-1.0,
    )
    world = GridWorld((5, 5), 3, 100, orbs, droid)
    world.reset(np.random.default_rng(0))

    # Place the orbs by hand, so the next step's outcome is known
    cells = {1: [2, 1], 2: [0, 0], 3: [4, 4]}
    for orb in world.ALL_ORBS:
        tier = orb.META.TIER
        assert tier is not None
        orb.position = cells[tier]

    return world


def _tier_1(world: GridWorld) -> BaseOrb:
    return next(orb for orb in world.ALL_ORBS if orb.META.TIER == 1)


# ================= #
#      Tests        #
# ================= #


class TestPerformDroidAction:
    """What one tick of the world reports: the orb eaten, and the reward."""

    def test_eating_reports_the_orb_and_what_the_droid_took_in(self, world: GridWorld):
        tier_1 = _tier_1(world)
        score_before = world.droid.score

        result = world.perform_droid_action(DroidAction.LEFT)

        assert result.consumed is tier_1.META
        assert result.reward == pytest.approx(world.droid.score - score_before)

    def test_an_empty_step_reports_nothing_and_costs_its_penalty(
        self, world: GridWorld
    ):
        result = world.perform_droid_action(DroidAction.RIGHT)

        assert result.consumed is None
        assert result.reward == pytest.approx(STEP_PENALTY)


class TestIsCellFree:
    """Whether an orb can be placed in a cell: no droid there, and no orb on the field."""

    def test_free_only_without_the_droid_or_an_orb_on_the_field(self, world: GridWorld):
        assert world.is_cell_free([1, 1])
        assert not world.is_cell_free([2, 2])
        assert not world.is_cell_free([2, 1])

        _tier_1(world).de_spawn()

        assert world.is_cell_free([2, 1])
