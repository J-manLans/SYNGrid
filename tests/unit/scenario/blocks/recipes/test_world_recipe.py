"""What a world recipe makes: a new world with the config's grid, step limit and droid, and every bundle's orbs
and digesters."""

import numpy as np

from syn_grid.config.models.scenario.common_models import DroidConf, GridConf, WorldConf
from syn_grid.core.droid.digestion.digesters.direct_digester import NegDigester
from syn_grid.core.droid.digestion.digesters.tier_chain.step_wise_digester import (
    StepWiseDigester,
)
from syn_grid.core.droid.synergy_droid import DroidAction
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.orb_meta import DirectKind, SynergyKind
from syn_grid.scenario.blocks.orb_bundle import OrbBundle
from syn_grid.scenario.blocks.populations.neg_population import NegPopulation
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)
from syn_grid.scenario.blocks.recipes.tier_digester_recipe import TierDigesterRecipe
from syn_grid.scenario.blocks.recipes.world_recipe import WorldRecipe

# ================= #
#     Constants     #
# ================= #


# One row of three cells: the droid starts in the middle, one step from either edge
ROWS, COLS = 1, 3
MAX_STEPS = 7
MAX_ENERGY = 20.0
STEP_PENALTY = -0.5
BOUNDARY_PENALTY = -2.0
MAX_TIER = 2
# The grid has two cells beside the droid's, and the bundles make three orbs
MAX_ACTIVE_ORBS = 2


# ================= #
#      Helpers      #
# ================= #


def _recipe() -> WorldRecipe:
    """A recipe with a two-tier chain, scored step-wise, and a negative orb."""

    world_conf = WorldConf(
        max_steps=MAX_STEPS,
        grid_conf=GridConf(grid_rows=ROWS, grid_cols=COLS),
        droid_conf=DroidConf(
            max_energy=MAX_ENERGY,
            step_penalty=STEP_PENALTY,
            boundary_penalty=BOUNDARY_PENALTY,
        ),
    )
    bundles = (
        OrbBundle(
            population=TierChainPopulation(
                max_tier=MAX_TIER,
                base_reward=1.0,
                growth_factor=1.0,
                cool_down=None,
                life_span=None,
            ),
            make_digester=TierDigesterRecipe(
                StepWiseDigester, max_tier=MAX_TIER, chain_break_penalty=-1.0
            ),
        ),
        OrbBundle(
            population=NegPopulation(orbs=1, reward=-1.0, cool_down=1, life_span=1),
            make_digester=NegDigester,
        ),
    )

    return WorldRecipe(world_conf, bundles, max_active_orbs=MAX_ACTIVE_ORBS)


def _reset(world: GridWorld) -> GridWorld:
    world.reset(np.random.default_rng(0))

    return world


# ================= #
#       Tests       #
# ================= #


class TestCall:
    """The world each call makes."""

    def test_it_has_the_configs_grid_step_limit_and_field_size(self):
        world = _reset(_recipe()())

        assert world.grid_dimensions == (ROWS, COLS)
        assert world.steps_left == MAX_STEPS
        assert len(world.orbs_on_field()) == MAX_ACTIVE_ORBS

    def test_it_hands_the_droid_the_configs_energy_and_penalties(self):
        # A droid does not show its penalties, so they are read from what its steps cost
        droid = _reset(_recipe()()).droid

        assert droid.energy == MAX_ENERGY
        assert droid.take_step(DroidAction.LEFT) == STEP_PENALTY
        # Now on the edge, so the next step left runs into it
        assert droid.take_step(DroidAction.LEFT) == STEP_PENALTY + BOUNDARY_PENALTY

    def test_it_holds_every_bundles_orbs_and_a_digester_for_each_bundle(self):
        world = _recipe()()

        kinds = sorted(
            (orb.META.KIND.name, orb.META.TIER or 0) for orb in world.ALL_ORBS
        )

        assert kinds == [
            (DirectKind.NEGATIVE.name, 0),
            (SynergyKind.TIER.name, 1),
            (SynergyKind.TIER.name, 2),
        ]
        assert isinstance(world.droid.get_digester(StepWiseDigester), StepWiseDigester)
        assert isinstance(world.droid.get_digester(NegDigester), NegDigester)

    def test_each_call_makes_a_new_world_that_shares_no_orb_droid_or_digester(self):
        recipe = _recipe()
        first, second = recipe(), recipe()

        assert first.droid is not second.droid
        assert not {id(orb) for orb in first.ALL_ORBS} & {
            id(orb) for orb in second.ALL_ORBS
        }
        assert first.droid.get_digester(StepWiseDigester) is not (
            second.droid.get_digester(StepWiseDigester)
        )
        assert first.droid.get_digester(NegDigester) is not (
            second.droid.get_digester(NegDigester)
        )
