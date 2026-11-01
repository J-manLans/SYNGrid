"""
The single tier chain family's builders.

One builder per registered single tier chain scenario, and the helpers that compose what they share.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from syn_grid.config.models.scenario.orb_models import NegOrbConf, TierOrbConf
from syn_grid.core.droid.digestion.digesters.neg_digester import NegDigester
from syn_grid.core.droid.digestion.digesters.tier_chain.max_tier_digester import (
    MaxTierDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.scenario.blocks.hud import HudBar, HudStat
from syn_grid.scenario.blocks.metrics import Metric
from syn_grid.scenario.blocks.observation import ObservationRules
from syn_grid.scenario.blocks.orb_bundle import OrbBundle
from syn_grid.scenario.blocks.populations.neg_population import NegPopulation
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)
from syn_grid.scenario.blocks.readers import energy, score, steps_left
from syn_grid.scenario.blocks.recipes.tier_digester_recipe import TierDigesterRecipe
from syn_grid.scenario.blocks.recipes.world_recipe import WorldRecipe
from syn_grid.scenario.families.goal.single_tier_chain.config import (
    TierDelayScenarioConf,
    TierScenarioConf,
)
from syn_grid.scenario.families.goal.single_tier_chain.termination import (
    SingleTierChainTermination,
)
from syn_grid.scenario.scenario import Scenario

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# ========================== #
# Single Tier Chain Scenario #
# ========================== #


def _single_tier_chain_scenario(
    scenario_name: str,
    scenario_tag: str,
    scenario_conf: TierScenarioConf,
    tier_bundle: OrbBundle,
) -> Scenario:
    """
    Goal/Single Tier Chain: collect every tier in order before the episode steps are used up.

    The shared helper for the single tier chain family. A builder decides what differs between its
    scenario and the others (the tier bundle, which carries the orbs and the scoring mode, and the
    delay) and passes it in; this composes everything the family has in common:

    - orb bundles: the tier bundle, plus `_negative_bundle` when the negative orb is enabled.
    - scenario tag: the builder's tag, then the episode's steps as `_steps100`, then `_Neg` when
      the negative orb is enabled.
    - orb field: it holds every orb the bundles make.
    - observation: the slot count and the sort limit are the number of orbs the bundles make.
      The tier bound is `max_tier`.
    - termination: out of energy, a broken chain, a finished chain, or the clock.
    - hud: energy, score, steps left and the current chain length.
    - metrics: chains broken, progressed and completed, plus negatives eaten when the negative
      orb is enabled.
    - build_world: a `WorldRecipe` holding this scenario's world config, bundles and field
      size.
    """

    world_conf = scenario_conf.world_conf

    hud = (
        HudBar("energy", world_conf.droid_conf.max_energy, energy),
        HudStat("score", score),
        HudBar("steps left", world_conf.max_steps, steps_left),
        HudStat("current tier chain", _chain_progress),
    )

    # The names are the old engine's stats keys. The logger's LogKey comes in phase 9.
    metrics = (
        Metric("chains_broken", _chains_broken),
        Metric("chain_progresses", _chain_progresses),
        Metric("chains_completed", _chains_completed),
    )

    orb_bundles = (tier_bundle,)
    scenario_tag += f"_steps{world_conf.max_steps}"

    if world_conf.neg_orb_conf.enabled:
        scenario_tag += "_Neg"
        metrics += (Metric("negatives_eaten", _negatives_eaten),)
        grid_conf = world_conf.grid_conf
        negative_life_span = _reachable_life_span(
            grid_conf.grid_rows, grid_conf.grid_cols
        )
        orb_bundles += (_negative_bundle(world_conf.neg_orb_conf, negative_life_span),)

    # How many orbs the field holds. The world's field size, the observation's slots and its sort
    # limit all follow from it.
    orb_count = sum(bundle.population.count for bundle in orb_bundles)

    observation = ObservationRules(
        perception=scenario_conf.obs_conf.observation_handler_conf.perception,
        observation_slot_count=orb_count,
        sort_limit=orb_count,
        global_features=(),
    )

    termination = SingleTierChainTermination(
        timeout_penalty=world_conf.goal_conf.timeout_penalty,
    )

    return Scenario(
        scenario_name=scenario_name,
        scenario_tag=scenario_tag,
        hud=hud,
        metrics=metrics,
        observation=observation,
        termination=termination,
        build_world=WorldRecipe(world_conf, orb_bundles, max_active_orbs=orb_count),
    )


# ----------------- #
#   World readers   #
# ----------------- #

# The family's own readers; the ones every scenario can use are in `scenario/blocks/readers.py`.
# Handed to the hud, the metrics and the observation. Module-level so that two scenarios built
# from the same config hold the same function objects and compare equal. Each takes its number
# from the digester that counts it, which the droid hands out by class.


def _chain_progress(world: GridWorld) -> float:
    """The length of the chain right now."""
    return world.droid.get_digester(TierDigester).chain_tier


def _chains_broken(world: GridWorld) -> float:
    """How many chains have broken this episode."""
    return world.droid.get_digester(TierDigester).chains_broken


def _chain_progresses(world: GridWorld) -> float:
    """How many times a chain has grown by one tier this episode."""
    return world.droid.get_digester(TierDigester).chain_progresses


def _chains_completed(world: GridWorld) -> float:
    """How many chains have been completed this episode."""
    return world.droid.get_digester(TierDigester).chains_completed


def _negatives_eaten(world: GridWorld) -> float:
    """How many negative orbs have been eaten this episode."""
    return world.droid.get_digester(NegDigester).negatives_eaten


# ----------- #
#   Helpers   #
# ----------- #


def _reachable_life_span(grid_rows: int, grid_cols: int) -> int:
    """
    The number of steps an orb stays on the field when it must be reachable from anywhere.

    It is the longest walk between two cells of the grid, from one corner to the opposite one,
    so a droid standing anywhere has time to reach an orb that appeared on the field.
    """

    return (grid_rows - 1) + (grid_cols - 1)


def _tier_bundle(
    tier_orb_conf: TierOrbConf,
    scoring_mode: Callable[[int, float], TierDigester],
) -> OrbBundle:
    """
    The tier orbs and their digester, scored the way the builder says.

    The orbs are one per tier, worth what the config's reward ladder says. `scoring_mode` is the
    tier digester class the scenario's worlds get. It is the one thing about the tier orbs that
    differs between the family's scenarios.
    """

    return OrbBundle(
        population=TierChainPopulation(
            max_tier=tier_orb_conf.max_tier,
            base_reward=tier_orb_conf.base_reward,
            growth_factor=tier_orb_conf.growth_factor,
            # No cool-down: an eaten tier orb does not return in that episode, so a chain can
            # never be broken with a tier 1.
            cool_down=None,
            # No lifespan: the whole chain is laid out at once and stays until it is eaten.
            life_span=None,
        ),
        make_digester=TierDigesterRecipe(
            scoring_mode=scoring_mode,
            max_tier=tier_orb_conf.max_tier,
            chain_break_penalty=tier_orb_conf.chain_break_penalty,
        ),
    )


def _negative_bundle(negative_conf: NegOrbConf, life_span: int) -> OrbBundle:
    """
    The negative orb and its digester, for a config whose negative orb is enabled.

    A single tier chain scenario holds one negative orb. It spawns, despawns at the end of
    `life_span` and returns elsewhere after its cool-down.
    """

    return OrbBundle(
        population=NegPopulation(
            reward=negative_conf.reward,
            cool_down=negative_conf.cool_down,
            life_span=life_span,
        ),
        make_digester=NegDigester,
    )


# ============ #
#   Builders   #
# ============ #


def build_single_tier_chain_spatial(
    scenario_name: str, scenario_conf: TierScenarioConf
) -> Scenario:
    """
    Single Tier Chain, Spatial: the complete tier chain is laid out across the grid, but the droid
    only sees a 3x3 window around itself.
    Difficulty axis: the grid size.
    Scoring: max-tier. Only the completed chain pays, and it pays what its orbs are worth together.

    This scenario tests spatial memory under partial observability. The droid may walk past a later
    tier while collecting an earlier one, and once it's out of the window, it's out of sight. It's
    like searching a dark room with a small flashlight: without remembering what the beam already
    passed over, you keep searching the same spots. The task never changes, only the grid size
    does, so grid size works as a standalone difficulty axis — giving the droid a bigger area to
    map.
    """

    world_conf = scenario_conf.world_conf
    grid_conf = world_conf.grid_conf
    scenario_tag = f"{grid_conf.grid_rows}x{grid_conf.grid_cols}"

    # Max-tier scoring: the digester pays the chain's whole worth when it is completed.
    tier_bundle = _tier_bundle(world_conf.tier_orb_conf, MaxTierDigester)

    return _single_tier_chain_scenario(
        scenario_name, scenario_tag, scenario_conf, tier_bundle
    )


def build_single_tier_chain_scaling_sparse(
    scenario_name: str, scenario_conf: TierScenarioConf
) -> Scenario:
    """
    Single Tier Chain, Scaling Sparse: the complete tier chain is laid out across the grid and the
    droid sees all of it.
    Difficulty axis: the chain length and the grid size.
    Scoring: max-tier. Only the completed chain pays, and it pays what its orbs are worth together.

    This scenario tests how the task scales with the length of the chain. Nothing is hidden, so
    the droid never has to remember where an orb was. What grows is the order it has to learn: the
    longer the chain, the longer it goes without being paid, and the more steps a wrong turn
    costs.
    """

    # NOTE(duplication): this body is spatial's apart from the tag, and dense's apart from the
    # digester. Left as a copy until delay is built and it is clear what the builders share.
    world_conf = scenario_conf.world_conf
    grid_conf = world_conf.grid_conf
    scenario_tag = f"tier{world_conf.tier_orb_conf.max_tier}_{grid_conf.grid_rows}x{grid_conf.grid_cols}"

    # Max-tier scoring: the digester pays the chain's whole worth when it is completed.
    tier_bundle = _tier_bundle(world_conf.tier_orb_conf, MaxTierDigester)

    return _single_tier_chain_scenario(
        scenario_name, scenario_tag, scenario_conf, tier_bundle
    )


def build_single_tier_chain_scaling_dense(
    scenario_name: str, scenario_conf: TierScenarioConf
) -> Scenario:
    """
    Single Tier Chain, Scaling Dense: the complete tier chain is laid out across the grid and the
    droid sees all of it.
    Difficulty axis: the chain length and the grid size.
    Scoring: threshold. A growing chain pays nothing. When it completes, or breaks, it pays what
    the orbs eaten so far are worth together, and a break also costs the chain break penalty.

    This scenario asks whether it is the silence that stops learning. The world is scaling
    sparse's, but a chain that breaks halfway still pays for the part it got right, so the droid
    is told when it is on the right track. It has no step penalty and a steeper reward ladder, as
    the thesis dense runs had.
    """

    world_conf = scenario_conf.world_conf
    grid_conf = world_conf.grid_conf
    scenario_tag = f"tier{world_conf.tier_orb_conf.max_tier}_{grid_conf.grid_rows}x{grid_conf.grid_cols}"

    # Threshold scoring: the digester holds the chain's worth while it grows and pays it when the
    # chain ends.
    tier_bundle = _tier_bundle(world_conf.tier_orb_conf, ThresholdDigester)

    return _single_tier_chain_scenario(
        scenario_name, scenario_tag, scenario_conf, tier_bundle
    )


def build_single_tier_chain_delay(
    scenario_name: str, scenario_conf: TierDelayScenarioConf
) -> Scenario:
    """
    Single Tier Chain, Delay: consuming an orb puts the whole field on cooldown.

    Axis and tag: the delay.
    Scoring: max-tier.
    """

    raise NotImplementedError("delay is phase 6")
