"""
The tier chain family's builders.

One builder per registered tier-chain scenario, the helper that composes what
they share, and the function that builds a tier-chain world.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from syn_grid.config.models.scenario.orb_models import NegOrbConf
from syn_grid.scenario.blocks.hud import HudStat
from syn_grid.scenario.blocks.metrics import Metric
from syn_grid.scenario.blocks.observation import ObservationRules
from syn_grid.scenario.blocks.orb_bundle import OrbBundle
from syn_grid.scenario.blocks.spawning import SpawningRules
from syn_grid.scenario.goal.tier_chain.config import (
    TierDelayScenarioConf,
    TierDenseScenarioConf,
    TierScenarioConf,
    TierWorldConf,
)
from syn_grid.scenario.goal.tier_chain.population import TierChainPopulation
from syn_grid.scenario.goal.tier_chain.termination import TierChainTermination
from syn_grid.scenario.scenario import Scenario

if TYPE_CHECKING:
    from syn_grid.core.droid.digestion.digestion import OrbDigester
    from syn_grid.core.grid_world import GridWorld


# ===================== #
#  Tier Chain Scenario  #
# ===================== #


def _tier_chain_scenario(
    scenario_name: str,
    scenario_tag: str,
    scenario_conf: TierScenarioConf,
    tier_bundle: OrbBundle,
) -> Scenario:
    """
    Goal/Tier Chain: collect every tier in order before the episode steps are used up.

    The shared helper for the tier-chain family. A builder decides what differs between its
    scenario and the others (the tier bundle, which carries the orbs and the scoring mode, and the
    delay) and passes it in; this composes everything the family has in common:

    - orb bundles: the tier bundle, plus `_negative_bundle` when the config has a negative block.
    - scenario tag: the builder's tag, plus `_Neg` when the config has a negative block.
    - spawning: the whole chain is on the field from the first step and no tier orb expires.
    - observation: slot count, sort limit and tier bound all follow from `max_tier`.
    - termination: a broken chain, a finished chain, or the clock.
    - hud: the current chain length.
    - metrics: chains broken, progressed and completed.
    - build_world: `_build_tier_chain_world`, bound to this scenario's pieces.
    """

    world_conf = scenario_conf.world_conf
    max_tier = world_conf.tier_orb_conf.max_tier

    orb_bundles = (tier_bundle,)
    if world_conf.neg_orb_conf is not None:
        orb_bundles += (_negative_bundle(world_conf.neg_orb_conf),)
        scenario_tag += "_Neg"

    spawning = SpawningRules(
        spawn_max_on_episode_start=True,
        max_active_orbs=max_tier,  # The chain defines the field size.
        tier_orb_expires=False,
    )

    observation = ObservationRules(
        perception=scenario_conf.obs_conf.observation_handler_conf.perception,
        max_steps=world_conf.max_steps,
        observation_slot_count=max_tier,  # Phase 8: one more when a negative orb is present.
        sort_limit=max_tier,
        max_tier=max_tier,
        global_features=(),
    )

    termination = TierChainTermination(
        timeout_penalty=world_conf.goal_conf.timeout_penalty,
        completion_reward=world_conf.goal_conf.completion_reward,
    )

    hud = (HudStat("current tier chain", _chain_progress),)

    # The names are the old engine's stats keys. The logger's LogKey comes in phase 9.
    metrics = (
        Metric("chains_broken", _chains_broken),
        Metric("chains_progressed", _chains_progressed),
        Metric("chains_completed", _chains_completed),
    )

    return Scenario(
        scenario_name=scenario_name,
        scenario_tag=scenario_tag,
        observation=observation,
        termination=termination,
        hud=hud,
        metrics=metrics,
        build_world=partial(_build_tier_chain_world, world_conf, orb_bundles, spawning),
    )


# ----------------- #
#   World readers   #
# ----------------- #

# Handed to the hud, the metrics and the observation. Module-level so that two scenarios built
# from the same config hold the same function objects and compare equal.


def _chain_progress(world: GridWorld) -> float:
    """The length of the chain right now."""
    return 0.0


def _chains_broken(world: GridWorld) -> float:
    """How many chains have broken this episode."""
    return 0.0


def _chains_progressed(world: GridWorld) -> float:
    """How many times a chain has grown by one tier this episode."""
    return 0.0


def _chains_completed(world: GridWorld) -> float:
    """How many chains have been completed this episode."""
    return 0.0


# ----------- #
#   Helpers   #
# ----------- #


def _build_tier_chain_world(
    world_conf: TierWorldConf,
    orb_bundles: tuple[OrbBundle, ...],
    spawning: SpawningRules,
) -> GridWorld:
    """
    Build one tier-chain world.

    Called once per environment, so everything that holds episode state -- the digesters, the
    droid, the orbs -- is created here rather than shared through the scenario. The orbs are every
    bundle's orbs joined and the digesters are one per bundle; nothing here asks which kinds they
    are.
    """

    # orbs = [orb for bundle in orb_bundles for orb in bundle.population.create()]
    # digesters = [bundle.make_digester() for bundle in orb_bundles]
    # grid = (world_conf.grid_conf.grid_rows, world_conf.grid_conf.grid_cols)
    # TODO(phase 3): build the droid from the digesters, the grid and `droid_conf`, then return
    # GridWorld(grid, droid, orbs, spawning).
    raise NotImplementedError(
        "_build_tier_chain_world waits for core to take pieces (phase 3)"
    )


def _make_tier_digester() -> OrbDigester:
    """Return a new max-tier digester."""
    raise NotImplementedError("the tier digester is not written yet (phase 1)")


def _negative_bundle(negative_conf: NegOrbConf) -> OrbBundle:
    """
    The negative orb and its digester, for a config that has a negative block.

    A tier chain holds one negative orb. It spawns, despawns at the end of its lifespan and
    returns elsewhere after its cool-down.
    """

    raise NotImplementedError("the negative orb bundle is phase 8")


# ============ #
#   Builders   #
# ============ #


def build_tier_chain_spatial(
    scenario_name: str, scenario_conf: TierScenarioConf
) -> Scenario:
    """
    Tier Chain, Spatial: the complete tier chain is laid out across the grid, but the droid only
    sees a 3x3 window around itself.
    Difficulty axis: the grid size.
    Scoring: Only the completed chain pays, so the orbs are worth nothing on their own.

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

    # Max-tier scoring: the orbs are worth nothing, the digester pays the completed chain.
    tier_bundle = OrbBundle(
        population=TierChainPopulation(world_conf.tier_orb_conf.max_tier),
        make_digester=_make_tier_digester,
    )

    return _tier_chain_scenario(scenario_name, scenario_tag, scenario_conf, tier_bundle)


def build_tier_chain_scaling_sparse(
    scenario_name: str, scenario_conf: TierScenarioConf
) -> Scenario:
    """
    Tier Chain, Scaling Sparse.

    Axis and tag: the chain length and the grid size.
    Scoring: max-tier.
    """

    raise NotImplementedError("scaling sparse is phase 8")


def build_tier_chain_scaling_dense(
    scenario_name: str, scenario_conf: TierDenseScenarioConf
) -> Scenario:
    """
    Tier Chain, Scaling Dense.

    Axis and tag: the chain length and the grid size.
    Scoring: threshold. Each tier is worth something, so this scenario's config carries the
    reward ladder.
    """

    raise NotImplementedError("scaling dense is phase 8")


def build_tier_chain_delay(
    scenario_name: str, scenario_conf: TierDelayScenarioConf
) -> Scenario:
    """
    Tier Chain, Delay: consuming an orb puts the whole field on cooldown.

    Axis and tag: the delay.
    Scoring: max-tier.
    """

    raise NotImplementedError("delay is phase 8")
