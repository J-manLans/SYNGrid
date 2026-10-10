from syn_grid.config.models.common_scenario_models import NegOrbConf, ScenarioConf
from syn_grid.scenario.scenario import Scenario
from syn_grid.scenario.goal.tier_chain.config import TierScenarioConf, TierWorldConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.blocks.orb_bundle import OrbBundle
from syn_grid.scenario.blocks.spawning import SpawningRules
from syn_grid.scenario.utils.helpers import neg_orb


# ===================== #
#  Tier Chain Scenario  #
# ===================== #


def _tier_chain_scenario(
    scenario_name: str,
    scenario_tag: str,
    scenario_conf: TierScenarioConf,
    tier_bundle: OrbBundle,
    delay_on_consume: int | None = None,
    max_score: int | None = None,
) -> Scenario:
    """
    Goal/Tier Chain: collect every tier in order before the episode steps are used up.

    The shared helper for the tier-chain family. A builder decides what differs between its
    scenario and the others (the tier bundle, which carries the orbs and the scoring mode, and the
    delay) and passes it in; this composes everything the family has in common:

    - orb bundles: the tier bundle, plus `_negative_bundle` when the config has a negative block.
    - spawning: the whole chain is on the field from the first step and no tier orb expires.
    - observation: slot count, sort limit and tier bound all follow from `max_tier`.
    - termination: a broken chain, a finished chain, or the clock.
    - hud: the current chain length.
    - metrics: chains broken, progressed and completed.
    - build_world: `_build_tier_chain_world`, bound to this scenario's pieces.
    """
    ...


# ----------------- #
#   World readers   #
# ----------------- #

# Handed to the hud, the metrics and the observation. Module-level so that two scenarios built
# from the same config hold the same function objects and compare equal.


def _chain_progress(world: GridWorld) -> float:
    """The length of the chain right now."""
    ...


def _chains_broken(world: GridWorld) -> float:
    """How many chains have broken this episode."""
    ...


def _chains_progressed(world: GridWorld) -> float:
    """How many times a chain has grown by one tier this episode."""
    ...


def _chains_completed(world: GridWorld) -> float:
    """How many chains have been completed this episode."""
    ...


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
    ...


def _negative_bundle(negative_conf: NegOrbConf) -> OrbBundle:
    """
    The negative orb and its digester, for a config that has a negative block.

    A tier chain holds one negative orb. It spawns, despawns at the end of its lifespan and
    returns elsewhere after its cool-down.
    """
    ...


# ============ #
#   Builders   #
# ============ #


def build_tier_chain_spatial(
    scenario_name: str, scenario_conf: ScenarioConf
) -> Scenario:
    """
    Tier Chain, Spatial: the chain is laid out across the grid, but the droid only sees a 3x3
    window around itself.

    Axis and tag: the grid size.
    Scoring: max-tier. Only the completed chain pays, so the orbs are worth nothing on their own.
    Both are settled here, in the tier bundle this builder makes.
    """
    ...


def build_tier_chain_scaling_sparse(
    scenario_name: str, scenario_conf: ScenarioConf
) -> Scenario:
    """
    Tier Chain, Scaling Sparse.

    Axis and tag: the chain length and the grid size.
    Scoring: max-tier.
    """
    ...


def build_tier_chain_scaling_dense(
    scenario_name: str, scenario_conf: ScenarioConf
) -> Scenario:
    """
    Tier Chain, Scaling Dense.

    Axis and tag: the chain length and the grid size.
    Scoring: threshold. Each tier is worth something, so this scenario's config carries the
    reward ladder.
    """
    ...


def build_tier_chain_delay(scenario_name: str, scenario_conf: ScenarioConf) -> Scenario:
    """
    Tier Chain, Delay: consuming an orb puts the whole field on cooldown.

    Axis and tag: the delay.
    Scoring: max-tier.
    """
    ...