from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.config.models.common_models import ScenarioConf, WorldConf
from syn_grid.core.digestion.digestion import OrbDigester
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.spawning import SpawningRules
from syn_grid.scenario.scenario import Scenario


# ============== #
#   Orb recipe   #
# ============== #


@dataclass(frozen=True)
class OrbRecipe:
    """
    One orb kind, as a continuous scenario is handed it: how to make its orbs and how to make the
    digester that scores them.

    A recipe and not instances, because one scenario builds many worlds and each world needs its
    own orbs and its own digester. Adding a kind to a scenario is adding one recipe.

    Sketched here because continuous is the first place that needs it. If the tier chain adopts it
    too, it moves to `blocks/`.

    Attributes:
        weight: this kind's share of the orb pool, relative to the other recipes.
        make_orbs: returns that many new orbs of this kind.
        make_digester: returns a new digester for this kind.
    """

    weight: int
    make_orbs: Callable[[int], list[BaseOrb]]
    make_digester: Callable[[], OrbDigester]


# ======================= #
#   Continuous Scenario   #
# ======================= #


def _continuous_scenario(
    scenario_name: str,
    scenario_tag: str,
    scenario_conf: ScenarioConf,
    orb_recipes: tuple[OrbRecipe, ...],
) -> Scenario:
    """
    Continuous: there is nothing to complete, so the droid collects for as long as the episode
    lasts.

    The shared helper for every continuous scenario. A builder decides which orb kinds its world
    holds and passes them in as recipes; this never reads an orb choice from the config. It
    composes everything continuous scenarios have in common:

    - spawning: the field starts with one orb and is topped up by one orb per step.
    - observation: slot count and sort limit follow from the size of the active field.
    - termination: the score running out, or the clock.
    - hud and metrics: not decided yet.
    - build_world: `_build_continuous_world`, bound to this scenario's pieces.
    """
    ...


# ----------- #
#   Helpers   #
# ----------- #


def _build_continuous_world(
    world_conf: WorldConf,
    orb_recipes: tuple[OrbRecipe, ...],
    spawning: SpawningRules,
) -> GridWorld:
    """
    Build one continuous world.

    Called once per environment. Each recipe supplies its share of the weighted pool and its own
    digester, so everything that holds episode state is created here rather than shared through
    the scenario.
    """
    ...


# ============ #
#   Builders   #
# ============ #


def build_continuous_sandbox(scenario_name: str, scenario_conf: ScenarioConf) -> Scenario:
    """
    Continuous, Sandbox: the user chooses which orb kinds are in the world and how they are
    weighted.

    The thinnest builder over `_continuous_scenario`: it turns every orb kind the config enables
    into a recipe and passes them straight through. Not a fixed benchmark task, so runs are only
    comparable when their configs are.

    Axis and tag: the enabled orb kinds.
    """
    ...
