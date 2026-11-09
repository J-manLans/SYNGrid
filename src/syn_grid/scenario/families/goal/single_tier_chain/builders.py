"""
How the single tier chain family composes a scenario.

`SingleTierChainEntry` is the family's scenario entry: what every scenario in the family shares,
composed from the few things they differ by, which are the scoring mode, the tag and the
watcher. The family's scenarios themselves are its catalogue, in ``scenarios.py``. The tag
functions, the readers and the bundle helpers it composes with are here too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, override

from syn_grid.config.models.scenario.orb_models import NegOrbConf, TierOrbConf
from syn_grid.core.droid.digestion.digesters.direct_digester import NegDigester
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
from syn_grid.scenario.blocks.watcher import NoWatcher, Watcher
from syn_grid.scenario.entry import ScenarioEntry
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

T = TypeVar("T", bound=TierScenarioConf)


# ========================== #
# Single Tier Chain Scenario #
# ========================== #


@dataclass(frozen=True, kw_only=True)
class SingleTierChainEntry(ScenarioEntry[T]):
    """
    Goal/Single Tier Chain: collect every tier in order before the episode steps are used up.

    A scenario of the family states what differs between it and the others, and this composes
    everything the family has in common:

    - orb bundles: the tier bundle, scored by `scoring`, plus `_negative_bundle` when the
      negative orb is enabled.
    - scenario tag: `tag`'s, then the episode's steps as `_steps100`, then `_Neg` when the
      negative orb is enabled.
    - orb field: it holds every orb the bundles make.
    - observation: the slot count and the sort limit are the number of orbs the bundles make.
      The tier bound is `max_tier`.
    - termination: out of energy, a broken chain, a finished chain, or the clock.
    - watcher: the one `watcher` makes from the config, or else one that never acts.
    - hud: energy, score, steps left and the current chain length.
    - metrics: chains broken, progressed and completed, plus negatives eaten when the negative
      orb is enabled.
    - build_world: a `WorldRecipe` holding this scenario's world config, bundles and field
      size.

    It is generic in the scenario's config class, so `tag` and `watcher` are held to the class
    `conf_class` names, as delay's read its `delay_conf`.

    Attributes:
        scoring: the tier digester class, which is the scoring mode.
        tag: where a run sits on the scenario's axis, from its config.
        watcher: the watcher recipe for a config, for a scenario that acts after the world's
            step.
    """

    scoring: Callable[[int, float], TierDigester]
    tag: Callable[[T], str]
    watcher: Callable[[T], Callable[[], Watcher]] | None = None

    @override
    def _compose(self, scenario_name: str, scenario_conf: T) -> Scenario:
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

        orb_bundles = (_tier_bundle(world_conf.tier_orb_conf, self.scoring),)
        scenario_tag = f"{self.tag(scenario_conf)}_steps{world_conf.max_steps}"

        if world_conf.neg_orb_conf.enabled:
            scenario_tag += "_Neg"
            metrics += (Metric("negatives_eaten", _negatives_eaten),)
            grid_conf = world_conf.grid_conf
            negative_life_span = _reachable_life_span(
                grid_conf.grid_rows, grid_conf.grid_cols
            )
            orb_bundles += (
                _negative_bundle(world_conf.neg_orb_conf, negative_life_span),
            )

        # How many orbs the field holds. The world's field size, the observation's slots and its
        # sort limit all follow from it.
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

        make_watcher = (
            NoWatcher if self.watcher is None else self.watcher(scenario_conf)
        )

        return Scenario(
            scenario_name=scenario_name,
            scenario_tag=scenario_tag,
            hud=hud,
            metrics=metrics,
            observation=observation,
            termination=termination,
            build_world=WorldRecipe(world_conf, orb_bundles, max_active_orbs=orb_count),
            make_watcher=make_watcher,
        )


# ----------------- #
#       Tags        #
# ----------------- #

# Where a run sits on its scenario's axis, as text for the run id. Module-level, so that two
# entries holding the same function compare equal.


def grid_tag(scenario_conf: TierScenarioConf) -> str:
    """The grid size, as `5x5`."""

    grid_conf = scenario_conf.world_conf.grid_conf
    return f"{grid_conf.grid_rows}x{grid_conf.grid_cols}"


def tier_grid_tag(scenario_conf: TierScenarioConf) -> str:
    """The chain length and the grid size, as `tier3_5x5`."""

    return f"tier{scenario_conf.world_conf.tier_orb_conf.max_tier}_{grid_tag(scenario_conf)}"


def delay_tag(scenario_conf: TierDelayScenarioConf) -> str:
    """The delay, as `delay390`."""

    return f"delay{scenario_conf.delay_conf.steps}"


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
    return world.droid.get_digester(NegDigester).eaten


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

    A single tier chain scenario holds one negative orb, and how many is not config. It spawns,
    despawns at the end of `life_span` and returns elsewhere after its cool-down.
    """

    return OrbBundle(
        population=NegPopulation(
            orbs=1,
            reward=negative_conf.reward,
            cool_down=negative_conf.cool_down,
            life_span=life_span,
        ),
        make_digester=NegDigester,
    )
