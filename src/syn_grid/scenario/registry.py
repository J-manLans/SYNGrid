"""
Scenario selection.

This module is the whole of "which scenario is this". A config names one, a
builder composes the rules for it, and nothing downstream re-derives identity
from booleans. Registering a new scenario means adding a builder here; the
environment, the world, the orb factory and the perceptions do not change.

The names are the domain's, not the config's. ``single_chain_mode`` described
a mechanism; "tier chain" describes the scenario that uses it, which is why the
old flag name is not carried forward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from syn_grid.config.models.global_models import Scenario_name
from syn_grid.config.models.scenario.orb_models import ScoringMode
from syn_grid.config.models.scenario.scenario_models import (
    ScenarioConf,
    TierScenarioConf,
)
from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.population import (
    OrbPopulation,
    TierChainPopulation,
)
from syn_grid.scenario.rules.spawning import (
    LeaveFieldAlone,
    ReactivateAllOrbs,
    SpawningRules,
)
from syn_grid.scenario.rules.termination import (
    GoalTermination,
)
from syn_grid.scenario.scenario import Scenario, ScenarioType

ScenarioBuilder = Callable[[Scenario_name, ScenarioConf], Scenario]


# ===================== #
#    Goal Scenarios     #
# ===================== #


# --------------------- #
#    Tier Scenarios     #
# --------------------- #


def _tier_chain_scenario(
    name: str,
    scenario_tag: str,
    scenario_conf: TierScenarioConf,
    *,
    delay: bool,
    curriculum: bool,
) -> Scenario:
    """
    Goal/Tier Chain: collect every tier in order before the episode steps are used up.

    The chain is the objective, so the world's shape follows from it. The pool is one orb per tier,
    all present from the first step, and nothing expires off the grid during the episode.
    """

    # A chain's orb count is its length. The old schema expressed this by
    # overwriting max_active_orbs with max_tier inside a validator, so the YAML
    # value was routinely a lie. The scenario states it outright.
    max_active_orbs = scenario_conf.world_conf.orb_conf.tier.max_tier

    population: OrbPopulation = TierChainPopulation(
        scenario_conf.world_conf.orb_conf.tier.max_tier, scenario_conf.world_conf.orb_conf.tier
    )

    spawning = SpawningRules(
        fill_pool_on_reset=True,
        max_active_orbs=max_active_orbs,
        tier_orb_expires=False,
        delay_on_consume=delay,
        after_action=ReactivateAllOrbs() if delay else LeaveFieldAlone(),
    )

    # Two different counts, and deliberately so. Under curriculum the
    # observation is sized for `tiers` slots even though the chain is shorter,
    # so the agent always sees the same width; the distance sort that fills
    # those slots has never followed that setting, so the trailing slots stay
    # zero. See ObservationRules.
    observation = ObservationRules(
        observation_slot_count=scenario_conf.obs_conf.perception_conf.tiers
        if curriculum
        else scenario_conf.world_conf.orb_conf.tier.max_tier,
        sort_limit=scenario_conf.world_conf.orb_conf.tier.max_tier,
        max_tier=scenario_conf.world_conf.orb_conf.tier.max_tier,
    )

    termination = GoalTermination(
        timeout_penalty=scenario_conf.world_conf.droid_conf.timeout_penalty,
        delay=delay,
        scoring=scenario_conf.world_conf.orb_conf.tier.scoring,
        curriculum=curriculum,
    )

    return Scenario(
        name,
        scenario_tag,
        ScenarioType.GOAL,
        scenario_conf.obs_conf.observation_handler_conf.perception,
        (
            scenario_conf.world_conf.grid_conf.grid_rows,
            scenario_conf.world_conf.grid_conf.grid_cols
        ),
        population,
        spawning,
        observation,
        termination,
    )

# ============ #
#   Builders   #
# ============ #


def build_tier_chain_spatial(name: str, scenario_conf: ScenarioConf) -> Scenario:
    """
    Tier Chain, Spatial: the chain is laid out across the grid, but the droid only
    sees a 3x3 window around itself.

    This scenario tests spatial memory under partial observability. The droid may
    walk past a later tier while collecting an earlier one, and once it's out of
    the window, it's out of sight. It's like searching a dark room with a small
    flashlight: without remembering what the beam already passed over, you keep
    searching the same spots. The task never changes, only the grid size does, so
    grid size works as a standalone difficulty axis.
    """

    scenario_conf = cast(TierScenarioConf, scenario_conf)
    grid_conf = scenario_conf.world_conf.grid_conf
    neg_orb = "_Neg" if scenario_conf.world_conf.orb_conf.negative else ""
    scenario_tag = f"{grid_conf.grid_rows}x{grid_conf.grid_cols}{neg_orb}"

    return _tier_chain_scenario(name, scenario_tag, scenario_conf, delay=False, curriculum=True)


def build_tier_chain_scaling_sparse(name: str, scenario_conf: ScenarioConf) -> Scenario:
    ...


def build_tier_chain_scaling_dense(name: str, scenario_conf: ScenarioConf) -> Scenario:
    ...


def build_tier_chain_delay(name: str, scenario_conf: ScenarioConf) -> Scenario:
   ...

# ============ #
#    Helpers   #
# ============ #

def neg_orb(self, scenario_conf: ScenarioConf) -> str:
    return "_Neg" if scenario_conf.world_conf.orb_conf.negative else ""



def _require_scoring(
    scenario_conf: TierScenarioConf, required: ScoringMode, scenario: str
) -> None:
    actual = scenario_conf.world_conf.orb_conf.tier.scoring
    if actual is not required:
        raise ValueError(
            f"Scenario '{scenario}' is defined by {required.value} scoring but the "
            f"config selects {actual.value}. Set tier_orb_conf.scoring to "
            f"{required.value!r}."
        )


# ============ #
#   Registry   #
# ============ #


SCENARIO_BUILDERS: dict[Scenario_name, ScenarioBuilder] = {
    Scenario_name.GOAL_TIER_CHAIN_SPATIAL: build_tier_chain_spatial,
    Scenario_name.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: build_tier_chain_scaling_sparse,
    Scenario_name.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: build_tier_chain_scaling_dense,
    Scenario_name.GOAL_TIER_CHAIN_DELAY: build_tier_chain_delay,
}


def build_scenario(scenario: Scenario_name, scenario_conf: ScenarioConf) -> Scenario:
    """Resolve a scenario name into the rules that define it."""

    try:
        builder = SCENARIO_BUILDERS[scenario]
    except KeyError:
        raise KeyError(
            f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIO_BUILDERS)}"
        ) from None

    return builder(scenario, scenario_conf)
