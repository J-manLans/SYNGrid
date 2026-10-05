"""Scenario selection and composition.

Each registered scenario is built here as a complete composition. The builder
chooses the concrete simulation components; GridWorld runs the mechanics of
the components it is given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.tier_chain_models import ScoringMode
from syn_grid.config.models.common_models import ScenarioConf
from syn_grid.config.models.tier_chain_models import TierScenarioConf
from syn_grid.core.digestion.new_builder import build_digestion
from syn_grid.core.droid.synergy_droid import SynergyDroid
from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.new_scenario import Scenario
from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.population import (
    OrbPopulation,
    TierChainPopulation,
)
from syn_grid.scenario.rules.spawning import LeaveFieldAlone, SpawningRules
from syn_grid.scenario.rules.termination import GoalTermination

ScenarioBuilder = Callable[[ScenarioName, ScenarioConf], Scenario]


def build_tier_chain_spatial(
    name: str, scenario_conf: ScenarioConf
) -> Scenario:
    """Build the spatial tier-chain scenario."""

    scenario_conf = cast(TierScenarioConf, scenario_conf)

    grid_conf = scenario_conf.world_conf.grid_conf
    droid_conf = scenario_conf.world_conf.droid_conf
    orb_conf = scenario_conf.world_conf.orb_conf
    tier_conf = orb_conf.tier

    neg_orb = "_Neg" if orb_conf.negative else ""
    scenario_tag = f"{grid_conf.grid_rows}x{grid_conf.grid_cols}{neg_orb}"

    population: OrbPopulation = TierChainPopulation(
        tier_conf.max_tier,
        tier_conf,
    )

    # Which digesters exist follows from the orb config, not from this builder.
    digestion = build_digestion(orb_conf, droid_conf)

    droid = SynergyDroid(
        droid_conf,
        digestion,
    )

    spawning = SpawningRules(
        fill_pool_on_reset=True,
        max_active_orbs=tier_conf.max_tier,
        tier_orb_expires=False,
        delay_on_consume=False,
        after_action=LeaveFieldAlone(),
    )

    world = GridWorld(
        grid_conf=grid_conf,
        droid=droid,
        population=population,
        spawning=spawning,
    )

    observation = ObservationRules(
        observation_slot_count=tier_conf.max_tier,
        sort_limit=tier_conf.max_tier,
        max_tier=tier_conf.max_tier,
    )

    termination = GoalTermination(
        timeout_penalty=droid_conf.timeout_penalty,
        delay=False,
        scoring=tier_conf.scoring,
        curriculum=True,
    )

    return Scenario(
        name=name,
        tag=scenario_tag,
        world=world,
        observation=observation,
        termination=termination,
    )


def build_tier_chain_scaling_sparse(
    name: str, scenario_conf: ScenarioConf
) -> Scenario:
    ...


def build_tier_chain_scaling_dense(
    name: str, scenario_conf: ScenarioConf
) -> Scenario:
    ...


def build_tier_chain_delay(
    name: str, scenario_conf: ScenarioConf
) -> Scenario:
    ...


def _require_scoring(
    scenario_conf: TierScenarioConf,
    required: ScoringMode,
    scenario: str,
) -> None:
    actual = scenario_conf.world_conf.orb_conf.tier.scoring
    if actual is not required:
        raise ValueError(
            f"Scenario '{scenario}' is defined by {required.value} scoring but "
            f"the config selects {actual.value}. Set tier_orb_conf.scoring to "
            f"{required.value!r}."
        )


SCENARIO_BUILDERS: dict[ScenarioName, ScenarioBuilder] = {
    ScenarioName.GOAL_TIER_CHAIN_SPATIAL: build_tier_chain_spatial,
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: build_tier_chain_scaling_sparse,
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: build_tier_chain_scaling_dense,
    ScenarioName.GOAL_TIER_CHAIN_DELAY: build_tier_chain_delay,
}


def build_scenario(
    scenario: ScenarioName,
    scenario_conf: ScenarioConf,
) -> Scenario:
    """Resolve a scenario name into its fully composed scenario."""

    try:
        builder = SCENARIO_BUILDERS[scenario]
    except KeyError:
        raise KeyError(
            f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIO_BUILDERS)}"
        ) from None

    return builder(scenario, scenario_conf)
