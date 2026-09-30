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

from syn_grid.config.models import ObsConfig, ScoringMode, WorldConfig
from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.population import (
    OrbPopulation,
    TierChainPopulation,
    WeightedPopulation,
)
from syn_grid.scenario.rules.spawning import (
    LeaveFieldAlone,
    ReactivateAllOrbs,
    RefillOrbPool,
    SpawningRules,
)
from syn_grid.scenario.rules.termination import (
    ContinuousTermination,
    GoalTermination,
    TerminationRules,
)
from syn_grid.scenario.scenario import Scenario, ScenarioType

ScenarioBuilder = Callable[[WorldConfig, ObsConfig], Scenario]


# ===================== #
#    Goal Scenarios     #
# ===================== #


# --------------------- #
#    Tier Scenarios     #
# --------------------- #


def _tier_chain(
    name: str,
    world_conf: WorldConfig,
    obs_conf: ObsConfig,
    *,
    delay: bool,
    curriculum: bool,
) -> Scenario:
    """
    Goal/Tier Chain: collect every tier in order before the episode steps are used up.

    The chain is the objective, so the world's shape follows from it. The pool is one orb per tier,
    all present from the first step, and nothing expires off the grid during the episode.
    """

    grid_conf = world_conf.grid_conf
    perception_conf = obs_conf.perception_conf

    if grid_conf.de_spawn_tiers:
        raise ValueError(
            f"Scenario '{name}' cannot de-spawn tiers: an orb "
            "that de-spawns because of a timer would ruin the chain. Disable de_spawn_tiers."
        )

    # A chain's orb count is its length. The old schema expressed this by
    # overwriting max_active_orbs with max_tier inside a validator, so the YAML
    # value was routinely a lie. The scenario states it outright.
    max_active_orbs = grid_conf.max_tier

    if grid_conf.max_tier >= grid_conf.grid_rows * grid_conf.grid_cols:
        raise ValueError(
            "max_tier can't be higher than number of cells in the grid, there will "
            "be no space for orbs"
        )

    population: OrbPopulation = TierChainPopulation(
        grid_conf.max_tier, world_conf.tier_orb_conf
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
        observation_slot_count=perception_conf.tiers
        if curriculum
        else grid_conf.max_tier,
        sort_limit=grid_conf.max_tier,
        max_tier=grid_conf.max_tier,
    )

    termination = GoalTermination(
        timeout_penalty=world_conf.droid_conf.timeout_penalty,
        delay=delay,
        scoring=world_conf.tier_orb_conf.scoring,
        curriculum=curriculum,
    )

    return Scenario(
        name,
        ScenarioType.GOAL,
        population,
        spawning,
        observation,
        termination,
    )


def build_tier_chain_spatial(world_conf: WorldConfig, obs_conf: ObsConfig) -> Scenario:
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

    return _tier_chain(
        "goal_tier_chain_spatial", world_conf, obs_conf, delay=False, curriculum=True
    )


def build_tier_chain_delay(world_conf: WorldConfig, obs_conf: ObsConfig) -> Scenario:
    """
    Tier Chain, Delay: consuming an orb puts the whole field on cooldown, first after that cooldown
    the orb spawns back in.

    This scenario tests temporal delay with empty visual feedback. The reward only arrives once the
    full chain is done, so it has to travel back across every silent stretch to reach the first
    correct orb, weakening with each step. It's like giving a dog its treat an hour after the
    trick: by then the link is faint. The task never changes, only the gap does, so delay works as
    a standalone difficulty axis.
    """

    return _tier_chain(
        "goal_tier_chain_delay", world_conf, obs_conf, delay=True, curriculum=False
    )


def build_tier_chain_scaling_sparse(
    world_conf: WorldConfig, obs_conf: ObsConfig
) -> Scenario:
    """
    Tier Chain, Scaling (dense): same layout as the sparse variant, but each correctly consumed orb
    earns reward, paid out when the chain breaks or completes.

    This scenario uses a dense scoring mode to strengthen the reward signal, so every correct orb
    counts even if the chain breaks before the end. (Same lock as the sparse variant, but now it
    clicks for every digit you get right.) Can be run at the tier where the sparse variant stops
    learning to confirm the task itself is learnable. If performance recovers, sparsity was the
    limit, not the chain.
    """

    _require_scoring(world_conf, ScoringMode.MAX_TIER, "tier_chain_scaling_sparse")
    return _tier_chain(
        "goal_tier_chain_scaling_sparse",
        world_conf,
        obs_conf,
        delay=False,
        curriculum=False,
    )


def build_tier_chain_scaling_dense(
    world_conf: WorldConfig, obs_conf: ObsConfig
) -> Scenario:
    """
    Tier Chain, Tier Scaling (dense): long chains under threshold scoring.

    Threshold scoring accumulates reward as the chain grows and pays it out on completion or hands
    back a partial amount when the chain breaks, which is what makes a long chain worth attempting.
    """

    _require_scoring(world_conf, ScoringMode.THRESHOLD, "tier_chain_scaling_dense")
    return _tier_chain(
        "goal_tier_chain_scaling_dense",
        world_conf,
        obs_conf,
        delay=False,
        curriculum=False,
    )


# ====================== #
#  Continuous Scenarios  #
# ====================== #


def _continuous(
    name: str,
    world_conf: WorldConfig,
    obs_conf: ObsConfig,
    *,
    delay: bool,
) -> Scenario:
    """Continuous: no objective, orbs keep coming, the clock is the only end.

    Orbs are drawn from a weighted pool and the field refills one at a time, so
    the agent is choosing what to spend its steps on rather than following a
    fixed sequence. Tier orbs may or may not expire off the board; that is a
    difficulty knob, and it is the one thing a tier chain is not allowed to do.
    """

    grid = world_conf.grid_conf
    max_active_orbs = grid.max_active_orbs

    if max_active_orbs <= 0:
        raise ValueError("max_active_orbs should be larger than 0")

    population: OrbPopulation = WeightedPopulation(
        world_conf.orb_factory_conf,
        world_conf.negative_orb_conf,
        world_conf.tier_orb_conf,
    )

    spawning = SpawningRules(
        fill_pool_on_reset=False,
        max_active_orbs=max_active_orbs,
        tier_orb_expires=grid.de_spawn_tiers,
        delay_on_consume=delay,
        after_action=RefillOrbPool(),
    )

    # The curriculum setting does not reach a continuous world: the slot count
    # is max_active_orbs either way, so there is nothing for it to change.
    observation = ObservationRules(
        observation_slot_count=max_active_orbs,
        sort_limit=max_active_orbs,
        max_tier=grid.max_tier,
    )

    termination: TerminationRules = ContinuousTermination(
        scoring=world_conf.tier_orb_conf.scoring
    )

    return Scenario(
        name=name,
        type=ScenarioType.CONTINUOUS,
        population=population,
        spawning=spawning,
        observation=observation,
        termination=termination,
    )


def build_continuous(world_conf: WorldConfig, obs_conf: ObsConfig) -> Scenario:
    """Continuous: the orb field is always available."""

    return _continuous("continuous", world_conf, obs_conf, delay=False)


def build_continuous_delay(world_conf: WorldConfig, obs_conf: ObsConfig) -> Scenario:
    """Continuous with delay: consuming an orb puts the whole field on cooldown.

    Distinct from a tier chain with delay in what the cooldown costs. Here the
    field refills from the weighted pool as the cooldown lapses, so a
    consumption costs a stretch of empty grid rather than a broken chain.
    """

    return _continuous("continuous_delay", world_conf, obs_conf, delay=True)


# ============ #
#    Helpers    #
# ============ #


def _require_scoring(
    world_conf: WorldConfig, required: ScoringMode, scenario: str
) -> None:
    actual = world_conf.tier_orb_conf.scoring
    if actual is not required:
        raise ValueError(
            f"Scenario '{scenario}' is defined by {required.value} scoring but the "
            f"config selects {actual.value}. Set tier_orb_conf.scoring to "
            f"{required.value!r}."
        )


# ============ #
#   Registry    #
# ============ #


SCENARIOS: dict[str, ScenarioBuilder] = {
    "goal_tier_chain_spatial": build_tier_chain_spatial,
    "goal_tier_chain_delay": build_tier_chain_delay,
    "goal_tier_chain_scaling_dense": build_tier_chain_scaling_dense,
    "goal_tier_chain_scaling_sparse": build_tier_chain_scaling_sparse,
    "continuous": build_continuous,
}


def build_scenario(name: str, world_conf: WorldConfig, obs_conf: ObsConfig) -> Scenario:
    """Resolve a scenario name into the rules that define it."""

    try:
        builder = SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}"
        ) from None

    return builder(world_conf, obs_conf)
