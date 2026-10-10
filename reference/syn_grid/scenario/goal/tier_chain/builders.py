from __future__ import annotations

from functools import partial
from typing import cast

from syn_grid.config.models.scenarios.common_models import ScenarioConf

from syn_grid.config.models.scenarios.tier_chain_models import (
    TierScenarioConf,
    TierWorldConf,
)
from syn_grid.core.digestion.digestion import OrbDigester
from syn_grid.core.digestion.engine import DigestionEngine
from syn_grid.core.digestion.negative_digester import NegativeDigester
from syn_grid.core.digestion.tier_digester import ScoringMode, TierOrbDigester, ChainBroken, ChainCompleted, ChainProgressed
from syn_grid.core.droid.synergy_droid import SynergyDroid
from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.blocks.observation import ObservationRules
from syn_grid.scenario.blocks.population import OrbPopulation
from syn_grid.scenario.goal.tier_chain.population import TierChainPopulation
from syn_grid.scenario.blocks.spawning import (
    LeaveOrbFieldAlone,
    ReactivateAllOrbs,
    SpawningRules,
)
from syn_grid.scenario.goal.tier_chain.termination import TierChainTermination
from syn_grid.scenario.scenario import Scenario
from syn_grid.scenario.utils.helpers import neg_orb
from syn_grid.scenario.blocks.hud import HudStat
from syn_grid.scenario.blocks.metrics import Metric
from syn_grid.gymnasium.utils.episode_logging.keys import LogKey

# ----------------------- #
#  Tier Chain Scenarios   #
# ----------------------- #


def _tier_chain_scenario(
    scenario_name: str,
    scenario_tag: str,
    scenario_conf: TierScenarioConf,
    orb_population: OrbPopulation,
    scoring_mode: ScoringMode,
    delay_on_consume: int | None = None,
    max_score: int | None = None,
) -> Scenario:
    """
    Goal/Tier Chain: collect every tier in order before the episode steps are used up.

    The chain is the objective, so the world's shape follows from it. The pool is one orb per tier,
    all present from the first step, and nothing expires off the grid during the episode.
    """

    world_conf = scenario_conf.world_conf
    tier_conf = world_conf.orb_conf.tier
    obs_handler_conf = scenario_conf.obs_conf.observation_handler_conf

    spawning = SpawningRules(
        spawn_max_on_episode_start=True,
        max_active_orbs=tier_conf.max_tier, # Max tier defines chain length and pool size.
        tier_orb_expires=False,
        _delay_on_consume=delay_on_consume,
        _after_action=ReactivateAllOrbs() if delay_on_consume is not None else LeaveOrbFieldAlone(),
    )

    observation = ObservationRules(
        perception=obs_handler_conf.perception,
        max_steps=obs_handler_conf.max_steps,
        max_score=max_score,
        observation_slot_count=tier_conf.max_tier,
        sort_limit=tier_conf.max_tier,
        max_tier=tier_conf.max_tier,
    )

    termination = TierChainTermination(world_conf.droid_conf.timeout_penalty)

    # hud = (HudStat("current tier chain", _chain_progress),)

    # metrics = (
    #     Metric(LogKey.CHAINS_BROKEN, _chains_broken),
    #     Metric(LogKey.CHAIN_PROGRESSED, _chains_progressed),
    #     Metric(LogKey.CHAINS_COMPLETED, _chains_completed),
    # )

    return Scenario(
        scenario_name,
        scenario_tag,
        observation,
        termination,
        build_world=partial(
            _build_tier_chain_world, world_conf, orb_population, scoring_mode, spawning
        )
    )


# ---------------- #
#     Helpers      #
# ---------------- #

def _chain_progress(world: GridWorld) -> float:
    return world.droid.digestion_engine.get(TierOrbDigester).chained_tiers

def _chains_broken(world: GridWorld) -> float:
    return world.droid.digestion_engine.count(ChainBroken)

def _chains_progressed(world: GridWorld) -> float:
    return world.droid.digestion_engine.count(ChainProgressed)

def _chains_completed(world: GridWorld) -> float:
    return world.droid.digestion_engine.count(ChainCompleted)


def _build_tier_chain_world(
    world_conf: TierWorldConf,
    orb_population: OrbPopulation,
    scoring_mode: ScoringMode,
    spawning: SpawningRules,
) -> GridWorld:
    """
    Build one tier-chain world.

    Called once per environment, so everything that holds episode state -- the digesters, the
    droid, the orbs -- is created here rather than shared through the scenario.
    """

    droid_conf = world_conf.droid_conf
    orb_conf = world_conf.orb_conf
    tier_conf = orb_conf.tier
    grid_conf = world_conf.grid_conf
    grid_dimensions = (grid_conf.grid_rows, grid_conf.grid_cols)

    # Which digesters exist follows from which orbs the config enables.
    digesters: list[OrbDigester] = [
        TierOrbDigester(
            scoring_mode,
            tier_conf.max_tier,
            droid_conf.completion_reward,
            droid_conf.chain_break_penalty,
        )
    ]

    if orb_conf.negative is not None:
        digesters.append(NegativeDigester())

    return GridWorld(
        grid_dimensions,
        SynergyDroid(droid_conf, grid_dimensions, DigestionEngine(digesters)),
        orb_population,
        spawning,
    )


# ============ #
#   Builders   #
# ============ #


def build_tier_chain_spatial(scenario_name: str, scenario_conf: ScenarioConf) -> Scenario:
    """Tier Chain, Spatial: the chain is laid out across the grid, but the droid only sees a 3x3
    window around itself.

    This scenario tests spatial memory under partial observability. The droid may walk past a later
    tier while collecting an earlier one, and once it's out of the window, it's out of sight. It's
    like searching a dark room with a small flashlight: without remembering what the beam already
    passed over, you keep searching the same spots. The task never changes, only the grid size
    does, so grid size works as a standalone difficulty axis — giving the droid a bigger area to
    map.
    """

    scenario_conf = cast(TierScenarioConf, scenario_conf)
    grid_conf = scenario_conf.world_conf.grid_conf
    scenario_tag = f"{grid_conf.grid_rows}x{grid_conf.grid_cols}{neg_orb(scenario_conf)}"

    # Only the completed chain pays, and the digester pays it, so the orbs are
    # worth nothing on their own.
    orb_population = TierChainPopulation(
        scenario_conf.world_conf.orb_conf.tier.max_tier,
        base_reward=0.0,
        growth_factor=1.0,
        linear_reward_growth=True,
        cool_down=0,
    )

    return _tier_chain_scenario(
        scenario_name,
        scenario_tag,
        scenario_conf,
        orb_population,
        ScoringMode.MAX_TIER
    )


def build_tier_chain_scaling_sparse(name: str, scenario_conf: ScenarioConf) -> Scenario:
    ...


def build_tier_chain_scaling_dense(name: str, scenario_conf: ScenarioConf) -> Scenario:
    ...


def build_tier_chain_delay(name: str, scenario_conf: ScenarioConf) -> Scenario:
   ...