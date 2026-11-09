"""
The single tier chain family's scenarios.

One entry per scenario, each with what it tests above it. An entry is the whole scenario: its full
config model, the perceptions with its visibility, its lock, and what the family composes it from.
The registry collects this catalogue with every other family's.
"""

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import Perception
from syn_grid.core.droid.digestion.digesters.tier_chain.max_tier_digester import (
    MaxTierDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)
from syn_grid.scenario.entry import Lock, ScenarioEntry
from syn_grid.scenario.families.goal.single_tier_chain.builders import (
    SingleTierChainEntry,
    delay_tag,
    grid_tag,
    tier_grid_tag,
)
from syn_grid.scenario.families.goal.single_tier_chain.config import (
    DelayScenarioConf,
    ScalingScenarioConf,
    SpatialScenarioConf,
    TierDelayScenarioConf,
    TierScenarioConf,
)
from syn_grid.scenario.families.goal.single_tier_chain.watcher import (
    DelayWatcherRecipe,
)
from syn_grid.utils.paths_util import get_syn_grid_path

_BASE_CONFIGS = get_syn_grid_path(
    "scenario", "families", "goal", "single_tier_chain", "base_configs"
)

SINGLE_TIER_CHAIN_SCENARIOS: dict[ScenarioName, ScenarioEntry] = {
    # Spatial: the complete tier chain is laid out across the grid, but the droid only sees a 3x3
    # window around itself.
    # Difficulty axis: the grid size.
    # Scoring: max-tier. Only the completed chain pays, and it pays what its orbs are worth
    # together.
    #
    # It tests spatial memory under partial observability. The droid may walk past a later tier
    # while collecting an earlier one, and once it is out of the window, it is out of sight. It is
    # like searching a dark room with a small flashlight: without remembering what the beam
    # already passed over, you keep searching the same spots. The task never changes, only the
    # grid size does, so grid size works as a standalone difficulty axis, giving the droid a
    # bigger area to map.
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL: SingleTierChainEntry(
        conf_class=TierScenarioConf,
        perceptions=(Perception.VECTOR_FOG_OF_WAR,),
        lock=Lock(SpatialScenarioConf, _BASE_CONFIGS / "spatial.yaml"),
        scoring=MaxTierDigester,
        tag=grid_tag,
    ),
    # Scaling Sparse: the complete tier chain is laid out across the grid and the droid sees all
    # of it.
    # Difficulty axis: the chain length and the grid size.
    # Scoring: max-tier. Only the completed chain pays, and it pays what its orbs are worth
    # together.
    #
    # It tests how the task scales with the length of the chain. Nothing is hidden, so the droid
    # never has to remember where an orb was. What grows is the order it has to learn: the longer
    # the chain, the longer it goes without being paid, and the more steps a wrong turn costs.
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_SPARSE: SingleTierChainEntry(
        conf_class=TierScenarioConf,
        perceptions=(Perception.VECTOR_MARKOVIAN,),
        lock=Lock(ScalingScenarioConf, _BASE_CONFIGS / "scaling_sparse.yaml"),
        scoring=MaxTierDigester,
        tag=tier_grid_tag,
    ),
    # Scaling Dense: the complete tier chain is laid out across the grid and the droid sees all
    # of it.
    # Difficulty axis: the chain length and the grid size.
    # Scoring: threshold. A growing chain pays nothing. When it completes, or breaks, it pays what
    # the orbs eaten so far are worth together, and a break also costs the chain break penalty.
    #
    # It asks whether it is the silence that stops learning. The world is scaling sparse's, but a
    # chain that breaks halfway still pays for the part it got right, so the droid is told when
    # it is on the right track.
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_DENSE: SingleTierChainEntry(
        conf_class=TierScenarioConf,
        perceptions=(Perception.VECTOR_MARKOVIAN,),
        lock=Lock(ScalingScenarioConf, _BASE_CONFIGS / "scaling_dense.yaml"),
        scoring=ThresholdDigester,
        tag=tier_grid_tag,
    ),
    # Delay: the complete tier chain is laid out across the grid and the droid sees all of it,
    # but eating a tier orb freezes the other tier orbs for the delay.
    # Difficulty axis: the delay.
    # Scoring: max-tier. Only the completed chain pays, and it pays what its orbs are worth
    # together.
    #
    # It tests whether an agent can bridge a gap it cannot see the end of. After each correct orb
    # the rest of the chain is gone from view for the delay, and returns in the same cells.
    # Nothing in the observation counts the gap down, so the droid has to carry what it knew
    # across it. The negative orb, when enabled, is not frozen.
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY: SingleTierChainEntry(
        conf_class=TierDelayScenarioConf,
        perceptions=(Perception.VECTOR_MARKOVIAN,),
        lock=Lock(DelayScenarioConf, _BASE_CONFIGS / "delay.yaml"),
        scoring=MaxTierDigester,
        tag=delay_tag,
        watcher=DelayWatcherRecipe.from_conf,
    ),
}
