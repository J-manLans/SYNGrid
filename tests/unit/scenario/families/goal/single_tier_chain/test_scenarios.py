"""What each scenario of the single tier chain family's catalogue is: how it is scored and what it may see."""

import pytest

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import Perception
from syn_grid.core.droid.digestion.digesters.tier_chain.max_tier_digester import (
    MaxTierDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#       Tests       #
# ================= #


class TestCatalogue:
    """Each scenario's scoring mode and the perceptions with its visibility, as its entry states."""

    @pytest.mark.parametrize(
        ("name", "scoring", "perceptions"),
        [
            (
                ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL,
                MaxTierDigester,
                (Perception.VECTOR_FOG_OF_WAR,),
            ),
            (
                ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_SPARSE,
                MaxTierDigester,
                (Perception.VECTOR_MARKOVIAN,),
            ),
            (
                ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_DENSE,
                ThresholdDigester,
                (Perception.VECTOR_MARKOVIAN,),
            ),
            (
                ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY,
                MaxTierDigester,
                (Perception.VECTOR_MARKOVIAN,),
            ),
        ],
    )
    def test_its_world_is_scored_by_its_mode_and_it_sees_only_with_its_visibility(
        self,
        name: ScenarioName,
        scoring: type[TierDigester],
        perceptions: tuple[Perception, ...],
    ):
        entry = SCENARIOS[name]

        world = entry.build(name.value, entry.base_conf()).build_world()

        assert type(world.droid.get_digester(TierDigester)) is scoring
        assert entry.perceptions == perceptions
