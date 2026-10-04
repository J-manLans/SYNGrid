
from pydantic import model_validator

from syn_grid.config.models.global_models import Scenario_name
from syn_grid.config.models.scenario.droid_models import TierDroidConf
from syn_grid.config.models.scenario.orb_models import TierOrbPoolConf
from syn_grid.config.models.scenario.scenario_common import ScenarioConf, WorldConf

# ======================= #
#   World Configuration   #
# ======================= #




class TierWorldConf(WorldConf, frozen=True, extra="forbid", strict=True):
    droid_conf: TierDroidConf
    orb_conf: TierOrbPoolConf

    @model_validator(mode="after")
    def validate_config(self):
        if self.orb_conf.tier.max_tier >= (
            self.grid_conf.grid_rows * self.grid_conf.grid_cols
        ):
            raise ValueError(
                "max_tier can't be higher than number of cells in the grid, "
                "there will be no space for orbs"
            )

        return self


# ============================= #
#    Top-Level Configurations   #
# ============================= #



class TierScenarioConf(ScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf


# ======================= #
#        Constants        #
# ======================= #

SCENARIO_MODELS = {
    Scenario_name.GOAL_TIER_CHAIN_SPATIAL: TierScenarioConf,
    Scenario_name.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: TierScenarioConf,
    Scenario_name.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: TierScenarioConf,
    Scenario_name.GOAL_TIER_CHAIN_DELAY: TierScenarioConf,
}
