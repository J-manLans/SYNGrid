from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from syn_grid.config.models.global_models import Scenario_name
from syn_grid.config.models.orb_models import NegativeConf, TierConf
from syn_grid.config.models.droid_models import TierOrbDroidConf

# ======================= #
#      Helper Types       #
# ======================= #


# TODO: when everything is working, see if this one can be used instead of explicitly stating
# the keywords in each class. think this can be good for the tests, since they can override the
# frozen keyword.
class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# ======================= #
#   World Configuration   #
# ======================= #


class GridConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_rows: int
    grid_cols: int


class DroidConf(BaseModel, frozen=True, extra="forbid", strict=True):
    starting_score: float
    step_penalty: float
    boundary_penalty: float
    # TODO: remember to check whether this one shall be used, and remember — less is more
    reward_multiplier: float

    @model_validator(mode="after")
    def validate_penalties(self):
        bad = [
            name
            for name in type(self).model_fields
            if name.endswith("_penalty") and getattr(self, name) > 0
        ]
        if bad:
            raise ValueError(f"{', '.join(bad)} must be 0 or negative")
        return self


class OrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    max_active_orbs: int
    negative: NegativeConf | None = None
    tier: TierConf | None = None


# ======================= #
#    Obs Configuration    #
# ======================= #


class ObservationHandlerConf(BaseModel, frozen=True, extra="forbid", strict=True):
    perception: str
    max_steps: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.perception not in [
            "vector_markovian_easy",
            "vector_markovian",
            "vector_fog_of_war",
            "composite_markovian",
            "composite_fully_pomdp",
            "composite_grid_markovian",
            "grid_pixel",
        ]:
            raise ValueError("The value of difficulty is not allowed")
        return self


# === PerceptionConf START === #


class EnabledOrbsConf(BaseModel, frozen=True, extra="forbid", strict=True):
    neg_enabled: bool
    tier_enabled: bool


class PerceptionConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """How an observation is encoded.

    The world-derived counts are not here. How many orb slots an observation
    holds and how far its tier channel reaches are properties of the world,
    which the scenario already describes; this block used to carry a second
    copy of each that had to be kept in step by hand through YAML anchors, and
    an observation space was only fully knowable by reading four files at once.
    """

    max_score: int
    max_steps: int
    grid_rows: int
    grid_cols: int
    include_timer: bool
    enabled_orbs: EnabledOrbsConf
    tiers: int


# === PerceptionConf END === #

# ======================= #
#   Domain Config Blocks  #
# ======================= #


class WorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_conf: GridConf


class TierWorldConf(WorldConf, frozen=True, extra="forbid", strict=True):
    droid_conf: TierOrbDroidConf
    orb_conf: OrbConf

    @model_validator(mode="after")
    def validate_config(self):
        tier_conf = self.orb_conf.tier
        if tier_conf is None:
            raise ValueError("TierWorldConf requires tier orb configuration")

        if tier_conf.max_tier >= (
            self.grid_conf.grid_rows * self.grid_conf.grid_cols
        ):
            raise ValueError(
                "max_tier can't be higher than number of cells in the grid, "
                "there will be no space for orbs"
            )

        return self


class ObsConf(BaseModel, frozen=True, extra="forbid", strict=True):
    observation_handler_conf: ObservationHandlerConf
    perception_conf: PerceptionConf


# ============================= #
#    Top-Level Configurations   #
# ============================= #

class ScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    ...


class TierScenarioConf(ScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf
    obs_conf: ObsConf


# ======================= #
#        Constants        #
# ======================= #

SCENARIO_MODELS = {
    Scenario_name.GOAL_TIER_CHAIN_SPATIAL: TierScenarioConf,
    Scenario_name.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: TierScenarioConf,
    Scenario_name.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: TierScenarioConf,
    Scenario_name.GOAL_TIER_CHAIN_DELAY: TierScenarioConf,
}
