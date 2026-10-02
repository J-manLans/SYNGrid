from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


# TODO: when everything is working, see if this one can be used instead of explicitly stating
# the keywords in each class. think this can be good for the tests, since they can override the
# frozen keyword.
class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ScoringMode(Enum):
    """
    How a tier chain's reward is paid out.

    One value rather than three mutually exclusive booleans. The booleans had
    to be validated against each other on every load, and a fourth copy of the
    question lived in the world config where the digestion engine could not see
    it and episode termination read the wrong one.
    """

    STEP_WISE = "step_wise"
    THRESHOLD = "threshold"
    MAX_TIER = "max_tier"


# ======================= #
#   World Configuration   #
# ======================= #


class GridConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_rows: int
    grid_cols: int
    max_active_orbs: int


# === DroidConf START === #


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


class GoalDroidConf(DroidConf, frozen=True, extra="forbid", strict=True):
    timeout_penalty: float


class TierOrbDroidConf(GoalDroidConf, frozen=True, extra="forbid", strict=True):
    chain_break_penalty: float
    tier_consumption_penalty: float


# === DroidConf END === #


# === OrbFactoryConf START === #


class OrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    enabled: bool
    weight: int


class TypesConf(BaseModel, frozen=True, extra="forbid", strict=True):
    negative: OrbConf
    tier: OrbConf


class OrbFactoryConf(BaseModel, frozen=True, extra="forbid", strict=True):
    types: TypesConf


# === OrbFactoryConf END === #


class NegativeConf(BaseModel, frozen=True, extra="forbid", strict=True):
    reward: float
    cool_down: int


class TierConf(BaseModel, frozen=True, extra="forbid", strict=True):
    base_reward: float
    cool_down: int
    growth_factor: float
    max_tier: int
    de_spawn_tiers: bool
    linear_reward_growth: bool
    scoring: ScoringMode
    # TODO: figure out how this one should be handled, it's specific for the delay scenario, why
    # have it as a value anywhere else? This might be true for other variables as well. For
    # example, no scenario with the negative orb exists yet. So does it have a meaning being here?
    # Like should orb factory be in the global config instead?
    delay: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.max_tier <= 0:
            raise ValueError("max_tier should be larger than 0")
        if self.growth_factor <= 0:
            raise ValueError(f"{self.growth_factor} must be a positive value.")

        return self


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
    # TODO: this is for next session, decide how to handle it. Need to look into the scenario
    # package as well as the orb factory itself
    orb_factory_conf: OrbFactoryConf
    tier_orb_conf: TierConf


class ObsConf(BaseModel, frozen=True, extra="forbid", strict=True):
    observation_handler_conf: ObservationHandlerConf
    perception_conf: PerceptionConf


###########################
#    Top Configurations   #
###########################


class TierScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf
    obs_conf: ObsConf
