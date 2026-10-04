
from pydantic import BaseModel, model_validator

from syn_grid.config.models.scenario.droid_models import DroidConf
from syn_grid.config.models.scenario.orb_models import OrbPoolConf


class GridConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_rows: int
    grid_cols: int


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
    include_timer: bool
    tiers: int


# === PerceptionConf END === #

# ======================= #
#   Domain Config Blocks  #
# ======================= #


class WorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_conf: GridConf
    droid_conf: DroidConf
    orb_conf: OrbPoolConf

class ObsConf(BaseModel, frozen=True, extra="forbid", strict=True):
    observation_handler_conf: ObservationHandlerConf
    perception_conf: PerceptionConf

class ScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    world_conf: WorldConf
    obs_conf: ObsConf
