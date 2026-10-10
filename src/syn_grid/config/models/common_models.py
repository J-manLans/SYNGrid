"""
The vocabulary every scenario is written in.

These are the blocks a scenario composes rather than defines: the grid, the two
observation blocks, the orb kinds, the droid, and the plain
`world_conf` / `obs_conf` / `scenario_conf` triple that every scenario's
configuration is an extension of. A block that names a scenario family's rule --
a tier chain -- is not here; it lives in that family's `config.py` in the
scenario package, next to its builders, such as
`scenario/goal/tier_chain/config.py`. What every goal scenario shares is one
level up, in `scenario/goal/config.py`.

The split is by family, not by scenario name. Names that are configured
identically share a file, because splitting them would mean four copies of the
same validators with four places to forget one.

`GlobalConf` and `RunnerConf` are shared too, but not by scenarios: they live
in `global_models.py` and `runner_models.py`.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

# ======================= #
#      Helper Types       #
# ======================= #


class Perception(str, Enum):
    VECTOR_FOG_OF_WAR = "vector_fog_of_war"
    VECTOR_MARKOVIAN = "vector_markovian"


# ======================= #
#     Grid Models         #
# ======================= #


class GridConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_rows: int = Field(gt=0)
    grid_cols: int = Field(gt=0)


# ======================= #
#     Droid Models        #
# ======================= #


class DroidConf(BaseModel, frozen=True, extra="forbid", strict=True):
    starting_score: float
    step_penalty: float
    boundary_penalty: float

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


# ======================= #
#      Orb Models         #
# ======================= #


class OrbKindConf(BaseModel, frozen=True, extra="forbid", strict=True):
    cool_down: int = Field(ge=0)
    weight: int = Field(gt=0)


class NegOrbConf(OrbKindConf, frozen=True, extra="forbid", strict=True):
    reward: float = Field(le=0)


class OrbPoolConf(BaseModel, frozen=True, extra="forbid", strict=True):
    negative: NegOrbConf | None = None


# ======================= #
#    Obs Configuration    #
# ======================= #


class ObservationHandlerConf(BaseModel, frozen=True, extra="forbid", strict=True):
    perception: Annotated[Perception, Field(strict=False)]
    max_steps: int = Field(gt=0)


class PerceptionConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """How an observation is encoded.

    The world-derived counts are not here. How many orb slots an observation
    holds and how far its tier channel reaches are properties of the world,
    which the scenario already describes; this block used to carry a second
    copy of each that had to be kept in step by hand through YAML anchors, and
    an observation space was only fully knowable by reading four files at once.
    """

    max_score: int


# ============================= #
#      Domain Config Blocks     #
# ============================= #


class WorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    grid_conf: GridConf
    droid_conf: DroidConf
    orb_conf: OrbPoolConf


class ObsConf(BaseModel, frozen=True, extra="forbid", strict=True):
    observation_handler_conf: ObservationHandlerConf


class ScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    world_conf: WorldConf
    obs_conf: ObsConf
