"""
The vocabulary every scenario is written in.

These are the blocks every scenario's configuration is built from: the grid,
the droid, the negative orb, the observation handler, and the base `WorldConf`
and `ScenarioConf` that each scenario type and family extends.

What every goal scenario shares is in `scenario/goal/config.py`, and a family's
own models are in its folder, such as `scenario/goal/tier_chain/config.py`.
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
#   Penalties and Droid   #
# ======================= #


class PenaltyCheckedConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """A block whose penalties must not be positive.

    Every field whose name ends in `_penalty` is checked, on this block and on
    any block that extends it, so a new penalty is covered by its name alone.
    """

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


class DroidConf(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """The droid's energy and what moving costs it.

    `max_energy` is both what the droid starts an episode with and the most it
    can hold: it starts fully charged and cannot be overcharged. Every reward
    and penalty moves the energy, which stays between 0 and `max_energy`, and
    the episode ends when it reaches 0.

    The score always starts at 0, moves with the same rewards and penalties,
    and has no bounds.
    """

    max_energy: float = Field(gt=0)
    step_penalty: float
    boundary_penalty: float


# ======================= #
#      Orb Models         #
# ======================= #


class NegOrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """A negative orb: what eating it costs, and how long it stays away.

    `cool_down` is the number of steps the orb is off the field after it is
    eaten or despawns, before it can appear again.
    """

    reward: float = Field(le=0)
    cool_down: int = Field(ge=0)


# ======================= #
#    Obs Configuration    #
# ======================= #


class ObservationHandlerConf(BaseModel, frozen=True, extra="forbid", strict=True):
    perception: Annotated[Perception, Field(strict=False)]


# ============================= #
#      Domain Config Blocks     #
# ============================= #


class WorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The world an episode plays out in, and everything the world acts on.

    `max_steps` is how long an episode can last. The world counts the steps, so
    anything that is handed the world can read how many are left.

    Each orb kind a world can hold has its own `*_orb_conf` block, carrying what
    that kind's orbs and its digester need. Any scenario may add negative orbs
    through `neg_orb_conf`; a family adds its own kinds on its world model.
    """

    max_steps: int = Field(gt=0)
    grid_conf: GridConf
    droid_conf: DroidConf
    neg_orb_conf: NegOrbConf | None = None


class ObsConf(BaseModel, frozen=True, extra="forbid", strict=True):
    observation_handler_conf: ObservationHandlerConf


class ScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """What every scenario's configuration has: the world, and how it is observed.

    `world_conf` is everything the world acts on. `obs_conf` is what the
    Gymnasium side needs to turn the world into an observation.
    """

    world_conf: WorldConf
    obs_conf: ObsConf
