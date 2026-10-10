"""
The tier chain family: Goal/Tier Chain and every variant built on it.

Four registered scenarios share this hierarchy -- spatial, delay, and the two
tier-scaling variants -- because they differ in *rules*, not in shape. A fogged
window, a delay on consume, a longer reward ladder: all of those are things a
scenario builder does, and none of them change what a config file has to contain.
So they get one file, and a config that names any of them validates against the
same models.

The split that matters is family, not scenario name. A scenario earns its own
file when it needs a field the family does not have -- at which point it is a
subclass of the block below, defined next to it, and the departure is visible as
a diff rather than spread across four files.

`continuous` is a different family and does not appear here; see
`config/models/common_models.py` for the vocabulary every family composes, and
`scenario/goal/config.py` for what every goal scenario shares.
"""

from pydantic import BaseModel, Field, model_validator

from syn_grid.config.models.common_models import (
    ObsConf,
    OrbPoolConf,
    PerceptionConf,
    ScenarioConf,
    WorldConf,
)
from syn_grid.scenario.goal.config import GoalDroidConf

# ===================== #
#      Droid Models      #
# ===================== #


class TierDroidConf(GoalDroidConf, frozen=True, extra="forbid", strict=True):
    """A droid chasing a chain rather than a running score.

    `chain_break_penalty` only means anything once there is a chain to lose.
    Its ratio to the timeout penalty is deliberate -- see
    `docs/dev/rppo-regression.md` for why the two had to become independent.
    """

    chain_break_penalty: float


# ===================== #
#       Orb Models       #
# ===================== #


class TierOrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """One orb per tier.

    `max_tier` is the length of the chain, and therefore also the number of orbs
    on the field: a tier chain derives its field size from the chain.

    Not an `OrbKindConf`: a chain's orbs are all present from the first step
    and never come back, so they have no spawn weight and no cool-down. A
    per-tier reward ladder is not here either: only a scenario that pays per
    tier has one, see `TierDenseOrbConf`. How the chain is scored is not a
    tunable at all; each scenario's builder states it.
    """

    max_tier: int = Field(gt=0)


class TierDelayOrbConf(TierOrbConf, frozen=True, extra="forbid", strict=True):
    delay: int = Field(gt=0)


class TierDenseOrbConf(TierOrbConf, frozen=True, extra="forbid", strict=True):
    """Tier orbs that are each worth something: the reward ladder.

    A tier's reward is `base_reward * tier` when growth is linear, and
    `base_reward * tier ** growth_factor` otherwise.
    """

    base_reward: float
    growth_factor: float = Field(gt=0)
    linear_reward_growth: bool


class TierOrbPoolConf(OrbPoolConf, frozen=True, extra="forbid", strict=True):
    tier: TierOrbConf


class TierDelayOrbPoolConf(TierOrbPoolConf, frozen=True, extra="forbid", strict=True):
    tier: TierDelayOrbConf


class TierDenseOrbPoolConf(TierOrbPoolConf, frozen=True, extra="forbid", strict=True):
    tier: TierDenseOrbConf


# ======================= #
#   World Configuration   #
# ======================= #


class TierWorldConf(WorldConf, frozen=True, extra="forbid", strict=True):
    droid_conf: TierDroidConf
    orb_conf: TierOrbPoolConf

    @model_validator(mode="after")
    def validate_chain_fits_grid(self):
        if self.orb_conf.tier.max_tier >= (
            self.grid_conf.grid_rows * self.grid_conf.grid_cols
        ):
            raise ValueError(
                "max_tier can't be higher than number of cells in the grid, "
                "there will be no space for orbs"
            )

        return self


class TierDelayWorldConf(TierWorldConf, frozen=True, extra="forbid", strict=True):
    orb_conf: TierDelayOrbPoolConf


class TierDenseWorldConf(TierWorldConf, frozen=True, extra="forbid", strict=True):
    orb_conf: TierDenseOrbPoolConf


class TierDenseObsConf(ObsConf, frozen=True, extra="forbid", strict=True):
    perception_conf: PerceptionConf


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class TierScenarioConf(ScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf


class TierDelayScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierDelayWorldConf


class TierDenseScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierDenseWorldConf
    obs_conf: TierDenseObsConf
