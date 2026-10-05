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
`common_models.py` for the vocabulary every family composes.
"""

from enum import Enum
from typing import Annotated

from pydantic import Field, model_validator

from syn_grid.config.models.common_models import (
    GoalDroidConf,
    OrbKindConf,
    OrbPoolConf,
    ScenarioConf,
    WorldConf,
)

# ======================= #
#      Helper Types       #
# ======================= #


class ScoringMode(str, Enum):
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

# ===================== #
#      Droid Models      #
# ===================== #


class TierDroidConf(GoalDroidConf, frozen=True, extra="forbid", strict=True):
    """A droid chasing a chain rather than a running score.

    `chain_break_penalty` and `tier_consumption_penalty` only mean anything once
    there is a chain to lose. Both ratios are deliberate and both were
    separately identified as the interesting reward knobs -- see
    `docs/dev/rppo-regression.md` for why the timeout penalty had to become
    independent of the chain-break one.
    """

    chain_break_penalty: float
    tier_consumption_penalty: float


# ===================== #
#       Orb Models       #
# ===================== #


class TierOrbConf(OrbKindConf, frozen=True, extra="forbid", strict=True):
    """One orb per tier.

    `max_tier` is the length of the chain, and therefore also the number of orbs
    on the field: a tier chain derives its field size from the chain.
    """

    max_tier: int = Field(gt=0)
    base_reward: float
    growth_factor: float = Field(gt=0)
    linear_reward_growth: bool
    scoring: Annotated[ScoringMode, Field(strict=False)]


class TierDelayOrbConf(TierOrbConf, frozen=True, extra="forbid", strict=True):
    delay: int = Field(gt=0)


class TierOrbPoolConf(OrbPoolConf, frozen=True, extra="forbid", strict=True):
    tier: TierOrbConf


class TierDelayOrbPoolConf(TierOrbPoolConf, frozen=True, extra="forbid", strict=True):
    tier: TierDelayOrbConf


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


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class TierScenarioConf(ScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf


class TierDelayScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierDelayWorldConf
