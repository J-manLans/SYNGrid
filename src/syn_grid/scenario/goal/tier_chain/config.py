"""
The tier chain family: Goal/Tier Chain and every variant built on it.

Four registered scenarios share this hierarchy -- spatial, delay, and the two
tier-scaling variants -- because they differ in *rules*, not in shape. A fogged
window, a delay on consume, a longer reward ladder: all of those are things a
scenario builder does, and none of them change what a config file has to contain.
So they get one file, and a config that names any of them validates against the
same models.

What the family adds to a goal scenario is one block, `tier_orb_conf`, beside
`world_conf`, `obs_conf`, `goal_conf` and the optional `neg_orb_conf`. It holds
what the tier orbs and their digester need. A variant that needs a field the
family lacks subclasses that block and narrows `tier_orb_conf` on its scenario
model, so the departure is two classes defined next to each other here.

`continuous` is a different family and does not appear here; see
`config/models/common_models.py` for the vocabulary every family composes, and
`scenario/goal/config.py` for what every goal scenario shares.
"""

from pydantic import Field, model_validator

from syn_grid.config.models.common_scenario_models import (
    ObsConf,
    PenaltyCheckedConf,
    PerceptionConf,
)
from syn_grid.scenario.goal.config import GoalScenarioConf

# ===================== #
#    Tier Orb Models     #
# ===================== #


class TierOrbConf(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """The chain: one orb per tier, and what losing it costs.

    `max_tier` is the length of the chain, and therefore also the number of orbs
    on the field: a tier chain derives its field size from the chain.

    `chain_break_penalty` only means anything once there is a chain to lose.
    Its ratio to the timeout penalty is deliberate -- see
    `docs/dev/rppo-regression.md` for why the two had to become independent.

    Not an `OrbKindConf`: a chain's orbs are all present from the first step
    and never come back, so they have no spawn weight and no cool-down. A
    per-tier reward ladder is not here either: only a scenario that pays per
    tier has one, see `TierDenseOrbConf`. How the chain is scored is not a
    tunable at all; each scenario's builder states it.
    """

    max_tier: int = Field(gt=0)
    chain_break_penalty: float


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


# ======================= #
#    Obs Configuration    #
# ======================= #


class TierDenseObsConf(ObsConf, frozen=True, extra="forbid", strict=True):
    perception_conf: PerceptionConf


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class TierScenarioConf(GoalScenarioConf, frozen=True, extra="forbid", strict=True):
    tier_orb_conf: TierOrbConf

    @model_validator(mode="after")
    def validate_chain_fits_grid(self):
        grid_conf = self.world_conf.grid_conf

        if self.tier_orb_conf.max_tier >= (grid_conf.grid_rows * grid_conf.grid_cols):
            raise ValueError(
                "max_tier can't be higher than number of cells in the grid, "
                "there will be no space for orbs"
            )

        return self


class TierDelayScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    tier_orb_conf: TierDelayOrbConf


class TierDenseScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    tier_orb_conf: TierDenseOrbConf
    obs_conf: TierDenseObsConf
