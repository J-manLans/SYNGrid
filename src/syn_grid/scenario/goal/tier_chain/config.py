"""
The tier chain family: Goal/Tier Chain and every variant built on it.

Four registered scenarios share this hierarchy: spatial, delay, and the two
tier-scaling variants.

What the family adds to a goal world is one block, `tier_orb_conf`, beside the
grid, the droid, `goal_conf` and the optional `neg_orb_conf`. It holds what the
tier orbs and their digester need. A variant that needs a field the family
lacks subclasses that block, narrows `tier_orb_conf` on its own world model and
narrows `world_conf` on its own scenario model.

See `config/models/common_scenario_models.py` for the vocabulary every family
composes, and `scenario/goal/config.py` for what every goal scenario shares.
"""

from pydantic import Field, model_validator

from syn_grid.config.models.common_scenario_models import PenaltyCheckedConf
from syn_grid.scenario.goal.config import GoalConf, GoalScenarioConf, GoalWorldConf

# ===================== #
#    Tier Orb Models     #
# ===================== #


class TierOrbConf(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """The chain: one orb per tier, and what losing it costs.

    `max_tier` is the length of the chain, and therefore also the number of
    tier orbs on the field. All of them are present from the first step.

    `chain_break_penalty` is what breaking the chain costs, added to whatever
    reward the chain was holding. It is set independently of the timeout
    penalty; see `docs/dev/rppo-regression.md`.

    How the chain is scored is stated by each scenario's builder.
    """

    max_tier: int = Field(gt=0)
    chain_break_penalty: float


class TierDelayOrbConf(TierOrbConf, frozen=True, extra="forbid", strict=True):
    delay: int = Field(gt=0)


class TierDenseOrbConf(TierOrbConf, frozen=True, extra="forbid", strict=True):
    """Tier orbs that are each worth something: the reward ladder.

    A tier's reward is `base_reward * tier` when growth is linear, and
    `base_reward * tier ** growth_factor` otherwise.

    `chain_break_penalty` defaults to 0, so a broken chain pays the reward it
    was holding.
    """

    base_reward: float
    growth_factor: float = Field(gt=0)
    linear_reward_growth: bool
    chain_break_penalty: float = 0.0


# ===================== #
#      Goal Models       #
# ===================== #


class TierDenseGoalConf(GoalConf, frozen=True, extra="forbid", strict=True):
    """The goal of a scenario whose tier orbs pay for themselves.

    Both values default to 0, so a timeout pays the reward the chain was
    holding and a completed chain pays what its orbs were worth.
    """

    timeout_penalty: float = 0.0
    completion_reward: float = 0.0


# ======================= #
#   World Configuration   #
# ======================= #


class TierWorldConf(GoalWorldConf, frozen=True, extra="forbid", strict=True):
    tier_orb_conf: TierOrbConf

    @model_validator(mode="after")
    def validate_chain_fits_grid(self):
        if self.tier_orb_conf.max_tier >= (
            self.grid_conf.grid_rows * self.grid_conf.grid_cols
        ):
            raise ValueError(
                "max_tier can't be higher than number of cells in the grid, "
                "there will be no space for orbs"
            )

        return self


class TierDelayWorldConf(TierWorldConf, frozen=True, extra="forbid", strict=True):
    tier_orb_conf: TierDelayOrbConf


class TierDenseWorldConf(TierWorldConf, frozen=True, extra="forbid", strict=True):
    goal_conf: TierDenseGoalConf = TierDenseGoalConf()
    tier_orb_conf: TierDenseOrbConf


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class TierScenarioConf(GoalScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf


class TierDelayScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierDelayWorldConf


class TierDenseScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierDenseWorldConf
