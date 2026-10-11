"""
The tier chain family: Goal/Tier Chain and every variant built on it.

Four registered scenarios share this hierarchy: spatial, delay, and the two
tier-scaling variants.

What the family adds to a goal world is two blocks beside the grid, the droid
and `goal_conf`: `tier_orb_conf`, which holds what the tier orbs and their
digester need, and the optional `neg_orb_conf`. A variant that needs a field
the family lacks subclasses the tier block, narrows `tier_orb_conf` on its own
world model and narrows `world_conf` on its own scenario model.

See `config/models/scenario/common_models.py` for the vocabulary every family
composes, `config/models/scenario/orb_models.py` for the orb kinds, and
`scenario/goal/config.py` for what every goal scenario shares.
"""

from pydantic import Field, model_validator

from syn_grid.config.models.scenario.orb_models import NegOrbConf, TierOrbConf
from syn_grid.scenario.goal.config import GoalScenarioConf, GoalWorldConf

# ===================== #
#    Tier Orb Models     #
# ===================== #


class TierDelayOrbConf(TierOrbConf, frozen=True, extra="forbid", strict=True):
    delay: int = Field(gt=0)


# ======================= #
#   World Configuration   #
# ======================= #


class TierWorldConf(GoalWorldConf, frozen=True, extra="forbid", strict=True):
    """A goal world whose objective is one chain of tier orbs.

    `tier_orb_conf.max_tier` is the length of the chain, and therefore also the
    number of tier orbs on the field. All of them are present from the first
    step. `tier_orb_conf.chain_break_penalty` is set independently of the
    timeout penalty; see `docs/dev/rppo-regression.md`.

    `neg_orb_conf` is optional. A world that has it holds one negative orb.
    """

    tier_orb_conf: TierOrbConf
    neg_orb_conf: NegOrbConf | None = None

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


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class TierScenarioConf(GoalScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf


class TierDelayScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierDelayWorldConf
