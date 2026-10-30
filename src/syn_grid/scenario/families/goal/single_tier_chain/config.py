"""
The single tier chain family: Goal/Single Tier Chain and every variant built on it.

Four registered scenarios share this hierarchy: spatial, delay, and the two
tier-scaling variants.

What the family adds to a goal world is two blocks beside the grid, the droid
and `goal_conf`: `tier_orb_conf`, which holds what the tier orbs and their
digester need, and `neg_orb_conf`, the negative orb and its switch. A variant
that needs a field the family lacks subclasses the tier block, narrows
`tier_orb_conf` on its own world model and narrows `world_conf` on its own
scenario model.

These are the full models, which a builder takes. A locked scenario also has a
small model, at the end of this file, holding only what a run may choose.

See `config/models/scenario/common_models.py` for the vocabulary every family
composes, `config/models/scenario/orb_models.py` for the orb kinds, and
`scenario/families/goal/config.py` for what every goal scenario shares.
"""

from pydantic import BaseModel, Field, model_validator

from syn_grid.config.models.scenario.common_models import GridConf, ObsConf
from syn_grid.config.models.scenario.orb_models import NegOrbConf, TierOrbConf
from syn_grid.scenario.families.goal.config import GoalScenarioConf, GoalWorldConf

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

    A world whose `neg_orb_conf` is enabled holds one negative orb.
    """

    tier_orb_conf: TierOrbConf
    neg_orb_conf: NegOrbConf

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


# ============================= #
#    Locked Scenario Models     #
# ============================= #
# What a run of a locked scenario may choose, under the full models' names and nesting, so a
# run's YAML merges onto the scenario's base config. A YAML that states anything else is rejected.


class NegOrbSwitchConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The negative orb's switch. Its values are the base config's."""

    enabled: bool


class SpatialWorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """What a run of spatial chooses about the world: the grid, the steps it is given, and
    whether the negative orb is on."""

    max_steps: int = Field(gt=0)
    grid_conf: GridConf
    neg_orb_conf: NegOrbSwitchConf


class SpatialScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """What a run of spatial chooses: its world, and the perception."""

    world_conf: SpatialWorldConf
    obs_conf: ObsConf


class ScalingTierOrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """What a run of a scaling scenario chooses about the tier orbs: the length of the chain."""

    max_tier: int = Field(gt=0)


class ScalingWorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """What a run of a scaling scenario chooses about the world: the grid, the steps it is
    given, the length of the chain, and whether the negative orb is on."""

    max_steps: int = Field(gt=0)
    grid_conf: GridConf
    tier_orb_conf: ScalingTierOrbConf
    neg_orb_conf: NegOrbSwitchConf


class ScalingScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """What a run of scaling sparse or scaling dense chooses: its world, and the perception."""

    world_conf: ScalingWorldConf
    obs_conf: ObsConf
