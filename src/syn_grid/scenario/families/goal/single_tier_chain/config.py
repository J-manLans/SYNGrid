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

These are the full models, which a builder takes. A locked scenario also has
its tunables, at the end of this file, holding only what a run may set.

See `config/models/scenario/common_models.py` for the vocabulary every family
composes, `config/models/scenario/orb_models.py` for the orb kinds, and
`scenario/families/goal/config.py` for what every goal scenario shares.
"""

from pydantic import BaseModel, Field, model_validator

from syn_grid.config.models.scenario.common_models import GridConf, ObsConf
from syn_grid.config.models.scenario.orb_models import NegOrbConf, TierOrbConf
from syn_grid.scenario.families.goal.config import GoalScenarioConf, GoalWorldConf

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


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class TierScenarioConf(GoalScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: TierWorldConf


# TODO(delay): add the delay, once it is settled where its mechanic lives. It is a rule of the
# field rather than of the tier orbs, so it may not belong in `tier_orb_conf`.
class TierDelayScenarioConf(TierScenarioConf, frozen=True, extra="forbid", strict=True):
    """The config of the delay scenario."""


# ============================== #
#    Locked Scenario Tunables    #
# ============================== #
# What a run of a locked scenario may set, under the full models' names and nesting, so a
# run's YAML merges onto the scenario's base config. A YAML that states anything else is rejected.

# === Globals === #


class NegOrbSwitchConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The negative orb's switch. Its values are the base config's."""

    enabled: bool


# === Spatial Scenario === #


class SpatialWorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The tunables of spatial's world: the grid, the steps it is given, and
    whether the negative orb is on."""

    max_steps: int = Field(gt=0)
    grid_conf: GridConf
    neg_orb_conf: NegOrbSwitchConf


class SpatialScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The tunables of spatial: its world, and the perception."""

    world_conf: SpatialWorldConf
    obs_conf: ObsConf


# === Scaling Scenario === #


class ScalingTierOrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The tunables of a scaling scenario's tier orbs: the length of the chain."""

    max_tier: int = Field(gt=0)


class ScalingWorldConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The tunables of a scaling scenario's world: the grid, the steps it is
    given, the length of the chain, and whether the negative orb is on."""

    max_steps: int = Field(gt=0)
    grid_conf: GridConf
    tier_orb_conf: ScalingTierOrbConf
    neg_orb_conf: NegOrbSwitchConf


class ScalingScenarioConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """The tunables of scaling sparse and scaling dense: the world, and the perception."""

    world_conf: ScalingWorldConf
    obs_conf: ObsConf
