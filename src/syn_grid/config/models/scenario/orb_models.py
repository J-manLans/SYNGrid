"""
The orb kinds a scenario can be built from.

Each block describes one orb kind itself, whichever scenario it appears in, and
holds what that kind's orbs and its digester need. A family that needs more
subclasses the block in its own folder, and a scenario's world model declares
the field.
"""

from pydantic import BaseModel, Field

from syn_grid.config.models.scenario.common_models import PenaltyCheckedConf

# ======================= #
#      Direct Orbs        #
# ======================= #


class NegOrbConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """A negative orb: whether the world holds one, what eating it costs, and
    how long it stays away.

    `enabled` is the switch. A disabled block keeps its values, so turning the
    orb on brings back the same orb.

    `cool_down` is the number of steps the orb is off the field after it is
    eaten or despawns, before it can appear again.
    """

    enabled: bool
    reward: float = Field(le=0)
    cool_down: int = Field(ge=0)


# ======================= #
#      Synergy Orbs       #
# ======================= #


class TierOrbConf(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """Tier orbs: how many tiers there are, what each is worth, and what
    breaking their order costs.

    `max_tier` is the highest tier.

    A tier's reward is `base_reward * tier ** growth_factor`. A growth factor
    of 1 makes the rewards grow linearly with the tier.

    `chain_break_penalty` is what eating a tier orb out of order costs.

    How tier orbs are scored is stated by each scenario's builder.
    """

    max_tier: int = Field(gt=0)
    base_reward: float
    growth_factor: float = Field(gt=0)
    chain_break_penalty: float
