from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class ScoringMode(Enum):
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

class OrbKindConf(BaseModel, frozen=True, extra="forbid", strict=True):
    cool_down: int
    weight: int

class NegOrbConf(OrbKindConf, frozen=True, extra="forbid", strict=True):
    reward: float


class TierOrbConf(OrbKindConf, frozen=True, extra="forbid", strict=True):
    max_tier: int
    base_reward: float
    growth_factor: float
    linear_reward_growth: bool
    scoring: Annotated[ScoringMode, Field(strict=False)]

    @model_validator(mode="after")
    def validate_config(self):
        if self.growth_factor <= 0:
            raise ValueError(f"{self.growth_factor} must be a positive value.")
        if self.max_tier <= 0:
            raise ValueError("max_tier should be larger than 0")

        return self

class OrbPoolConf(BaseModel, frozen=True, extra="forbid", strict=True):
    max_active_orbs: int
    negative: NegOrbConf | None = None

class TierOrbPoolConf(OrbPoolConf, frozen=True, extra="forbid", strict=True):
    tier: TierOrbConf
    negative: NegOrbConf | None = None
