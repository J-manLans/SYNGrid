from enum import Enum

from pydantic import BaseModel, model_validator

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

class NegativeConf(BaseModel, frozen=True, extra="forbid", strict=True):
    reward: float | None = None
    cool_down: int | None = None
    weight: int | None = None


class TierConf(BaseModel, frozen=True, extra="forbid", strict=True):
    max_tier: int
    base_reward: float
    growth_factor: float
    linear_reward_growth: bool
    scoring: ScoringMode
    cool_down: int
    weight: int


    @model_validator(mode="after")
    def validate_config(self):
        if self.growth_factor <= 0:
            raise ValueError(f"{self.growth_factor} must be a positive value.")
        if self.max_tier <= 0:
            raise ValueError("max_tier should be larger than 0")

        return self