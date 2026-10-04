from enum import Enum

from pydantic import BaseModel, model_validator

# ======================= #
#      Helper Types       #
# ======================= #

class Scenario_name(str, Enum):
    GOAL_TIER_CHAIN_SPATIAL = "goal_tier_chain_spatial"
    GOAL_TIER_CHAIN_TIER_SCALING_SPARSE = "goal_tier_chain_tier_scaling_sparse"
    GOAL_TIER_CHAIN_TIER_SCALING_DENSE = "goal_tier_chain_tier_scaling_dense"
    GOAL_TIER_CHAIN_DELAY = "goal_tier_chain_delay"

# ======================= #
#  Nested Configurations  #
# ======================= #


class SnapshotConf(BaseModel, frozen=True, extra="forbid", strict=True):
    enabled: bool
    id: str

    @model_validator(mode="after")
    def validate_config(self):
        if self.enabled and not self.id:
            raise ValueError("snapshot.id must be set when snapshot is enabled")
        return self

# ============================= #
#    Top-Level Configurations   #
# ============================= #

class GlobalConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """
    Settings that apply to every run, whatever the scenario or runner.

    `scenario` is only a name. What it means, and which tunables it accepts, is decided by the
    scenario file and the scenario itself, not here.
    """

    snapshot: SnapshotConf
    scenario: Scenario_name
    human_control: bool