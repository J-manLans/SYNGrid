from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

# ======================= #
#      Helper Types       #
# ======================= #


class ScenarioName(str, Enum):
    GOAL_SINGLE_TIER_CHAIN_SPATIAL = "goal_single_tier_chain_spatial"
    GOAL_SINGLE_TIER_CHAIN_SCALING_SPARSE = "goal_single_tier_chain_scaling_sparse"
    GOAL_SINGLE_TIER_CHAIN_SCALING_DENSE = "goal_single_tier_chain_scaling_dense"
    GOAL_SINGLE_TIER_CHAIN_DELAY = "goal_single_tier_chain_delay"


# ============================= #
#    Top-Level Configurations   #
# ============================= #


class GlobalConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """
    Settings that apply to every run, whatever the scenario or runner.

    `scenario` names the scenario to run. Its tunables are in the YAML file of the same name.
    """

    scenario: Annotated[ScenarioName, Field(strict=False)]
    human_control: bool
