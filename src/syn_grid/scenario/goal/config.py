"""
What every goal scenario's configuration shares.

A goal scenario has an objective to reach and a deadline to miss. The blocks
here are the ones that follow from that, whatever the family; a family's own
models live in its folder, such as `tier_chain/config.py`, and extend these.
"""

from syn_grid.config.models.common_scenario_models import (
    PenaltyCheckedConf,
    ScenarioConf,
    WorldConf,
)

# ======================= #
#      Goal Models        #
# ======================= #


class GoalConf(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """An objective, and therefore a deadline to miss.

    `completion_reward` is what reaching the objective pays, added to the
    reward of the step that reached it. `timeout_penalty` is what missing the
    deadline costs, added to whatever reward was still being held.
    """

    timeout_penalty: float
    completion_reward: float


# ======================= #
#   World Configuration   #
# ======================= #


class GoalWorldConf(WorldConf, frozen=True, extra="forbid", strict=True):
    goal_conf: GoalConf


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class GoalScenarioConf(ScenarioConf, frozen=True, extra="forbid", strict=True):
    world_conf: GoalWorldConf
