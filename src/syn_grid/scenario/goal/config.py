"""
What every goal scenario's configuration shares.

A goal scenario has an objective to reach and a deadline to miss. The blocks
here are the ones that follow from that, whatever the family; a family's own
models live in its folder, such as `tier_chain/config.py`, and extend these.
"""

from syn_grid.config.models.common_scenario_models import PenaltyCheckedConf, ScenarioConf

# ======================= #
#      Goal Models        #
# ======================= #


class GoalConf(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """An objective, and therefore a deadline to miss.

    Scenarios that have no goal -- nothing to finish, so the episode only ends
    when the clock or the score does -- have no such block.

    `completion_reward` is what reaching the objective pays, and
    `timeout_penalty` what missing the deadline costs.
    """

    timeout_penalty: float
    completion_reward: float


# ============================= #
#    Top-Level Configuration   #
# ============================= #


class GoalScenarioConf(ScenarioConf, frozen=True, extra="forbid", strict=True):
    goal_conf: GoalConf
