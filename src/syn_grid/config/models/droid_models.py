from syn_grid.config.models.scenario_models import DroidConf


# ===================== #
#     Top Scenarios     #
# ===================== #

class GoalDroidConf(DroidConf, frozen=True, extra="forbid", strict=True):
    timeout_penalty: float


# ===================== #
#  Lower-down Scenarios  #
# ===================== #


class TierOrbDroidConf(GoalDroidConf, frozen=True, extra="forbid", strict=True):
    chain_break_penalty: float
    tier_consumption_penalty: float