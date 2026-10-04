from pydantic import BaseModel, model_validator

# ===================== #
#     Top Scenarios     #
# ===================== #

class DroidConf(BaseModel, frozen=True, extra="forbid", strict=True):
    starting_score: float
    step_penalty: float
    boundary_penalty: float
    # TODO: remember to check whether this one shall be used, and remember — less is more
    reward_multiplier: float

    @model_validator(mode="after")
    def validate_penalties(self):
        bad = [
            name
            for name in type(self).model_fields
            if name.endswith("_penalty") and getattr(self, name) > 0
        ]
        if bad:
            raise ValueError(f"{', '.join(bad)} must be 0 or negative")
        return self

class GoalDroidConf(DroidConf, frozen=True, extra="forbid", strict=True):
    timeout_penalty: float


# ===================== #
#  Lower-down Scenarios  #
# ===================== #


class TierDroidConf(GoalDroidConf, frozen=True, extra="forbid", strict=True):
    chain_break_penalty: float
    tier_consumption_penalty: float