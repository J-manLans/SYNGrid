from syn_grid.core.orbs.orb_meta import OrbCategory, DirectType, SynergyType
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.config.models import ObsConfig


class MediumDifficulty(BaseDifficulty):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, obs_conf: ObsConfig):
        self._medium_conf = obs_conf.medium_difficulty

    # ================= #
    #        API        #
    # ================= #

    def get_max_values(self) -> list[int]:
        max_steps = self._medium_conf.max_steps
        max_score = self._medium_conf.max_score
        max_tier_chain = self._medium_conf.max_tier
        max_category = len(OrbCategory) - 1
        max_type = max(len(DirectType) - 1, len(SynergyType) - 1)
        max_tier = self._medium_conf.max_tier
        max_orb_lifespan = (self._medium_conf.grid_rows - 1) + (
            self._medium_conf.grid_cols - 1
        )

        return [
            # agent values
            max_steps,
            max_score,
            max_tier_chain,
            # orb values
            max_category,
            max_type,
            max_tier,
            max_orb_lifespan,
        ]
