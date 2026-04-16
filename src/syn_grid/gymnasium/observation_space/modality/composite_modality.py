from syn_grid.core.orbs.orb_meta import OrbCategory, DirectType, SynergyType
from syn_grid.config.models import HardDifficultyConf, OrbFactoryConf
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.gymnasium.observation_space.modality.base_modality import (
    BaseModality,
)

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from typing import Final


class CompositeModality(BaseModality):
    # ================= #
    #       Init        #
    # ================= #

    hard_obs_high: Final[NDArray]

    def __init__(self, orb_conf: OrbFactoryConf, hard_conf: HardDifficultyConf):
        self.hard_obs_high = self._get_hard_obs_high(hard_conf)

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Space:
        return difficulty.get_max_values(self.hard_obs_high)

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    def _get_hard_obs_high(self, hard_conf: HardDifficultyConf) -> NDArray:
        max_agent_present = 1
        max_category = len(OrbCategory) - 1
        max_type = max(len(DirectType) - 1, len(SynergyType) - 1)
        max_tier = hard_conf.max_curriculum_tier

        return np.asarray(
            [max_agent_present, max_category, max_type, max_tier], dtype=np.float32
        )
