from syn_grid.config.models import OrbFactoryConf, ObsConfig, ObservationHandlerConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.observation_space.observation_registry import (
    MODALITIES,
    DIFFICULTIES,
)
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.gymnasium.observation_space.modality.base_modality import (
    BaseModality,
)

from gymnasium import spaces
import numpy as np
from typing import Final


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    modality: Final[BaseModality]
    difficulty: Final[BaseDifficulty]

    def __init__(self, orb_conf: OrbFactoryConf, obs_conf: ObsConfig):
        self._max_steps = obs_conf.medium_difficulty.max_steps
        self.modality = MODALITIES[obs_conf.observation_handler.modality](
            orb_conf, obs_conf.hard_difficulty
        )
        self.difficulty = DIFFICULTIES[obs_conf.observation_handler.difficulty](
            obs_conf
        )

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self) -> spaces.Space:
        return self.modality.setup_obs_space(self.difficulty)

    def reset(self):
        self.steps_left = self._max_steps

    def get_observation(self, state: GridWorld) -> np.ndarray:
        return self.difficulty.get_observation(state, self.steps_left)
