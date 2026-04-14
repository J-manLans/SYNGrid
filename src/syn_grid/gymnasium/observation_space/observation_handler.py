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
from numpy.typing import NDArray
from typing import Final


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    modality: Final[BaseModality]
    difficulty: Final[BaseDifficulty]

    def __init__(self, world: GridWorld, orb_conf: OrbFactoryConf, obs_conf: ObsConfig):
        self.modality = MODALITIES[obs_conf.observation_handler.modality](
            orb_conf, obs_conf.hard_difficulty
        )
        self.difficulty = DIFFICULTIES[obs_conf.observation_handler.difficulty](
            world, obs_conf.medium_difficulty
        )
        self._obs_conf = obs_conf

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self) -> spaces.Space:
        return self.modality.setup_obs_space(self.difficulty)

    def reset(self):
        self.step_count_down = self._obs_conf.medium_difficulty.max_steps
        # TODO: think I need to look over step_count down here that I give to medium obs
        # look at how I do it in old obs handler. Need it for get observation
        ...

    def get_observation(self, state) -> NDArray:
        filtered = self.difficulty.apply(state)
        return self.modality.encode(filtered)
        # Maybe like this instead
        # return self.difficulty.apply(state)


