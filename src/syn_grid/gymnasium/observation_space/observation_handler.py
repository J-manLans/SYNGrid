from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.gymnasium.observation_space.perceptions.vector import (
    EasyVectorPerception,
    MediumVectorPerception,
    HardVectorPerception,
)
from syn_grid.gymnasium.observation_space.perceptions.composite import (
    EasyCompositePerception,
    MediumCompositePerception,
    HardCompositePerception,
)
from syn_grid.config.models import ObsConfig
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces
from typing import Final, Type

PERCEPTIONS = {
    "vector_easy": EasyVectorPerception,
    "vector_medium": MediumVectorPerception,
    "vector_hard": HardVectorPerception,
    "composite_easy": EasyCompositePerception,
    "composite_medium": MediumCompositePerception,
    "composite_hard": HardCompositePerception,
}


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: ObsConfig, orbs: int) -> None:
        self._max_steps: Final[int] = conf.observation_handler.max_steps
        perception_type: Type[BasePerception] = PERCEPTIONS[
            conf.observation_handler.perception
        ]
        self.perception: Final[BasePerception] = perception_type(conf.perception, orbs)

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self) -> spaces.Space:
        return self.perception.setup_obs_space()

    def reset(self) -> None:
        self.steps_left: int = self._max_steps
        self.perception.reset()

    def get_observation(self, state: GridWorld) -> np.ndarray:
        return self.perception.get_observation(state, self.steps_left)
