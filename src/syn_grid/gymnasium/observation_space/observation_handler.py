from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.gymnasium.observation_space.perceptions.vector import (
    VectorMarkovian,
    VectorFullyPOMDP,
)
from syn_grid.gymnasium.observation_space.perceptions.composite import (
    CompositeMarkovian,
    CompositeFullyPOMDP,
    CompositeGridMarkovian
)
from syn_grid.gymnasium.observation_space.perceptions.spatial import (
    GridPixel,
)
from syn_grid.config.models import ObsConfig
from syn_grid.core.grid_world import GridWorld

from gymnasium import spaces
from typing import Final, Type, Any

PERCEPTIONS = {
    "vector_markovian": VectorMarkovian,
    "vector_fully_pomdp": VectorFullyPOMDP,
    "composite_markovian": CompositeMarkovian,
    "composite_fully_pomdp": CompositeFullyPOMDP,
    "composite_grid_markovian": CompositeGridMarkovian,
    "grid_pixel": GridPixel,
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

    def get_observation(self, state: GridWorld) -> Any:
        return self.perception.get_observation(state, self.steps_left)
