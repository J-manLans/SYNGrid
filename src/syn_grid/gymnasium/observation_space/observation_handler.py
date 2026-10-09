from typing import Any, Final

from gymnasium import spaces

from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.gymnasium.observation_space.perceptions.composite import (
    CompositeFullyPOMDP,
    CompositeGridMarkovian,
    CompositeMarkovian,
)
from syn_grid.gymnasium.observation_space.perceptions.spatial import (
    GridPixel,
)
from syn_grid.gymnasium.observation_space.perceptions.vector import (
    VectorFogOfWar,
    VectorMarkovian,
    VectorMarkovianEasy,
)
from syn_grid.scenario.blocks.observation import ObservationRules

PERCEPTIONS = {
    "vector_markovian_easy": VectorMarkovianEasy,
    "vector_markovian": VectorMarkovian,
    "vector_fog_of_war": VectorFogOfWar,
    "composite_markovian": CompositeMarkovian,
    "composite_fully_pomdp": CompositeFullyPOMDP,
    "composite_grid_markovian": CompositeGridMarkovian,
    "grid_pixel": GridPixel,
}


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        observation_rules: ObservationRules,
        grid_dimensions: tuple[int, int],
        orbs: int,
        max_identity: int,
    ) -> None:
        self._max_steps: Final[int] = observation_rules.max_steps
        perception_type: type[BasePerception] = PERCEPTIONS[
            observation_rules.perception
        ]
        self.perception: Final[BasePerception] = perception_type(
            observation_rules, grid_dimensions, orbs, max_identity
        )

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
