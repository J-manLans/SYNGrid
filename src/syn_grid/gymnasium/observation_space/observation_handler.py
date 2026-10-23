from typing import Any, Final

from gymnasium import spaces

from syn_grid.config.models.scenario.common_models import Perception
from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.gymnasium.observation_space.perceptions.vector import (
    VectorFogOfWar,
    VectorMarkovian,
)
from syn_grid.scenario.blocks.observation import ObservationRules

# TODO(phase 7): the perceptions outside the `Perception` enum are rewritten from their copies in
# `legacy/` in the observation pass: VectorMarkovianEasy, CompositeMarkovian, CompositeFullyPOMDP,
# CompositeGridMarkovian and GridPixel.
PERCEPTIONS: Final[dict[Perception, type[BasePerception]]] = {
    Perception.VECTOR_FOG_OF_WAR: VectorFogOfWar,
    Perception.VECTOR_MARKOVIAN: VectorMarkovian,
}


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        observation: ObservationRules,
        grid_dimensions: tuple[int, int],
        max_identity: int,
    ) -> None:
        perception_type: type[BasePerception] = PERCEPTIONS[observation.perception]
        self.perception: Final[BasePerception] = perception_type(
            observation, grid_dimensions, max_identity
        )

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self) -> spaces.Space:
        return self.perception.setup_obs_space()

    def reset(self) -> None:
        self.perception.reset()

    def get_observation(self, state: GridWorld) -> Any:
        return self.perception.get_observation(state)
