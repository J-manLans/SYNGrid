from syn_grid.config.models import OrbFactoryConf, ObservationConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.observation_space_developing.observation_registry import (
    MODALITIES,
    DIFFICULTIES,
)
from syn_grid.gymnasium.observation_space_developing.difficulty.base_difficulty import BaseDifficulty
from syn_grid.gymnasium.observation_space_developing.modality.base_modality import BaseModality

from gymnasium import spaces
from numpy.typing import NDArray
from typing import Final


class ObservationHandlerDeveloping:
    # ================= #
    #       Init        #
    # ================= #

    modality: Final[BaseModality]
    difficulty: Final[BaseDifficulty]

    def __init__(
        self, world: GridWorld, orb_conf: OrbFactoryConf, obs_conf: ObservationConf
    ):
        self.modality = MODALITIES[obs_conf.modality](orb_conf, obs_conf)
        self.difficulty = DIFFICULTIES[obs_conf.difficulty](world)

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self) -> spaces.Space:
        return self.modality.setup_obs_space(self.difficulty)

    def get_observation(self, state) -> NDArray:
        filtered = self.difficulty.apply(state)
        return self.modality.encode(filtered)
