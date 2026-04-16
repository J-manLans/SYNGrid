from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)

from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np


class BaseModality(ABC):
    @abstractmethod
    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Space: ...

    @abstractmethod
    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray: ...
