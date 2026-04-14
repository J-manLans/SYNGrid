from syn_grid.core.grid_world import GridWorld

from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray


class BaseDifficulty(ABC):
    @abstractmethod
    def setup_obs_space(self, hard_obs_high: NDArray) -> spaces.Space: ...

    @abstractmethod
    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray: ...
