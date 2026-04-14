from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)

from abc import ABC, abstractmethod
from gymnasium import spaces
from numpy.typing import NDArray


class BaseModality(ABC):
    @abstractmethod
    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Space: ...

    # TODO: Will look over this method, don't even know if I need it
    @abstractmethod
    def encode(self, difficulty: dict[str, NDArray]) -> NDArray: ...
