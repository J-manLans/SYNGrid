from syn_grid.gymnasium.observation_space_developing.difficulty.base_difficulty import BaseDifficulty

from abc import ABC, abstractmethod
from gymnasium import spaces
from numpy.typing import NDArray

class BaseModality(ABC):
    @abstractmethod
    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Dict:
        ...

    @abstractmethod
    def encode(self, difficulty: BaseDifficulty) -> NDArray:
        ...