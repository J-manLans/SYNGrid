from abc import ABC, abstractmethod
from gymnasium import spaces
from numpy.typing import NDArray


class BaseDifficulty(ABC):
    @abstractmethod
    def setup_obs_space(self, hard_obs_high: NDArray) -> spaces.Space: ...

    @abstractmethod
    def apply(self, state) -> dict[str, NDArray]: ...
