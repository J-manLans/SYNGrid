from abc import ABC, abstractmethod
from typing import Any
from gymnasium import spaces


class BaseDifficulty(ABC):
    @abstractmethod
    def setup_obs_space(self, spatial_obs: spaces.Box) -> spaces.Dict:
        ...

    @abstractmethod
    def apply(self) -> dict[str, Any]:
        ...