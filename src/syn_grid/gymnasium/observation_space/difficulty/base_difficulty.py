from abc import ABC, abstractmethod


class BaseDifficulty(ABC):
    @abstractmethod
    def get_max_values(self) -> list[int]: ...
