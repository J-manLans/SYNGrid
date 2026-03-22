from abc import ABC, abstractmethod
import numpy as np
from synergygrid.core.resources import ResourceMeta
from typing import Final


class BaseResource(ABC):
    position = [np.int64(-1), np.int64(-1)]
    is_active = False

    class Timer:
        def __init__(self):
            self.remaining = 0

        def is_completed(self) -> bool:
            return self.remaining <= 0

        def set(self, duration: int) -> None:
            self.remaining = duration

        def tick(self) -> None:
            if self.remaining > 0:
                self.remaining -= 1

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        reward: int,
        cool_down: int,
        meta: ResourceMeta,
    ):
        self.REWARD = reward
        self._cool_down = cool_down
        self.meta = meta
        self.timer = self.Timer()

    @classmethod
    def set_life_span(cls, grid_rows: int, grid_cols: int) -> None:
        """
        Set the maximum lifespan based on the grid size.

        The lifespan is defined as the maximum number of steps required to traverse the grid from one corner to the opposite corner using Manhattan distance.

        :param grid_rows: Number of rows
        :param grid_cols: Number of columns
        """

        cls._LIFE_SPAN = (grid_rows - 1) + (grid_cols - 1)

    def reset(self) -> None:
        self.is_active = False
        self.timer.set(0)

    # ================= #
    #        API        #
    # ================= #

    def deplete_resource(self) -> None:
        """Removes the resource without giving any reward."""

        self.is_active = False
        self.timer.set(self._cool_down)

    def spawn(self, position: list[np.int64]):
        """Spawns the resource."""

        self.position = position
        self.is_active = True
        self.timer.set(self._LIFE_SPAN)

    # ================= #
    #      Abstract     #
    # ================= #

    @abstractmethod
    def consume(self) -> "BaseResource":
        """Let's the agent consume the resource."""

    # ================= #
    #      Helpers      #
    # ================= #

    def _consume(self) -> None:
        """Sets up the consume() method."""

        self.is_active = False
        self.timer.set(self._cool_down)
