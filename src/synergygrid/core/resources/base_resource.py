from abc import ABC, abstractmethod
import numpy as np
from synergygrid.core.resources import ResourceMeta
from typing import Final


class BaseResource(ABC):
    position = [np.int64(-1), np.int64(-1)]
    is_active = False
    _chained_tiers: Final[list[int]] = []

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
        world_boundaries: tuple[int, int],
        cool_down: int,
        meta: ResourceMeta,
    ):
        if world_boundaries[0] < 1 or world_boundaries[1] < 1:
            raise ValueError("grid_cols and grid_rows should be larger than 0")

        self._world_boundaries = world_boundaries
        # Max steps needed to reach resource diagonally anywhere on the grid
        self._LIFE_SPAN = (world_boundaries[0] - 1) + (world_boundaries[1] - 1)
        self._cool_down = cool_down
        self.meta = meta
        self.timer = self.Timer()

    def reset(self):
        self.is_active = False
        self.timer.set(0)
        self._chained_tiers.clear()

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
    def consume(self) -> int:
        """Defines how an agent interacts with the resource."""

    # ================= #
    #      Helpers      #
    # ================= #

    def _consume(self):
        """Sets up the consume() method."""

        self.is_active = False
        self.timer.set(self._cool_down)

    def _break_tier_chain(self) -> None:
        """Clears the tier list and return the given reward."""

        self._chained_tiers.clear()
