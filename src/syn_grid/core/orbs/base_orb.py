from syn_grid.core.orbs.orb_meta import OrbMeta
from syn_grid.core.utils.timer import Timer

from abc import ABC
from typing import Final


class BaseOrb(ABC):
    # ================= #
    #       Init        #
    # ================= #

    is_active = False

    def __init__(
        self,
        reward: float,
        cool_down: int,
        meta: OrbMeta,
    ):
        self.position: list[int] = [-1, -1]
        self.REWARD: Final[float] = reward
        self._COOL_DOWN: Final[int] = cool_down
        self.META: Final[OrbMeta] = meta
        self.TIMER: Final[Timer] = Timer()

    @classmethod
    def set_life_span(cls, grid_rows: int, grid_cols: int) -> None:
        """
        Set the maximum lifespan based on the grid size.

        The lifespan is defined as the maximum number of steps required to traverse the grid from one corner to the opposite corner using Manhattan distance.

        :param grid_rows: Number of rows
        :param grid_cols: Number of columns
        """

        cls._life_span = (grid_rows - 1) + (grid_cols - 1)

    def reset(self) -> None:
        self.is_active = False
        self.TIMER.reset()

    # ================= #
    #        API        #
    # ================= #

    def spawn(self, position: list[int]) -> None:
        """Spawns the orb."""

        self.position = position
        self.is_active = True
        self.TIMER.set(self._life_span)

    def de_spawn(self) -> None:
        """Removes the orb without giving any reward."""

        self.is_active = False
        self.TIMER.set(self._COOL_DOWN)

    def consume(self) -> "BaseOrb":
        """Let's the droid consume the orb."""

        self.is_active = False
        self.TIMER.set(self._COOL_DOWN)
        return self
