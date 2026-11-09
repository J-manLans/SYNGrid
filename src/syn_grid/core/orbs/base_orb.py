from abc import ABC
from typing import Final

from syn_grid.core.orbs.orb_meta import OrbMeta
from syn_grid.core.utils.timer import Timer


class BaseOrb(ABC):
    """
    An orb that is either on the field or off it. It carries its reward, its own cool-down and lifespan, and one
    timer that counts whichever of the two applies. It knows when it has expired and when it is ready to return,
    and leaves placing it to the world.
    """

    # ================= #
    #       Init        #
    # ================= #

    is_active = False

    def __init__(
        self,
        reward: float,
        cool_down: int | None,
        life_span: int | None,
        meta: OrbMeta,
    ):
        self.position: list[int] = [-1, -1]
        self.REWARD: Final[float] = reward
        # A cool-down of None is an orb that does not return in the episode, and a lifespan of None
        # is an orb that does not expire: in both cases the timer never completes.
        self._COOL_DOWN: Final[int | None] = cool_down
        self._LIFE_SPAN: Final[int | None] = life_span
        self.META: Final[OrbMeta] = meta
        self.TIMER: Final[Timer] = Timer()

    def reset(self) -> None:
        """Take the orb off the field and make it ready to spawn."""

        self.is_active = False
        self.TIMER.reset()

    # ================= #
    #        API        #
    # ================= #

    def spawn(self, position: list[int]) -> None:
        """Spawns the orb."""

        self.position = position
        self.is_active = True
        self.TIMER.set(self._LIFE_SPAN)

    def de_spawn(self) -> None:
        """Removes the orb without giving any reward."""

        self.is_active = False
        self.TIMER.set(self._COOL_DOWN)

    def consume(self) -> "BaseOrb":
        """Let's the droid consume the orb."""

        self.is_active = False
        self.TIMER.set(self._COOL_DOWN)
        return self

    def has_expired(self) -> bool:
        """Return whether the orb is on the field and its lifespan has run out."""

        return self.is_active and self.TIMER.is_completed()

    def is_ready(self) -> bool:
        """Return whether the orb is off the field and its cool-down has run out."""

        return not self.is_active and self.TIMER.is_completed()
