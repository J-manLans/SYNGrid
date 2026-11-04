"""
What the delay scenario does after the world's step.

This is the single tier chain family's implementation of ``Watcher``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld, StepResult
    from syn_grid.core.orbs.base_orb import BaseOrb
    from syn_grid.core.orbs.orb_meta import OrbKind


# ################## #
#      Watcher       #
# ################## #


class DelayWatcher:
    """Freezes the other orbs of an eaten orb's kind, and releases them after the delay.

    Asleep, it does nothing. When the droid eats an orb, it freezes every orb of the same kind on
    the field: each leaves the field and keeps its cell. It counts one step each time it reacts,
    and when the delay has passed it releases each frozen orb whose cell has neither the droid nor
    an orb on the field in it. An orb whose cell is taken stays frozen until the cell clears. When
    every orb is released it sleeps again. While orbs are frozen, an orb eaten of another kind
    changes nothing.

    A frozen orb is taken off the field with `de_spawn`, which keeps its position, and brought
    back with `spawn`. That holds only for an orb with no cool-down, which the world never brings
    back on its own. Tier orbs have none, and eating a tier orb freezes only tier orbs. In a single
    tier chain the negative orb is the only one of its kind, so eating it freezes nothing.

    Attributes:
        delay: how many steps the frozen orbs stay off the field before they can return.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, delay: int):
        self._DELAY: Final[int] = delay

    def reset(self) -> None:
        """Release nothing and forget every frozen orb, at the start of an episode. The world's
        own reset puts the orbs back."""

        self._frozen_orbs: list[BaseOrb] = []
        self._steps_left: int = 0

    # ================= #
    #        API        #
    # ================= #

    def react(self, world: GridWorld, result: StepResult) -> None:
        """Count the delay down and release what it can, or freeze the eaten orb's kind."""

        if self._frozen_orbs:
            self._steps_left = max(self._steps_left - 1, 0)

            if self._steps_left == 0:
                self._release_where_free(world)
        elif result.consumed is not None:
            self._freeze(world, result.consumed.KIND)

    # ================= #
    #      Helpers      #
    # ================= #

    def _freeze(self, world: GridWorld, kind: OrbKind) -> None:
        self._frozen_orbs = [
            orb for orb in world.orbs_on_field() if orb.META.KIND == kind
        ]

        for orb in self._frozen_orbs:
            orb.de_spawn()

        self._steps_left = self._DELAY

    def _release_where_free(self, world: GridWorld) -> None:
        still_frozen = []

        for orb in self._frozen_orbs:
            if world.is_cell_free(orb.position):
                orb.spawn(orb.position)
            else:
                still_frozen.append(orb)

        self._frozen_orbs = still_frozen


# ################## #
#  Watcher Recipe    #
# ################## #


@dataclass(frozen=True)
class DelayWatcherRecipe:
    """Makes a new delay watcher each time it is called.

    It is what the delay scenario holds as `make_watcher`, since a watcher holds episode state and
    each environment needs its own. Frozen so that two recipes with the same delay compare equal.

    Attributes:
        delay: how many steps the frozen orbs stay off the field before they can return.
    """

    delay: int

    def __call__(self) -> DelayWatcher:
        return DelayWatcher(self.delay)
