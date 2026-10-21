"""
What a scenario shows on the HUD.

A scenario lists its HUD elements; the renderer knows how to draw each kind.
Every element is a static description plus a reader that is handed the world.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


@dataclass(frozen=True)
class HudStat:
    """A labelled number.

    Attributes:
        label: the name shown for the value.
        read: returns the current value from the world it is handed.
    """

    label: str
    read: Callable[[GridWorld], float]


@dataclass(frozen=True)
class HudBar:
    """A labelled bar that fills toward a maximum.

    Attributes:
        label: the name shown for the bar.
        maximum: the value at which the bar is full.
        read: returns the current value from the world it is handed.
    """

    label: str
    maximum: float
    read: Callable[[GridWorld], float]


HudElement = HudStat | HudBar


# ----------------- #
#   World readers   #
# ----------------- #

# What every scenario can show, whatever its orbs. A family's own readers live beside its
# builder. Module-level so that two scenarios built from the same config hold the same function
# objects and compare equal.


def energy(world: GridWorld) -> float:
    """The droid's energy right now."""
    return world.droid.energy


def score(world: GridWorld) -> float:
    """The droid's score so far this episode."""
    return world.droid.score


def steps_left(world: GridWorld) -> float:
    """How many steps the episode has left."""
    return world.steps_left
