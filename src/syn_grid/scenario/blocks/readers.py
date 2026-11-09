"""
The world readers every scenario can use, whatever its orbs.

A reader is a module-level function that is handed the world and returns a number. The HUD, the
metrics and the observation's global features each pair a static description with a reader, and
the same reader can serve all three. A family's own readers live beside its builder.
Module-level so that two scenarios built from the same config hold the same function objects and
compare equal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


def energy(world: GridWorld) -> float:
    """The droid's energy right now."""
    return world.droid.energy


def score(world: GridWorld) -> float:
    """The droid's score so far this episode."""
    return world.droid.score


def steps_left(world: GridWorld) -> float:
    """How many steps the episode has left."""
    return world.steps_left
