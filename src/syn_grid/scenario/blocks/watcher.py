"""
What a scenario does after the world's step.

A watcher is handed the world and what its step reported, and acts on the world through its
public methods and the orbs' own. It is how a scenario reacts to something that happened in the
world, such as an orb being eaten, without the world holding any of the scenario's code.

A watcher holds episode state, so a scenario holds a way to make one and each environment makes
its own. This file is the interface and the default that never acts. A scenario's own watcher
lives in its family's folder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld, StepResult


# ################## #
#     Interface      #
# ################## #


class Watcher(Protocol):
    """Reacts to what the world's step reported."""

    def reset(self) -> None:
        """Clear this watcher's episode state, at the start of an episode."""
        ...

    def react(self, world: GridWorld, result: StepResult) -> None:
        """Act on the world after its step. Called once per step, before termination and the
        observation."""
        ...


# ################## #
#      Default       #
# ################## #


class NoWatcher:
    """A watcher that never acts. It is the watcher of every scenario that states none."""

    def reset(self) -> None:
        pass

    def react(self, world: GridWorld, result: StepResult) -> None:
        pass
