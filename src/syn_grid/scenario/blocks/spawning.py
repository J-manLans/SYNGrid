"""
How a scenario drives the orb field over the course of an episode.

The world owns the mechanics (spawn, cooldown, reactivate). A rule only decides
which of them to invoke and when.
"""

# ruff: noqa: PIE790

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# ################## #
#     Interface      #
# ################## #


class OrbFieldUpdate(Protocol):
    """What the orb field does once a step's worth of movement is resolved."""

    def apply(self, world: GridWorld) -> None: ...


# ################## #
#       Rules        #
# ################## #


@dataclass(frozen=True)
class SpawningRules:
    """The scenario's orb-field rules.

    Frozen so that two rules built for the same world compare equal by value.

    Attributes:
        spawn_max_on_episode_start: spawn the whole field at episode start.
        max_active_orbs: how large the active field is allowed to get. In a
            tier chain it is the whole chain, `max_tier`.
        tier_orb_expires: whether tier orbs run down and despawn on their timer
            like direct orbs do. A tier chain cannot allow it, because a chain
            that loses a link is not a chain.
        delay_on_consume: steps the whole field goes on cooldown after an orb is
            consumed, or None for no delay.
    """

    spawn_max_on_episode_start: bool
    max_active_orbs: int
    tier_orb_expires: bool
    # _after_action: OrbFieldUpdate # Will be added once I wired the scenario to the environment

    def on_reset(self, world: GridWorld) -> None:
        """Fill the field at the start of an episode."""
        ...

    def on_orb_consumed(self, world: GridWorld) -> None:
        """React to an orb being consumed, for instance by starting a delay."""
        ...

    def after_step(self, world: GridWorld) -> None:
        """What the field does once a step's movement is resolved."""
        ...
