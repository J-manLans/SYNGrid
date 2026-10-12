"""
How a scenario fills the orb pool.

The two root scenarios build their pools in genuinely different ways, and that
difference is not a parameter: a tier chain has exactly one orb per tier and no
weighting anywhere, while a continuous world apportions a weighted pool. So this
is two implementations behind one call, not one implementation with a flag.

This file is the call. The implementations are in ``scenario/blocks/populations/``,
and a scenario's builder hands one the values it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from syn_grid.core.orbs.base_orb import BaseOrb


# ################## #
#     Interface      #
# ################## #


class OrbPopulation(Protocol):
    """Makes one orb kind's orbs."""

    @property
    def count(self) -> int:
        """How many orbs `create` returns."""
        ...

    def create(self) -> list[BaseOrb]:
        """Return new orbs. Called once per world, so it holds no episode state."""
        ...
