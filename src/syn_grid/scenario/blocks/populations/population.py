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
    """Makes one orb kind's orbs.

    A population is a frozen description of how many orbs of one kind a world gets and what each
    one is like. A scenario's builder makes it from plain values. It reads no config, knows no
    scenario and holds no episode state, so two scenarios built from the same config hold equal
    populations.
    """

    @property
    def count(self) -> int:
        """How many orbs `create` returns, known without making any.

        A scenario is built before any world exists, and it states the size of its orb side while it
        is built: how many orbs the field holds, how many slots the observation has for them and
        how many of them the observation sorts. A family with a fixed field adds up the counts of
        its bundles for those numbers, so a kind that joins a scenario needs no number added by
        hand. A family with a sampled field takes the number on the field at once from its config,
        and the counts only size the pool. Calling `create` to count would make orbs only to throw
        them away.
        """
        ...

    def create(self) -> list[BaseOrb]:
        """Return `count` new orbs.

        The world calls it once, when it is built. Each call returns new orbs, so no two worlds
        share one. The orbs come back off the field, and the world places them.
        """
        ...
