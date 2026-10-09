"""The digestion contract.

Digestion turns "an orb was consumed" into a reward. It is agnostic about
scenarios: it only knows orb *kinds*. Each kind has one digester that owns the
rules for that kind, and the engine routes every consumed orb to the right one.

Nothing in this module mentions tiers, chains or scoring. Those belong to the
digesters that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectType, OrbCategory, SynergyType


# ################## #
#         API        #
# ################## #


def kind_of(orb: BaseOrb) -> OrbKind:
    return (orb.META.CATEGORY, orb.META.TYPE)


# ################## #
#        Event       #
# ################## #


@dataclass(frozen=True)
class Event:
    """Something that happened while digesting an orb.

    Digesters define their own events by subclassing this. The engine never
    looks inside one, it only records which type occurred.
    """


# ################## #
#     Interface      #
# ################## #


# What an orb *is*, as read from its META. This is what the engine routes on.
OrbKind: TypeAlias = tuple[OrbCategory, DirectType | SynergyType]


@dataclass(frozen=True)
class DigestionResult:
    """Outcome of digesting one orb: a reward plus whatever happened on the way."""

    reward: float
    events: tuple[Event, ...] = ()


class OrbDigester(Protocol):
    """The rules for one orb kind.

    A digester keeps its own episode state and never holds a reference to the
    engine or to other digesters, so each one can be tested on its own.
    """

    @property
    def kind(self) -> OrbKind:
        """The orb kind this digester owns."""
        ...

    def reset(self) -> None:
        """Start of episode: clear this digester's state."""
        ...

    def digest(self, orb: BaseOrb) -> DigestionResult:
        """Digest an orb of this digester's own kind."""
        ...

    def notice(self, orb: BaseOrb) -> tuple[Event, ...]:
        """React to an orb of *another* kind being consumed.

        This is how a negative orb can break a tier chain without the two
        digesters knowing about each other. Most digesters have nothing to
        react to and return an empty tuple.
        """
        ...
