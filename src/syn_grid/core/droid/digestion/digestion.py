"""
The digestion contract.

Digestion turns a consumed orb into a reward. It knows orb kinds and nothing
about scenarios: each kind has one digester that owns the rules for it, and
every consumed orb is routed to the digester of its kind.
"""

from dataclasses import dataclass
from typing import Protocol, TypeAlias, Final

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectType, SynergyType

# ################## #
#       Result       #
# ################## #


@dataclass(frozen=True)
class Event:
    """Something that happened while an orb was digested.

    A digester defines its own events by subclassing this.
    """


@dataclass(frozen=True)
class DigestionResult:
    """What digesting one orb gave.

    Attributes:
        reward: what the orb was worth to the droid on this step.
        events: what happened while it was digested, in order.
    """

    reward: float
    events: tuple[Event, ...] = ()


# ################## #
#       Routing      #
# ################## #


# What an orb is routed on: its type. Digestion treats it as an opaque key.
OrbKind: TypeAlias = DirectType | SynergyType


# ################## #
#     Interface      #
# ################## #


class OrbDigester(Protocol):
    """The rules for one orb kind.

    A digester keeps its own episode state and holds no reference to the engine
    or to another digester.
    """

    KIND: Final[OrbKind]
    """The orb kind this digester owns."""


    def reset(self) -> None:
        """Clear this digester's episode state, at the start of an episode."""
        ...

    def digest(self, orb: BaseOrb) -> DigestionResult:
        """Digest an orb of this digester's kind and return what it gave."""
        ...
