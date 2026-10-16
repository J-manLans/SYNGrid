"""
The digestion contract.

Digestion turns a consumed orb into a reward. It knows orb kinds and nothing
about scenarios: each kind has one digester that owns the rules for it, and
every consumed orb is routed to the digester of its kind.
"""

from typing import Final, Protocol

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbKind

# ################## #
#     Interface      #
# ################## #


class OrbDigester(Protocol):
    """The rules for one orb kind.

    A digester keeps its own episode state and holds no reference to the engine
    or to another digester.
    """

    KIND: Final[OrbKind]
    """The orb kind this digester owns. Digestion routes on it and treats it as an opaque key."""

    def reset(self) -> None:
        """Clear this digester's episode state, at the start of an episode."""
        ...

    def digest(self, orb: BaseOrb) -> float:
        """Digest an orb of this digester's kind and return its reward."""
        ...

    def undigested(self) -> float:
        """Return what this digester has gathered and not paid, if the episode ends now.

        It changes nothing. A digester that pays every orb when it is eaten
        returns 0, and so does one whose unpaid sum is lost when the episode
        ends.
        """
        ...
