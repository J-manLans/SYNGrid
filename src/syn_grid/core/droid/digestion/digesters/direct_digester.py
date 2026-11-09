"""
The direct digesters: what eating a direct orb pays.

A direct orb pays its own reward on the step it is eaten and changes nothing
else. The base holds that rule, and each direct kind has a subclass that only
names its kind, since digestion routes on the kind and readers look a digester
up by its class.
"""

from typing import Final

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectKind, OrbKind

# ################## #
# Digestion Strategy #
# ################## #


class DirectDigester:
    """The rules for a direct orb kind.

    Attributes:
        KIND: the orb kind this digester owns.
        eaten: how many orbs of the kind have been eaten this episode.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, kind: DirectKind):
        self.KIND: Final[OrbKind] = kind

        self.reset()

    def reset(self) -> None:
        """Clear the count, at the start of an episode."""

        self.eaten = 0

    # ================= #
    #        API        #
    # ================= #

    def digest(self, orb: BaseOrb) -> float:
        """Count an eaten orb and return its reward."""

        self.eaten += 1

        return orb.REWARD

    def undigested(self) -> float:
        """Return 0: a direct orb is paid when it is eaten."""

        return 0.0


class NegDigester(DirectDigester):
    """The direct digester for negative orbs."""

    def __init__(self):
        super().__init__(DirectKind.NEGATIVE)


class PosDigester(DirectDigester):
    """The direct digester for positive orbs."""

    def __init__(self):
        super().__init__(DirectKind.POSITIVE)
