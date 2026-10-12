"""
The negative digester: what eating a negative orb costs.

A negative orb pays its own reward on the step it is eaten and changes nothing
else. The digester counts how many were eaten.
"""

from typing import Final

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectKind, OrbKind

# ################## #
# Digestion Strategy #
# ################## #


class NegDigester:
    """The rules for negative orbs.

    Attributes:
        KIND: the orb kind this digester owns.
        negatives_eaten: how many negative orbs have been eaten this episode.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self):
        self.KIND: Final[OrbKind] = DirectKind.NEGATIVE

        self.reset()

    def reset(self) -> None:
        """Clear the counter, at the start of an episode."""

        self.negatives_eaten = 0

    # ================= #
    #        API        #
    # ================= #

    def digest(self, orb: BaseOrb) -> float:
        """Count an eaten negative orb and return its reward."""

        self.negatives_eaten += 1

        return orb.REWARD
