"""
What an orb is: its type, and its tier if it is a tier orb.

The meta says nothing about how an orb acts. It is what the rest of the code
reads to tell orbs apart: digestion routes on the type, the renderer picks an
image from it, and the agent sees the identity.
"""

from enum import Enum
from typing import Final


class OrbCategory(Enum):
    """The two groups of orb types. One digit of an orb's identity."""

    NONE = 0
    DIRECT = 1
    SYNERGY = 2


class DirectType(Enum):
    """Orbs that change the score and nothing else, on the step they are eaten."""

    NONE = 0
    NEGATIVE = 1


class SynergyType(Enum):
    """Orbs that leave something behind which changes a later outcome."""

    # TODO: when adding a new orb, ensure that ole MAX_SENTINEL for tier orbs wasn't needed. The
    # one introduced when the radix identity was added

    NONE = 0
    TIER = 1


class OrbMeta:
    """What one orb is.

    Attributes:
        TYPE: the orb's type, a member of `DirectType` or `SynergyType`.
        TIER: the orb's tier. An int for a tier orb and None for every other
            type.
        IDENTITY: the number the agent sees for this orb. Orbs of the same type
            and tier share it. It starts as the radix identity, and the world
            replaces it with a dense index when the world is built.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        type: DirectType | SynergyType,
        tier: int | None = None,
    ):
        self._assert_tier_matches_type(type, tier)

        self.TYPE: Final[DirectType | SynergyType] = type
        self.TIER: Final[int | None] = tier
        self.IDENTITY = self.compute_radix_identity()

    # ================= #
    #      Helpers      #
    # ================= #

    def _assert_tier_matches_type(
        self,
        type: DirectType | SynergyType,
        tier: int | None,
    ) -> None:
        """Check that a tier orb has a tier of at least 1, and that no other type has one."""

        if not isinstance(type, (DirectType, SynergyType)):
            raise TypeError("The orb's type needs to be a direct or a synergy type")

        if type == SynergyType.TIER:
            if tier is None:
                raise ValueError("A tier orb needs a tier")

            if tier < 1:
                raise ValueError("Tier orbs can't have tiers less than 1")

        elif tier is not None:
            raise ValueError("Only tier orbs have a tier")

    def compute_radix_identity(self) -> int:
        """
        Encode category, type and tier as one number using mixed-radix encoding.

        Each of the three occupies its own positional slot, similar to how hours,
        minutes and seconds encode into a total number of seconds. The category is
        read from which enum the type belongs to, and keeps a direct type and a
        synergy type with the same value apart. The tier is the highest slot, so
        sorting by identity orders orbs by tier first, then type, then category.
        """

        category = (
            OrbCategory.DIRECT
            if isinstance(self.TYPE, DirectType)
            else OrbCategory.SYNERGY
        )

        return (
            category.value
            + ((len(OrbCategory) + 1) * self.TYPE.value)
            + (
                ((len(OrbCategory) + 1) * (max(len(DirectType), len(SynergyType)) + 1))
                * (self.TIER or 0)
            )
        )
