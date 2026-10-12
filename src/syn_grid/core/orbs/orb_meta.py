"""
What an orb is: its kind, and its tier if it is a tier orb.

The meta says nothing about how an orb acts. It is what the rest of the code
reads to tell orbs apart: digestion routes on the kind, the renderer picks an
image from it, and the agent sees the identity.
"""

from enum import Enum
from typing import Final, TypeAlias


class OrbCategory(Enum):
    """The two groups of orb kinds. One digit of an orb's identity."""

    NONE = 0
    DIRECT = 1
    SYNERGY = 2


class DirectKind(Enum):
    """Orbs that change the score and nothing else, on the step they are eaten."""

    NONE = 0
    NEGATIVE = 1


class SynergyKind(Enum):
    """Orbs that leave something behind which changes a later outcome."""

    # TODO: when adding a new orb, ensure that ole MAX_SENTINEL for tier orbs wasn't needed. The
    # one introduced when the radix identity was added

    NONE = 0
    TIER = 1


# An orb's kind: a member of either enum. It is what digestion routes on.
OrbKind: TypeAlias = DirectKind | SynergyKind


class OrbMeta:
    """What one orb is.

    Attributes:
        KIND: the orb's kind, a member of `DirectKind` or `SynergyKind`.
        TIER: the orb's tier. An int for a tier orb and None for every other
            kind.
        IDENTITY: the number the agent sees for this orb. Orbs of the same kind
            and tier share it. It starts as the radix identity, and the world
            replaces it with a dense index when the world is built.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        kind: OrbKind,
        tier: int | None = None,
    ):
        self._assert_tier_matches_kind(kind, tier)

        self.KIND: Final[OrbKind] = kind
        self.TIER: Final[int | None] = tier
        self.IDENTITY = self.compute_radix_identity()

    # ================= #
    #      Helpers      #
    # ================= #

    def _assert_tier_matches_kind(
        self,
        kind: OrbKind,
        tier: int | None,
    ) -> None:
        """Check that a tier orb has a tier of at least 1, and that no other kind has one."""

        if not isinstance(kind, (DirectKind, SynergyKind)):
            raise TypeError("The orb's kind needs to be a direct or a synergy kind")

        if kind == SynergyKind.TIER:
            if tier is None:
                raise ValueError("A tier orb needs a tier")

            if tier < 1:
                raise ValueError("Tier orbs can't have tiers less than 1")

        elif tier is not None:
            raise ValueError("Only tier orbs have a tier")

    def compute_radix_identity(self) -> int:
        """
        Encode category, kind and tier as one number using mixed-radix encoding.

        Each of the three occupies its own positional slot, similar to how hours,
        minutes and seconds encode into a total number of seconds. The category is
        read from which enum the kind belongs to, and keeps a direct kind and a
        synergy kind with the same value apart. The tier is the highest slot, so
        sorting by identity orders orbs by tier first, then kind, then category.
        """

        category = (
            OrbCategory.DIRECT
            if isinstance(self.KIND, DirectKind)
            else OrbCategory.SYNERGY
        )

        return (
            category.value
            + ((len(OrbCategory) + 1) * self.KIND.value)
            + (
                ((len(OrbCategory) + 1) * (max(len(DirectKind), len(SynergyKind)) + 1))
                * (self.TIER or 0)
            )
        )
