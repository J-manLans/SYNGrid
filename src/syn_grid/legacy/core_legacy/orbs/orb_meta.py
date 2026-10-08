from enum import Enum
from typing import Final


class OrbCategory(Enum):
    NONE = 0
    DIRECT = 1
    SYNERGY = 2


class DirectType(Enum):
    NONE = 0
    NEGATIVE = 1


_MAX_SENTINEL = object()


class SynergyType(Enum):
    NONE = 0
    TIER = _MAX_SENTINEL

    def __new__(cls, value):
        """Ensures TIER always gets the highest value"""

        if value is _MAX_SENTINEL:
            value = max((m.value for m in cls), default=0) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class OrbMeta:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        category: OrbCategory,
        type: DirectType | SynergyType,
        tier: int | None = None,
    ):
        self._assert_type_and_tier_matches_category(category, type, tier)

        # These values are for finding correct image to render in PygameRenderer
        self.CATEGORY: Final[OrbCategory] = category
        self.TYPE: Final[DirectType | SynergyType] = type
        self.TIER: Final[int] = tier if (tier is not None) else 0
        # And this is used by the agent to identify orbs
        self.IDENTITY = self.compute_radix_identity()

    # ================= #
    #      Helpers      #
    # ================= #

    def _assert_type_and_tier_matches_category(
        self,
        category: OrbCategory,
        type: DirectType | SynergyType,
        tier: int | None,
    ) -> None:
        if category == OrbCategory.DIRECT:
            if not isinstance(type, DirectType):
                raise TypeError(
                    "If the category is DIRECT, the orb need to be of direct type"
                )
            if tier is not None:
                raise ValueError(
                    "If the category is DIRECT, tier should not be applied"
                )

        if category == OrbCategory.SYNERGY:
            if not isinstance(type, SynergyType):
                raise TypeError(
                    "If the category is SYNERGY, the orb need to be of synergy type"
                )

            if tier is None:
                raise ValueError("If the category is SYNERGY, tier should be applied")

            if tier < 1:
                raise ValueError("Tier orbs can't have tiers less than 1")

    def compute_radix_identity(self) -> int:
        """
        Encodes category, type and tier as a single unique scalar using mixed-radix encoding. Each dimension occupies its own positional slot, similar to how hours, minutes and seconds encode into a total number of seconds.
        """

        return (
            self.CATEGORY.value
            + ((len(OrbCategory) + 1) * self.TYPE.value)
            + (
                ((len(OrbCategory) + 1) * (max(len(DirectType), len(SynergyType)) + 1))
                * self.TIER
            )
        )
