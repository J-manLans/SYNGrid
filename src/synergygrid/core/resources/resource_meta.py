from enum import Enum


class ResourceCategory(Enum):
    DIRECT = 0
    SYNERGY = 1


class SynergyType(Enum):
    TIER = 1


class DirectType(Enum):
    NEGATIVE = 0


class ResourceMeta:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        category: ResourceCategory,
        type: DirectType | SynergyType,
        tier: int | None = None,
    ):
        self._assert_type_matches_category(category, type, tier)
        # For finding correct image to render together with type and helping the agent identifying the resource
        self.category = category
        self.type = type  # Same as above

        if not tier == None and tier < 0:
            raise ValueError("Tier can't be less than 0")
        # Resources tier.
        # 0 if not applicable
        # ...n for rest of the tier resources
        # this is so the observation stays consistent
        self.tier = tier if tier is not None else -1

    # ================= #
    #      Helpers      #
    # ================= #

    def _assert_type_matches_category(
        self,
        category: ResourceCategory,
        type: DirectType | SynergyType,
        tier: int | None,
    ) -> None:
        if category == ResourceCategory.DIRECT:
            if not isinstance(type, DirectType):
                raise TypeError(
                    "If the category is DIRECT, the resource need to be of direct type"
                )
            if tier is not None:
                raise ValueError(
                    "If the category is DIRECT, tier should not be applied"
                )

        if category == ResourceCategory.SYNERGY:
            if not isinstance(type, SynergyType):
                raise TypeError(
                    "If the category is SYNERGY, the resource need to be of synergy type"
                )
            if tier is None:
                raise ValueError("If the category is SYNERGY, tier should be applied")
