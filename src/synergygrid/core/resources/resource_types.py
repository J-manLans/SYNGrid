from enum import Enum


class ResourceCategory(Enum):
    DIRECT = 0
    SYNERGY = 1


class SynergyType(Enum):
    TIER = 0


class DirectType(Enum):
    NEGATIVE = 0
    POSITIVE = 1


class ResourceMeta:
    def __init__(
        self,
        category: ResourceCategory,
        type: DirectType | SynergyType,
        tier: int | None = None,
    ):
        # For finding correct image to render together with subtype and helping the agent identifying the resource resource
        self.category = category
        self.type = type  # Same as above, just subtype
        # Resources tier.
        # 0 if not applicable
        # ...n for rest of the tier resources
        self.tier = 0 if tier == None else tier
