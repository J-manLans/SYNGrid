from synergygrid.core.resources.base_tier_resource import BaseTierResource
from synergygrid.core.resources.resource_meta import ResourceMeta
from synergygrid.core.resources.resource_meta import ResourceCategory
from synergygrid.core.resources.resource_meta import SynergyType


class TierBase(BaseTierResource):
    """
    A resource that gives the agent a positive score.

    It also serves as a base for the tier resource chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, tier: int, world_boundaries: tuple[int, int], cool_down: int = 5
    ):
        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.SYNERGY, type=SynergyType.TIER_BASE, tier=tier
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()
        super()._resolve_tier_progression()
        return self._TIER_BASE_REWARD
