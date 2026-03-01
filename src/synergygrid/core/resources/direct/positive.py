from synergygrid.core.resources import (
    BaseTierResource,
    ResourceMeta,
    ResourceCategory,
    DirectType,
)


class PositiveResource(BaseTierResource):
    """
    A resource that gives the agent a positive score.

    It also serves as a base for the tier resource chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_boundaries: tuple[int, int], cool_down: int = 5):
        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.DIRECT, type=DirectType.POSITIVE, tier=1
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()
        super()._resolve_tier_progression()
        self._is_restart_needed()
        return  self._TIER_BASE_REWARD #, thinking it might insentivice it to go for tier 2 and 3

    # ================= #
    #      Helpers      #
    # ================= #

    def _is_restart_needed(self):
        """If we just broke the chain, restart it since this is the base for the tier resources"""
        if len(self._chained_tiers) == 0:
            super()._chain_tier()
