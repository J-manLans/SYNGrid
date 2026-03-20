from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.core.resources.resource_meta import ResourceMeta
from abc import ABC
from typing import Final


class BaseTierResource(BaseResource, ABC):
    _TIER_BASE_REWARD: Final[int] = 2
    MAX_TIER: int

    def _resolve_tier_progression(self) -> bool:
        """Checks if tier progression is correct and either continue the chain or breaks it."""

        current_tier = self.meta.tier

        # Start chain if first tier
        if len(self._chained_tiers) == 0:
            if current_tier == 1:
                self._chain_tier()
                return True
            return False

        # If the last tier added was one less that current resources tier, continue the chain and
        # return True
        if self._chained_tiers[-1] == current_tier - 1:
            if self.meta.tier == self.MAX_TIER:
                super()._break_tier_chain()
            else:
                self._chain_tier()
            return True

        # If it wasn't, break the chain, check if the resource is a tier 1, if so — restart the chain, then return False
        super()._break_tier_chain()
        if current_tier == 1:
            self._chain_tier()
        return False

    def _chain_tier(self) -> None:
        """Adds the tier of the resource to the tiers list."""

        self._chained_tiers.append(self.meta.tier)
