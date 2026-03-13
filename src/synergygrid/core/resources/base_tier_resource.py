from synergygrid.core.resources import BaseResource
from typing import Final


class BaseTierResource(BaseResource):
    _TIER_BASE_REWARD: Final[int] = 3

    def _resolve_tier_progression(self) -> bool:
        """Checks if tier progression is correct and either continue the chain or breaks it."""

        # If no tiers have been chained yet, just return False
        if len(self._chained_tiers) == 0:
            return False

        # If the last tier added was one less that current resources tier, continue the chain and
        # return True
        if self._chained_tiers[-1] == self.meta.tier - 1:
            self._chain_tier()
            return True

        # If it wasn't, break the chain and return False
        super()._break_tier_chain()
        return False

    def _chain_tier(self) -> None:
        """Adds the tier of the resource to the tiers list."""

        self._chained_tiers.append(self.meta.tier)
