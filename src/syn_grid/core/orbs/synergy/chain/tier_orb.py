from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    OrbMeta,
    SynergyKind,
)


class TierOrb(BaseOrb):
    """
    An orb that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 orb a tier 1 and tier 2 must have first been collected in that order without breaking the chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        tier: int,
        max_tier: int,
        base_reward: float,
        growth_factor: float,
        cool_down: int | None,
        life_span: int | None,
    ):
        if tier > max_tier:
            raise ValueError("Tier is higher than the allowed max")

        self._tier_base_reward = base_reward
        self._growth_factor = growth_factor

        super().__init__(
            self._calculate_reward(tier),
            cool_down,
            life_span,
            OrbMeta(SynergyKind.TIER, tier),
        )

    # ================= #
    #      Helpers      #
    # ================= #

    def _calculate_reward(self, tier: int) -> float:
        """Return what an orb of this tier is worth.

        The base reward times the tier raised to the growth factor. A growth
        factor of 1 makes the rewards grow linearly with the tier.
        """

        return self._tier_base_reward * tier**self._growth_factor
