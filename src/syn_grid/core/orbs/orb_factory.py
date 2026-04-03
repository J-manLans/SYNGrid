from syn_grid.config.models import OrbConf, OrbFactoryConf, NegativeConf, TierConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb


class OrbFactory:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        orb_manager_conf: OrbFactoryConf,
        negative_orb_conf: NegativeConf,
        tier_orb_conf: TierConf
    ):
        self.orb_manager_conf = orb_manager_conf
        self.negative_orb_conf = negative_orb_conf
        self.tier_orb_conf = tier_orb_conf
        self.max_active_orbs = orb_manager_conf.max_active_orbs
        self.max_tier = orb_manager_conf.max_tier

    # ================= #
    #        API        #
    # ================= #

    def create_orbs(self) -> list[BaseOrb]:
        """Create all orbs according to the config and weights"""
        enabled_orbs = self._get_enabled_orbs()
        total_weight = sum(enabled_orbs.values())

        # Shared setup
        BaseOrb.set_life_span(self.max_active_orbs, self.max_active_orbs)
        TierOrb.MAX_TIER = self.max_tier

        orbs: list[BaseOrb] = []
        for orb_type, weight in enabled_orbs.items():
            ratio = weight / total_weight
            count = self._compute_spawn_count(ratio)

            if orb_type == "negative":
                orbs.extend([NegativeOrb(self.negative_orb_conf) for _ in range(count)])
            elif orb_type == "tier":
                for tier in range(self.max_tier + 1):
                    orbs.extend([TierOrb(tier, self.tier_orb_conf) for _ in range(count)])

        return orbs

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_enabled_orbs(self) -> dict[str, int]:
        """Return enabled orb types and their weights from orb_manager_conf"""

        enabled_orbs = {}
        for orb_type, orb_conf in self.orb_manager_conf.types.items():
            if orb_conf.enabled:
                enabled_orbs[orb_type] = orb_conf.weight
        if not enabled_orbs:
            raise ValueError("At least one orb must be enabled")
        return enabled_orbs

    def _compute_spawn_count(self, ratio: float) -> int:
        """Compute how many orbs to spawn based on weight ratio"""

        return max(1, int((self.max_active_orbs * ratio) + 0.5))