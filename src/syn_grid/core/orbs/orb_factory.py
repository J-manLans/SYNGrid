from syn_grid.config.models import OrbConf, OrbFactoryConf, NegativeConf, TierConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb

from typing import Final


class OrbFactory:
    # ================= #
    #       Init        #
    # ================= #
    _MIN_POOL_SIZE: Final[int]

    def __init__(
        self,
        orb_manager_conf: OrbFactoryConf,
        negative_orb_conf: NegativeConf,
        tier_orb_conf: TierConf,
    ):
        self._MIN_POOL_SIZE = orb_manager_conf.max_active_orbs * 3
        self._orb_manager_conf = orb_manager_conf
        self._negative_orb_conf = negative_orb_conf
        self._tier_orb_conf = tier_orb_conf
        self._max_active_orbs = orb_manager_conf.max_active_orbs
        self._max_tier = orb_manager_conf.max_tier

    # ================= #
    #        API        #
    # ================= #

    def create_orbs(self) -> list[BaseOrb]:
        """Create all orbs according to the config and weights"""

        # Gets the ena—bed orbs and calculate their total weight
        enabled_orbs = self._get_conf_enabled_orbs()
        total_weight = sum(enabled_orbs.values())

        # Shared setup
        BaseOrb.set_life_span(self._max_active_orbs, self._max_active_orbs)
        TierOrb.MAX_TIER = self._max_tier

        # Calculate counts through ratios via orb weights
        ratios = [(orb_weight / total_weight) for orb_weight in enabled_orbs.values()]
        counts = self._scale_ratios_to_counts(ratios)
        counts = self._ensure_min_pool_size(counts, ratios)

        # Initialize orbs based on count per orb type
        orbs: list[BaseOrb] = []
        for i, orb_type in enumerate(enabled_orbs):
            if orb_type == "negative":
                orbs.extend(
                    [NegativeOrb(self._negative_orb_conf) for _ in range(counts[i])]
                )
            elif orb_type == "tier":
                # TODO: here I need to make sure my new tier system works. Should start at 1, not 0
                self._initialize_tier_orbs(orbs, counts[i])

        return orbs

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_conf_enabled_orbs(self) -> dict[str, int]:
        """Return enabled orb types and their weights from orb_manager_conf"""

        enabled_orbs = {}
        for orb_type, orb_conf in self._orb_manager_conf.types:
            if orb_conf.enabled:
                enabled_orbs[orb_type] = orb_conf.weight
        if not enabled_orbs:
            raise ValueError("At least one orb must be enabled")
        return enabled_orbs

    def _scale_ratios_to_counts(self, ratios: list[float]) -> list[int]:
        scaling_factor = 1 / min(ratios)
        counts = [max(1, int(ratio * scaling_factor)) for ratio in ratios]
        return counts

    def _ensure_min_pool_size(
        self, counts: list[int], ratios: list[float]
    ) -> list[int]:
        """Ensure total orb count meets minimum pool size by rescaling if needed."""

        if sum(counts) >= self._MIN_POOL_SIZE:
            return counts

        scaled = [self._MIN_POOL_SIZE * ratio for ratio in ratios]
        return self._normalize_counts(scaled)

    def _normalize_counts(self, counts: list[float]) -> list[int]:
        counts_int = [int(c) for c in counts]
        diff = self._MIN_POOL_SIZE - sum(counts_int)

        if diff == 0:
            return counts_int

        counts_int.sort()
        for i in range(diff):
            # Distribute +1 starting from the largest counts,
            # wrapping around if diff > len(counts)
            counts_int[-(i % len(counts_int) + 1)] += 1

        return counts_int

    def _initialize_tier_orbs(self, orbs: list[BaseOrb], count: int):
        # Default behavior when the projected total orb pool exceeds the minimum:
        # spawn one orb per tier and return early.
        if self._max_tier >= count:
            for tier in range(self._max_tier + 1):
                orbs.append(TierOrb(tier, self._tier_orb_conf))
            return

        # If total count can be evenly divided across tiers, spawn exactly that many orbs per tier.
        # Else, if total orbs cannot be evenly divided, distribute them one by one across tiers,
        # looping back to the first tier as needed.
        orbs_per_tier = count / (self._max_tier + 1)
        if orbs_per_tier.is_integer():
            for tier in range(self._max_tier + 1):
                for _ in range(int(orbs_per_tier)):
                    orbs.append(TierOrb(tier, self._tier_orb_conf))
        else:
            for i in range(count):
                tier = i % (self._max_tier + 1)
                orbs.append(TierOrb(tier, self._tier_orb_conf))
