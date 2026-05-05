from syn_grid.config.models import OrbFactoryConf, NegativeConf, TierConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb

from typing import Final


class OrbFactory:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        orb_factory_conf: OrbFactoryConf,
        negative_orb_conf: NegativeConf,
        tier_orb_conf: TierConf,
    ):
        self._conf = orb_factory_conf
        self._min_pool_size = self._conf.max_active_orbs * 3
        self.negative_orb_conf = negative_orb_conf
        self.tier_orb_conf = tier_orb_conf

    # ================= #
    #        API        #
    # ================= #

    def create_orbs(self) -> list[BaseOrb]:
        """Create all orbs according to the config and weights"""

        # Shared setup
        BaseOrb.set_life_span(self._conf.grid_rows, self._conf.grid_cols)
        TierOrb.max_tier = self._conf.max_tier

        if self._conf.single_chain_mode:
            return self._single_chain_mode_orbs()

        # Gets the ena—bed orbs and calculate their total weight
        enabled_orbs = self._get_conf_enabled_orbs()
        total_weight = sum(enabled_orbs.values())

        # Calculate counts through ratios via orb weights
        ratios = [(orb_weight / total_weight) for orb_weight in enabled_orbs.values()]
        orb_counts = self._scale_ratios_to_counts(ratios)
        orb_counts = self._ensure_min_pool_size(orb_counts, ratios)

        # Initialize orbs based on count per orb type
        orbs: list[BaseOrb] = []
        for i, orb_type in enumerate(enabled_orbs):
            if orb_type == "negative":
                orbs.extend(
                    [NegativeOrb(self.negative_orb_conf) for _ in range(orb_counts[i])]
                )
            elif orb_type == "tier":
                self._initialize_tier_orbs(orbs, orb_counts[i])

        return orbs

    # ================= #
    #      Helpers      #
    # ================= #

    def _single_chain_mode_orbs(self) -> list[BaseOrb]:
        orbs: list[BaseOrb] = []
        for tier in range(1, self._conf.max_tier + 1):
            orbs.append(TierOrb(tier, self.tier_orb_conf))

        return orbs

    def _get_conf_enabled_orbs(self) -> dict[str, int]:
        """Return enabled orb types and their weights from orb_manager_conf"""

        enabled_orbs = {}
        for orb_type, orb_conf in self._conf.types:
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

        if sum(counts) >= self._min_pool_size:
            return counts

        scaled = [self._min_pool_size * ratio for ratio in ratios]
        return self._normalize_counts(scaled)

    def _normalize_counts(self, counts: list[float]) -> list[int]:
        counts_int = [int(c) for c in counts]
        diff = self._min_pool_size - sum(counts_int)

        if diff == 0:
            return counts_int

        counts_int.sort()
        for i in range(diff):
            # Distribute +1 starting from the largest counts,
            # wrapping around if diff > len(counts)
            counts_int[-(i % len(counts_int) + 1)] += 1

        return counts_int

    def _initialize_tier_orbs(self, orbs: list[BaseOrb], orb_count: int):
        # Default behavior when the projected total orb pool exceeds the minimum:
        # spawn one orb per tier and return early.
        if self._conf.max_tier >= orb_count:
            for tier in range(1, self._conf.max_tier + 1):
                orbs.append(TierOrb(tier, self.tier_orb_conf))
            return

        # If total count can be evenly divided across tiers, spawn exactly that many orbs per tier.
        # Else, if total orbs cannot be evenly divided, distribute them one by one across tiers,
        # looping back to the first tier as needed.
        orbs_per_tier = orb_count / (self._conf.max_tier)
        if orbs_per_tier.is_integer():
            for tier in range(1, self._conf.max_tier + 1):
                for _ in range(int(orbs_per_tier)):
                    orbs.append(TierOrb(tier, self.tier_orb_conf))
        else:
            for i in range(orb_count):
                tier = (i % self._conf.max_tier) + 1
                orbs.append(TierOrb(tier, self.tier_orb_conf))
