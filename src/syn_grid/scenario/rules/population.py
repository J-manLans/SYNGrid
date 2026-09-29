"""
How a scenario fills the orb pool.

The two root scenarios build their pools in genuinely different ways, and that
difference is not a parameter -- a tier chain has exactly one orb per tier and
no weighting anywhere, while a continuous world apportions a weighted pool.
So this is two implementations behind one call, not one implementation with a
flag.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from syn_grid.config.models import NegativeConf, OrbFactoryConf, TierConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb


@runtime_checkable
class OrbPopulation(Protocol):
    """Builds the world's orb pool."""

    def create(self) -> list[BaseOrb]: ...


class TierChainPopulation:
    """One orb per tier, tiers 1..max_tier, nothing weighted.

    A tier chain is a fixed sequence, so the pool is the sequence. ``max_tier``
    is the pool size, which is why a tier chain spawns ``max_tier`` orbs at
    reset rather than a configured count.
    """

    def __init__(self, max_tier: int, tier_conf: TierConf) -> None:
        self._max_tier = max_tier
        self._tier_conf = tier_conf

    def create(self) -> list[BaseOrb]:
        return [
            TierOrb(tier, self._tier_conf, self._max_tier)
            for tier in range(1, self._max_tier + 1)
        ]


class WeightedPopulation:
    """A weighted pool over the enabled orb types.

    Counts come from the configured weights, scaled so the rarest type gets at
    least one orb and the total is at least ``max_active_orbs * 3`` so there is
    always something to spawn.
    """

    def __init__(
        self,
        orb_factory_conf: OrbFactoryConf,
        negative_conf: NegativeConf,
        tier_conf: TierConf,
    ) -> None:
        self._conf = orb_factory_conf
        self._negative_conf = negative_conf
        self._tier_conf = tier_conf
        self._min_pool_size = orb_factory_conf.max_active_orbs * 3

    def create(self) -> list[BaseOrb]:
        enabled_orbs = self._get_conf_enabled_orbs()
        total_weight = sum(enabled_orbs.values())

        ratios = [orb_weight / total_weight for orb_weight in enabled_orbs.values()]
        orb_counts = self._scale_ratios_to_counts(ratios)
        orb_counts = self._ensure_min_pool_size(orb_counts, ratios)

        orbs: list[BaseOrb] = []
        for i, orb_type in enumerate(enabled_orbs):
            if orb_type == "negative":
                orbs.extend(
                    [NegativeOrb(self._negative_conf) for _ in range(orb_counts[i])]
                )
            elif orb_type == "tier":
                self._initialize_tier_orbs(orbs, orb_counts[i])

        return orbs

    # === Helpers === #

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
        return [max(1, int(ratio * scaling_factor)) for ratio in ratios]

    def _ensure_min_pool_size(
        self, counts: list[int], ratios: list[float]
    ) -> list[int]:
        """Ensure total orb count meets minimum pool size by rescaling if needed."""

        if sum(counts) >= self._min_pool_size:
            return counts

        scaled = [self._min_pool_size * ratio for ratio in ratios]
        return self._normalize_counts(scaled)

    def _normalize_counts(self, counts: list[float]) -> list[int]:
        """
        Index correspondence is load-bearing: counts[i] is built from the weight of the i-th enabled orb type and the caller reads the result back the same way, so the returned list must not be reordered. Rank the indices, never sort the counts themselves.
        """

        counts_int = [int(c) for c in counts]
        diff = self._min_pool_size - sum(counts_int)

        if diff == 0:
            return counts_int

        # Largest-remainder apportionment. Rank by fractional part only --
        # counts_int itself must never be reordered, because index i has to keep
        # referring to the orb type it was given on entry.
        remainders = [c - int(c) for c in counts]
        order = sorted(range(len(counts)), key=lambda i: remainders[i], reverse=True)

        for k in range(abs(diff)):
            counts_int[order[k % len(order)]] += 1

        return counts_int

    def _initialize_tier_orbs(self, orbs: list[BaseOrb], orb_count: int) -> None:
        # Default behavior when the projected total orb pool exceeds the minimum:
        # spawn one orb per tier and return early.
        if self._conf.max_tier >= orb_count:
            for tier in range(1, self._conf.max_tier + 1):
                orbs.append(TierOrb(tier, self._tier_conf, self._conf.max_tier))
            return

        # If total count can be evenly divided across tiers, spawn exactly that many orbs per tier.
        # Else, if total orbs cannot be evenly divided, distribute them one by one across tiers,
        # looping back to the first tier as needed.
        orbs_per_tier = orb_count / (self._conf.max_tier)
        if orbs_per_tier.is_integer():
            for tier in range(1, self._conf.max_tier + 1):
                for _ in range(int(orbs_per_tier)):
                    orbs.append(TierOrb(tier, self._tier_conf, self._conf.max_tier))
        else:
            for i in range(orb_count):
                tier = (i % self._conf.max_tier) + 1
                orbs.append(TierOrb(tier, self._tier_conf, self._conf.max_tier))
