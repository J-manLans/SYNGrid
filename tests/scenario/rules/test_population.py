"""Weighted orb population, and the factory that delegates to it.

These assert real apportionment behaviour -- minimum pool size, weight-proportional
shares, even tier spread -- and all of it belongs to ``WeightedPopulation`` now.
The factory kept the class-attribute setup, which is a separate contract and is
tested at the bottom.
"""

from collections import Counter

import pytest

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_factory import OrbFactory
from syn_grid.core.orbs.synergy.tier_orb import TierOrb
from syn_grid.scenario.rules.population import (
    TierChainPopulation,
    WeightedPopulation,
)
from tests.utils.config_helpers import get_test_config, update_conf


class TestWeightedPopulation:
    @pytest.fixture
    def population(self) -> WeightedPopulation:
        return self._make(max_tier=1, max_active_orbs=3)

    # ================= #
    #       Tests       #
    # ================= #

    @pytest.mark.parametrize("max_tier", [1, 2, 3, 4, 5, 6])
    def test_fills_to_min_pool_size_with_limited_active_orbs(self, max_tier: int):
        population = self._make(max_tier=max_tier, max_active_orbs=3)
        orbs = population.create()

        actual_tier_counts = Counter(orb.META.TIER for orb in orbs)
        num_neg_orbs = actual_tier_counts.pop(0)
        expected_tier_counts = self._expected_tier_counts(
            max_tier, (len(orbs) - num_neg_orbs)
        )

        assert actual_tier_counts == expected_tier_counts
        assert len(orbs) == population._min_pool_size

    @pytest.mark.parametrize("max_tier", [i for i in range(100, 120)])
    def test_orbs_one_per_tier_after_min_pool_with_limited_active_orbs(
        self, max_tier: int
    ):
        population = self._make(max_tier=max_tier, max_active_orbs=3)
        orbs = population.create()

        assert len(orbs) == max_tier + 3

    @pytest.mark.parametrize("max_active_orbs", [i for i in range(1, 10)])
    def test_respects_different_max_active_orbs(self, max_active_orbs: int):
        population = self._make(max_tier=1, max_active_orbs=max_active_orbs)
        orbs = population.create()

        assert len(orbs) == max_active_orbs * 3

    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(1, 10), (1, 2), (1, 3), (1, 5), (4, 20), (30, 23123)],
    )
    def test_tier_orb_counts_follow_weight_ratios(self, neg_weight, tier_weight):
        population = self._make(neg_weight=neg_weight, tier_weight=tier_weight)
        orbs = population.create()

        actual_tier_counts = Counter(orb.META.TIER for orb in orbs)
        num_neg_orbs = actual_tier_counts.pop(0)
        expected_counts = self._expected_tier_counts(
            population._conf.max_tier, (len(orbs) - num_neg_orbs)
        )

        assert actual_tier_counts == expected_counts

    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(1, 10), (1, 2), (1, 3), (1, 5), (4, 20), (30, 23123)],
    )
    def test_neg_vs_tier_ratio(self, neg_weight, tier_weight):
        """
        Each orb type's share of the pool must match its configured weight,
        to within one orb of the exact proportional split.
        """

        population = self._make(neg_weight=neg_weight, tier_weight=tier_weight)
        orbs = population.create()

        self._assert_proportional(orbs, neg_weight, tier_weight)

    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(3, 1), (4, 1), (5, 1), (5, 2)],
    )
    def test_ratio_when_negative_outweighs_tier(self, neg_weight, tier_weight):
        """Same proportional-share contract, for the weight ordering that used to
        break it: apportioning the remainder by sorting the counts rather than
        ranking the indices inverted the shares whenever the first enabled type
        outweighed the second."""

        population = self._make(neg_weight=neg_weight, tier_weight=tier_weight)
        orbs = population.create()

        self._assert_proportional(orbs, neg_weight, tier_weight)

    @pytest.mark.parametrize("max_tier", [7, 8])
    def test_tier_orb_distribution_at_min_pool_boundary(self, max_tier):
        """
        With only tier orbs enabled, every tier must be represented and the
        surplus orbs (pool size exceeds the tier count) spread as evenly as the
        tier count allows.
        """

        population = self._make(max_tier=max_tier, max_active_orbs=3, neg_enabled=False)
        orbs = population.create()

        counts = Counter(orb.META.TIER for orb in orbs)

        assert set(counts) == set(range(1, max_tier + 1))
        assert sum(counts.values()) == len(orbs)
        assert max(counts.values()) - min(counts.values()) <= 1

    @pytest.mark.parametrize("neg_weight", [1, 2, 3, 5, 20, 30, 23123])
    def test_negative_orbs_fill_min_pool_and_ignores_weight(self, neg_weight):
        population = self._make(neg_weight=neg_weight, tier_enabled=False)
        orbs = population.create()

        assert len(orbs) == population._min_pool_size

    def test_at_least_one_orb_type_must_be_enabled(self):
        population = self._make(neg_enabled=False, tier_enabled=False)

        with pytest.raises(ValueError, match="At least one orb must be enabled"):
            population.create()

    # ================= #
    #     Helpers       #
    # ================= #

    def _assert_proportional(self, orbs, neg_weight, tier_weight):
        counts_actual = [
            sum(1 for orb in orbs if orb.META.TIER == 0),
            sum(1 for orb in orbs if orb.META.TIER != 0),
        ]

        total_weight = neg_weight + tier_weight
        pool_size = len(orbs)
        expected = [
            pool_size * neg_weight / total_weight,
            pool_size * tier_weight / total_weight,
        ]

        for actual, ideal in zip(counts_actual, expected):
            assert abs(actual - ideal) <= 1, (
                f"orb count {actual} is more than 1 away from the "
                f"weight-implied share {ideal:.2f}"
            )

    def _expected_tier_counts(self, num_tiers: int, total_tier_orbs) -> dict[int, int]:
        # gives both quotient and remainder
        base_per_tier, tiers_with_extra = divmod(total_tier_orbs, num_tiers)

        return {
            tier: base_per_tier + (1 if tier <= tiers_with_extra else 0)
            for tier in range(1, num_tiers + 1)
        }

    def _make(
        self,
        max_tier: int = 1,
        max_active_orbs: int = 3,
        neg_enabled: bool = True,
        neg_weight: int = 1,
        tier_enabled: bool = True,
        tier_weight: int = 2,
    ) -> WeightedPopulation:
        conf = get_test_config().world
        orb_factory_conf = update_conf(
            conf.orb_factory_conf,
            {
                "max_tier": max_tier,
                "max_active_orbs": max_active_orbs,
                "types": {
                    "negative": {"enabled": neg_enabled, "weight": neg_weight},
                    "tier": {"enabled": tier_enabled, "weight": tier_weight},
                },
            },
        )

        return WeightedPopulation(
            orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
        )


class TestTierChainPopulation:
    """A chain's pool is the chain: one orb per tier, and nothing weighted."""

    @pytest.mark.parametrize("max_tier", [1, 2, 3, 5, 8])
    def test_one_orb_per_tier(self, max_tier: int):
        conf = get_test_config().world
        orbs = TierChainPopulation(max_tier, conf.tier_orb_conf).create()

        assert len(orbs) == max_tier
        assert [orb.META.TIER for orb in orbs] == list(range(1, max_tier + 1))

    def test_ignores_the_weights_entirely(self):
        """A chain's pool is its tiers. It takes no orb-type configuration at all,
        which is the point: the weights belong to the continuous world, and a
        scenario that needed them tuned per weight would be a different scenario."""

        conf = get_test_config().world
        orbs = TierChainPopulation(3, conf.tier_orb_conf).create()

        assert len(orbs) == 3
        assert all(orb.META.TIER != 0 for orb in orbs)


class TestOrbFactoryDelegation:
    """The factory sets up orb class state and hands pool construction over."""

    def test_uses_the_population_it_was_given(self):
        conf = get_test_config().world
        population = TierChainPopulation(3, conf.tier_orb_conf)

        factory = OrbFactory(
            conf.orb_factory_conf,
            conf.negative_orb_conf,
            conf.tier_orb_conf,
            population,
        )

        orbs = factory.create_orbs()

        assert [o.META.TIER for o in orbs] == [1, 2, 3]
        assert all(isinstance(o, TierOrb) for o in orbs)

    def test_sets_the_class_level_lifespan_the_perception_snapshots(self):
        """Orb lifespan is the grid's Manhattan diameter, held as a class
        attribute shared by every GridWorld in the process, and a perception
        snapshots it at construction. The factory still owns that setup."""

        conf = get_test_config().world
        orb_factory_conf = update_conf(
            conf.orb_factory_conf, {"grid_rows": 7, "grid_cols": 4}
        )

        OrbFactory(
            orb_factory_conf,
            conf.negative_orb_conf,
            conf.tier_orb_conf,
            TierChainPopulation(3, conf.tier_orb_conf),
        ).create_orbs()

        assert BaseOrb._life_span == (7 - 1) + (4 - 1)
