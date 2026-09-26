from collections import Counter

import pytest

from syn_grid.core.orbs.orb_factory import OrbFactory
from tests.utils.config_helpers import get_test_config, update_conf


class TestOrbFactory:
    # ================= #
    #       Init        #
    # ================= #

    @pytest.fixture
    def factory(self) -> OrbFactory:
        conf = get_test_config().world

        return OrbFactory(
            conf.orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
        )

    # ================= #
    #       Tests       #
    # ================= #

    @pytest.mark.parametrize("max_tier", [1, 2, 3, 4, 5, 6])
    def test_create_orbs_fills_to_min_pool_size_with_limited_active_orbs(
        self, max_tier: int
    ):
        factory = self._make_adjusted_factory(max_tier=max_tier, max_active_orbs=3)
        orbs = factory.create_orbs()

        actual_tier_counts = Counter(orb.META.TIER for orb in orbs)
        num_neg_orbs = actual_tier_counts.pop(0)
        expected_tier_counts = self._expected_tier_counts(
            (factory._orb_factory_conf.max_tier), (len(orbs) - num_neg_orbs)
        )

        assert actual_tier_counts == expected_tier_counts
        assert len(orbs) == factory._min_pool_size

    @pytest.mark.parametrize("max_tier", [i for i in range(100, 120)])
    def test_orbs_one_per_tier_after_min_pool_with_limited_active_orbs(
        self, max_tier: int
    ):
        factory = self._make_adjusted_factory(max_tier=max_tier, max_active_orbs=3)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._orb_factory_conf.max_tier + 3

    @pytest.mark.parametrize("max_active_orbs", [i for i in range(1, 10)])
    def test_create_orbs_respects_different_max_active_orbs(self, max_active_orbs: int):
        factory = self._make_adjusted_factory(
            max_tier=1, max_active_orbs=max_active_orbs
        )
        orbs = factory.create_orbs()

        assert len(orbs) == factory._orb_factory_conf.max_active_orbs * 3

    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(1, 10), (1, 2), (1, 3), (1, 5), (4, 20), (30, 23123)],
    )
    def test_tier_orb_counts_follow_weight_ratios(self, neg_weight, tier_weight):
        factory = self._make_adjusted_factory(
            neg_weight=neg_weight, tier_weight=tier_weight
        )
        orbs = factory.create_orbs()

        actual_tier_counts = Counter(orb.META.TIER for orb in orbs)
        num_neg_orbs = actual_tier_counts.pop(0)
        expected_counts = self._expected_tier_counts(
            (factory._orb_factory_conf.max_tier), (len(orbs) - num_neg_orbs)
        )

        assert actual_tier_counts == expected_counts

    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(1, 10), (1, 2), (1, 3), (1, 5), (4, 20), (30, 23123)],
    )
    def test_orb_factory_neg_vs_tier_ratio(self, neg_weight, tier_weight):
        """
        Each orb type's share of the pool must match its configured weight,
        to within one orb of the exact proportional split.
        """

        factory = self._make_adjusted_factory(
            neg_weight=neg_weight, tier_weight=tier_weight
        )
        orbs = factory.create_orbs()

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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "OrbFactory._normalize_counts sorts the count list before distributing "
            "the remainder, which destroys the index-to-orb-type mapping whenever the "
            "first enabled type outweighs the second. Weights end up inverted. "
            "Correct behaviour is largest-remainder apportionment that does not reorder."
        ),
    )
    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(3, 1), (4, 1), (5, 1), (5, 2)],
    )
    def test_orb_factory_ratio_when_negative_outweighs_tier(
        self, neg_weight, tier_weight
    ):
        """Same proportional-share contract, for the weight ordering that breaks it."""

        factory = self._make_adjusted_factory(
            neg_weight=neg_weight, tier_weight=tier_weight
        )
        orbs = factory.create_orbs()

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

    @pytest.mark.parametrize("max_tier", [7, 8])
    def test_tier_orb_distribution_at_min_pool_boundary(self, max_tier):
        """
        With only tier orbs enabled, every tier must be represented and the
        surplus orbs (pool size exceeds the tier count) spread as evenly as the
        tier count allows.
        """

        factory = self._make_adjusted_factory(
            max_tier=max_tier, max_active_orbs=3, neg_enabled=False
        )
        orbs = factory.create_orbs()

        counts = Counter(orb.META.TIER for orb in orbs)

        assert set(counts) == set(range(1, max_tier + 1))
        assert sum(counts.values()) == len(orbs)
        assert max(counts.values()) - min(counts.values()) <= 1

    @pytest.mark.parametrize("neg_weight", [1, 2, 3, 5, 20, 30, 23123])
    def test_negative_orbs_fill_min_pool_and_ignores_weight(self, neg_weight):
        factory = self._make_adjusted_factory(neg_weight=neg_weight, tier_enabled=False)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._min_pool_size

    # ================= #
    #     Helpers       #
    # ================= #

    def _make_adjusted_factory(
        self,
        max_tier: int = 1,
        max_active_orbs: int = 3,
        neg_enabled: bool = True,
        neg_weight: int = 1,
        tier_enabled: bool = True,
        tier_weight: int = 2,
    ) -> OrbFactory:
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

        factory = OrbFactory(
            orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
        )

        return factory

    def _expected_tier_counts(self, num_tiers: int, total_tier_orbs) -> dict[int, int]:
        # gives both quotient and remainder
        base_per_tier, tiers_with_extra = divmod(total_tier_orbs, num_tiers)

        counts = {
            tier: base_per_tier + (1 if tier <= tiers_with_extra else 0)
            for tier in range(1, num_tiers + 1)
        }

        return counts
