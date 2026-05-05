from syn_grid.config.models import OrbFactoryConf
from syn_grid.core.orbs.orb_factory import OrbFactory
from syn_grid.core.orbs.base_orb import BaseOrb
from tests.utils.config_helpers import get_test_config, update_conf

import pytest
from collections import Counter


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
            (factory._conf.max_tier), (len(orbs) - num_neg_orbs)
        )

        assert actual_tier_counts == expected_tier_counts
        assert len(orbs) == factory._min_pool_size

    @pytest.mark.parametrize("max_tier", [i for i in range(100, 120)])
    def test_orbs_one_per_tier_after_min_pool_with_limited_active_orbs(
        self, max_tier: int
    ):
        factory = self._make_adjusted_factory(max_tier=max_tier, max_active_orbs=3)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._conf.max_tier + 3

    @pytest.mark.parametrize("max_active_orbs", [i for i in range(1, 10)])
    def test_create_orbs_respects_different_max_active_orbs(self, max_active_orbs: int):
        factory = self._make_adjusted_factory(
            max_tier=1, max_active_orbs=max_active_orbs
        )
        orbs = factory.create_orbs()

        assert len(orbs) == factory._conf.max_active_orbs * 3

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
            (factory._conf.max_tier), (len(orbs) - num_neg_orbs)
        )

        assert actual_tier_counts == expected_counts

    @pytest.mark.parametrize(
        "neg_weight, tier_weight",
        [(1, 10), (1, 2), (1, 3), (1, 5), (4, 20), (30, 23123)],
    )
    def test_orb_factory_neg_vs_tier_ratio(self, neg_weight, tier_weight):
        factory = self._make_adjusted_factory(
            neg_weight=neg_weight, tier_weight=tier_weight
        )
        orbs = factory.create_orbs()

        counts_actual = [
            sum(1 for orb in orbs if orb.META.TIER == 0),
            sum(1 for orb in orbs if orb.META.TIER != 0),
        ]

        total_weight = neg_weight + tier_weight
        ratios = [neg_weight / total_weight, tier_weight / total_weight]
        counts_expected = factory._scale_ratios_to_counts(ratios)
        counts_expected = factory._ensure_min_pool_size(counts_expected, ratios)

        assert counts_actual == counts_expected

    @pytest.mark.parametrize("max_tier", [7, 8])
    def test_tier_orb_distribution_at_min_pool_boundary(self, max_tier):
        factory = self._make_adjusted_factory(
            max_tier=max_tier, max_active_orbs=3, neg_enabled=False
        )
        orbs = factory.create_orbs()

        expected_counts = self._expected_tier_counts(factory._conf.max_tier + 1, len(orbs))
        if max_tier == 7:
            for tier, count in expected_counts.items():
                assert count == 2 if tier == 0 else 1
        else:
            for count in expected_counts.values():
                assert count == 1

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
