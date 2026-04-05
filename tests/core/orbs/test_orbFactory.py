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
    def factory_tuple(self) -> tuple[OrbFactory, OrbFactoryConf]:
        conf = get_test_config().world

        return (
            OrbFactory(
                conf.orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
            ),
            conf.orb_factory_conf,
        )

    # ================= #
    #       Tests       #
    # ================= #

    @pytest.mark.parametrize("max_tier", [0, 1, 2, 3, 4, 5])
    def test_create_orbs_fills_to_min_pool_size_with_limited_active_orbs(
        self, max_tier: int
    ):
        factory = self._make_tier_and_max_active_adjusted_factory(max_tier, 3)
        orbs = factory.create_orbs()

        tier_counts = Counter(orb.meta.tier for orb in orbs if orb.meta.tier > -1)
        assert tier_counts == self._expected_tier_counts(max_tier + 1, factory._MIN_POOL_SIZE - 3)

        assert len(orbs) == factory._MIN_POOL_SIZE

    @pytest.mark.parametrize("tier", [i for i in range(100, 120)])
    def test_orbs_one_per_tier_after_min_pool_with_limited_active_orbs(self, tier: int):
        factory = self._make_tier_and_max_active_adjusted_factory(tier, 3)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._max_tier + 4

    @pytest.mark.parametrize("max_active_orbs", [i for i in range(1, 10)])
    def test_create_orbs_respects_different_max_active_orbs(self, max_active_orbs: int):
        factory = self._make_tier_and_max_active_adjusted_factory(1, max_active_orbs)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._max_active_orbs * 3

    def test_create_orbs_respects_different_weights(
        self, factory_tuple: tuple[OrbFactory, OrbFactoryConf]
    ):
        factory, conf = factory_tuple

        orb_factory_conf = update_conf(
            conf,
            {
                "max_tier": 1,
                "max_active_orbs": 3,
                "types": {
                    "negative": {"enabled": True, "weight": 1},
                    "tier": {"enabled": True, "weight": 11},
                },
            },
        )

        orbs = factory.create_orbs()

        assert len(orbs) == factory._max_active_orbs * 3

    # ================= #
    #     Helpers       #
    # ================= #

    def _make_tier_and_max_active_adjusted_factory(
        self, tier: int, max_active_orbs: int
    ) -> OrbFactory:
        conf = get_test_config().world
        orb_factory_conf = update_conf(
            conf.orb_factory_conf,
            {"max_tier": tier, "max_active_orbs": max_active_orbs},
        )

        factory = OrbFactory(
            orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
        )

        return factory

    def _expected_tier_counts(
        self, num_tiers: int, total_tier_orbs
    ) -> dict[int, int]:
        # gives both quotient and remainder
        base_per_tier, tiers_with_extra = divmod(total_tier_orbs, num_tiers)

        counts = {
            tier: base_per_tier + (1 if tier < tiers_with_extra else 0)
            for tier in range(num_tiers)
        }

        return counts
