from syn_grid.config.models import OrbFactoryConf
from syn_grid.core.orbs.orb_factory import OrbFactory
from syn_grid.core.orbs.base_orb import BaseOrb
from tests.utils.config_helpers import get_test_config, update_conf

import pytest


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

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4, 5])
    def test_create_orbs_fills_to_min_pool_size_with_limited_active_orbs(
        self, tier: int
    ):
        factory = self._make_adjusted_factory(tier, 3)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._MIN_POOL_SIZE

    @pytest.mark.parametrize("tier", [i for i in range(100, 120)])
    def test_orbs_one_per_tier_after_min_pool_with_limited_active_orbs(self, tier: int):
        factory = self._make_adjusted_factory(tier, 3)
        orbs = factory.create_orbs()

        assert len(orbs) == factory._max_tier + 4

    @pytest.mark.parametrize("max_active_orbs", [0, 1, 2, 3, 4, 5])
    def test_create_orbs_respects_different_max_active_orbs(
        self, max_active_orbs:int
    ):
        factory = self._make_adjusted_factory(1, max_active_orbs)
        orbs = factory.create_orbs()


    # ================= #
    #     Helpers       #
    # ================= #

    def _make_adjusted_factory(self, tier: int, max_active_orbs: int) -> OrbFactory:
        conf = get_test_config().world
        orb_factory_conf = update_conf(
            conf.orb_factory_conf, {"max_tier": tier, "max_active_orbs": max_active_orbs}
        )

        factory = OrbFactory(
            orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
        )

        return factory
