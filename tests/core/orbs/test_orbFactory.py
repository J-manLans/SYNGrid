from syn_grid.config.models import OrbFactoryConf
from syn_grid.core.orbs.orb_factory import OrbFactory
from tests.utils.config_helpers import get_test_config, update_conf

import pytest

class TestOrbFactory:

    @pytest.fixture
    def factory_tuple(self) -> tuple[OrbFactory, OrbFactoryConf]:
        conf = get_test_config().world

        return OrbFactory(conf.orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf), conf.orb_factory_conf

    def test_create_orbs_default(self, factory_tuple: tuple[OrbFactory, OrbFactoryConf]):
        factory, conf = factory_tuple
        orbs = factory.create_orbs()

        assert len(orbs) == conf.max_active_orbs * 3

